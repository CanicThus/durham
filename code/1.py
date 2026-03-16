import os, io, re, zipfile, tarfile, json, math, time
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from sklearn.cluster import KMeans
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import io
import os
import json
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image








# Paths and dataset locations
DATA_ROOT = "../dataset"         # Root folder containing all datasets
COCOTILES_ZIP = os.path.join(DATA_ROOT, "CocoTiles.zip")
DAFNE_ZIP     = os.path.join(DATA_ROOT, "Dafne.zip")
OUTPUT_MODEL = "example_username-model.pth"

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Dataset and augmentation settings
AUGMENTATION = "None"                 # Options: "None", "Simple", "Moderate", "Hard"
FRAGMENT_FIXED_SIZE = (24, 24)        # (width, height) in pixels
NORMALIZE_RGB = True

# Training and model configuration
EPOCHS = 10
LEARNING_RATE=3e-4
NUMBER_WORKERS=0
BATCH_SIZE=4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ZIP = DAFNE_ZIP
IS_COCO = True
if IS_COCO:
    DATASET_ZIP=COCOTILES_ZIP

DATASET_MODE= 'None' # Note can be simple ...

# Configuration summary (sanity check)
print("DATA_ROOT:      ", DATA_ROOT)
print("COCOTILES_ZIP:  ", COCOTILES_ZIP)
print("DAFNE_ZIP:      ", DAFNE_ZIP)
print("AUGMENTATION:   ", AUGMENTATION)
print("EPOCHS:         ", EPOCHS)
print("FRAGMENT_SIZE:  ", FRAGMENT_FIXED_SIZE)
print("NORMALIZE_RGB:  ", NORMALIZE_RGB)
print("DEVICE:         ", DEVICE)
print("================")
print("RUNNING ON      ",DATASET_ZIP)


class UnifiedPuzzleSetZipDataset(Dataset):
    """
    Unified set-of-fragments dataset backed by a *flat-root* ZIP archive.

    Key features:
    - One ZIP with splits: train/ val/ test/
    - Each puzzle contains:
        parameters.txt     (small text metadata; parsed + cached per puzzle)
        fragments.txt      (fragment ids, xy, rotation; parsed + cached per puzzle)
        fragments/*.png    (fragment RGBA images; read per __getitem__)
        rebuilt_image.png
        groundtruth.png

    Augmentation policy (applied inside __getitem__ before returning):
      - augment_mode="None"    : return data exactly as stored in the ZIP (default; aligns with Dafne).
      - augment_mode="Simple"  : CocoTiles => force rotations to 0; Dafne => force rotations to 0.
      - augment_mode="Moderate": CocoTiles => random rotation in {0,90,180,270} per fragment, update labels.
                                Dafne     => ERROR (rotations are already included in data).
      - augment_mode="Hard"    : CocoTiles => Moderate + simulated border damage (crop/pad), alpha erosion, noise.
                                Dafne     => ERROR (rotations are already included in data).

    XY normalisation:
      - If normalize_xy01=True, xy is normalised to [0,1] by dividing by the *solution canvas size* (W,H).
      - (W,H) are read from parameters.txt using:
            solution_width + solution_height
         or canvas_width + canvas_height
         or reference_size (square)
         or square_size    (square)
         or original_width + original_height
      - If none are present and normalize_xy01=True, a clear error is raised.
    """

    def __init__(
            self,
            zip_path: str,
            split: str,
            fixed_size: Optional[Tuple[int, int]] = None,  # (H,W) or None
            normalize_rgb: bool = False,
            return_optional_images: bool = False,
            index_name: str = "dataset_index.json",  # always at ZIP root

            # Augmentation settings
            augment_mode: str = "None",  # "None" | "Simple" | "Moderate" | "Hard"

            # Hard-mode parameters (only used when augment_mode="Hard")
            hard_max_crop_frac: float = 0.08,  # fraction of width cropped per side
            hard_erode_kernel: int = 5,  # alpha erosion kernel size
            hard_noise_sigma: float = 0.05,  # Gaussian noise std (RGB)

            # Coordinate normalisation
            normalize_xy01: bool = True,  # xy := (x/W, y/H) using solution canvas size
    ):
        super().__init__()
        assert split in ("train", "val", "test")

        self.zip_path = zip_path
        self.split = split
        self.fixed_size = fixed_size
        self.normalize_rgb = normalize_rgb
        self.return_optional_images = return_optional_images
        self.index_name = index_name

        self.augment_mode = augment_mode
        if self.augment_mode not in ("None", "Simple", "Moderate", "Hard"):
            raise ValueError("augment_mode must be one of: 'None', 'Simple', 'Moderate', 'Hard'")
        self.unbake_val_rotations = (
                split in ("val", "test")
                and self.augment_mode in ("Simple")
        )

        # Resize fragments (if requested)
        self._resize = transforms.Resize(
            fixed_size,
            interpolation=transforms.InterpolationMode.BILINEAR
        ) if fixed_size else None

        # Convert PIL->Tensor in [0,1]
        self._to_tensor = transforms.ToTensor()

        # Optional ImageNet normalisation for RGB channels only
        self._norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ) if normalize_rgb else None

        # Hard-mode parameter validation
        if not (0.0 <= hard_max_crop_frac <= 0.5):
            raise ValueError("hard_max_crop_frac must be in [0, 0.5]")
        if hard_erode_kernel < 1:
            raise ValueError("hard_erode_kernel must be >= 1")
        if hard_noise_sigma < 0:
            raise ValueError("hard_noise_sigma must be >= 0")

        self.hard_max_crop_frac = hard_max_crop_frac
        self.hard_erode_kernel = hard_erode_kernel
        self.hard_noise_sigma = hard_noise_sigma

        self.normalize_xy01 = normalize_xy01

        # One ZipFile per process (DataLoader workers must not share handles)
        self._zip_by_pid: Dict[int, zipfile.ZipFile] = {}

        # Cache small text files per process
        self._text_cache: Dict[str, str] = {}

        # Cache parsed per-puzzle metadata per process (prevents repeated parameters/fragments reads)
        # Key: puzzle_prefix, e.g. "train/CocoTiles_006001"
        self._puzzle_meta_cache: Dict[str, Dict[str, Any]] = {}

        # Read index once in the main process
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            if self.index_name not in set(zf.namelist()):
                raise FileNotFoundError(f"Missing {self.index_name} at ZIP root (flat ZIP expected).")
            index = json.loads(zf.read(self.index_name).decode("utf-8", errors="replace"))

        if "splits" not in index or split not in index["splits"]:
            raise ValueError(
                f"{self.index_name} must contain key 'splits' with '{split}' list.\n"
                f"Got keys: {list(index.keys())}"
            )

        self.puzzle_names: List[str] = index["splits"][self.split]
        if not self.puzzle_names:
            raise RuntimeError(f"No puzzles listed for split='{self.split}' in {self.index_name}")

    def _zip(self) -> zipfile.ZipFile:
        """Return a per-process ZipFile handle (safe for multi-worker DataLoader usage)."""
        pid = os.getpid()
        if pid not in self._zip_by_pid:
            self._zip_by_pid[pid] = zipfile.ZipFile(self.zip_path, "r")
        return self._zip_by_pid[pid]

    def __len__(self) -> int:
        return len(self.puzzle_names)

    # -----------------------------
    # Parsing conventions
    # -----------------------------

    @staticmethod
    def _puzzle_id6_from_puzzle_name(puzzle_name: str) -> str:
        """Extract trailing 6-digit puzzle id from 'Name_XXXXXX'."""
        if len(puzzle_name) < 7 or puzzle_name[-7] != "_" or not puzzle_name[-6:].isdigit():
            raise ValueError(f"PuzzleName must end with _XXXXXX (6 digits), got: '{puzzle_name}'")
        return puzzle_name[-6:]

    @staticmethod
    def _dataset_name_from_puzzle_name(puzzle_name: str) -> str:
        """Extract dataset name prefix from 'Dataset_XXXXXX'."""
        if "_" not in puzzle_name:
            raise ValueError(f"PuzzleName must contain an underscore, got: '{puzzle_name}'")
        return puzzle_name.rsplit("_", 1)[0]

    @staticmethod
    def _num_pieces6_from_parameters(parameters_txt: str) -> str:
        """Parse 'num_pieces <int>' and return zero-padded 6-digit string."""
        num_pieces = None
        for line in parameters_txt.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[0] == "num_pieces":
                num_pieces = int(parts[1])
                break
        if num_pieces is None:
            raise ValueError("parameters.txt missing required line: 'num_pieces <int>'")
        return f"{num_pieces:06d}"

    @staticmethod
    def _parse_fragments_txt(fragments_txt: str) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Parse fragments.txt lines:
            <fragment_id> <x> <y> <rotation_deg>
        Returns:
            fragment_ids: list[int]
            xy:  [N,2] float32
            rot: [N]   float32
        """
        rows: List[Tuple[int, float, float, float]] = []
        for line in fragments_txt.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 4:
                continue
            fid = int(toks[0])
            x = float(toks[1])
            y = float(toks[2])
            r = float(toks[3])
            rows.append((fid, x, y, r))

        rows.sort(key=lambda t: t[0])
        fragment_ids = [t[0] for t in rows]
        xy = torch.tensor([[t[1], t[2]] for t in rows], dtype=torch.float32) if rows else torch.empty((0, 2),
                                                                                                      dtype=torch.float32)
        rot = torch.tensor([t[3] for t in rows], dtype=torch.float32) if rows else torch.empty((0,),
                                                                                               dtype=torch.float32)
        return fragment_ids, xy, rot

    # -----------------------------
    # parameters.txt parsing: solution canvas size
    # -----------------------------

    @staticmethod
    def _parse_parameters_kv(parameters_txt: str) -> Dict[str, str]:
        """Parse simple 'key value' lines into a dict."""
        kv: Dict[str, str] = {}
        for line in parameters_txt.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                k = parts[0]
                v = " ".join(parts[1:])
                kv[k] = v
        return kv

    def _get_solution_canvas_wh(self, parameters_txt: str) -> Tuple[float, float]:
        """
        Extract solution/canvas size (W,H) from parameters.txt.

        Accepted keys:
          - solution_width, solution_height
          - canvas_width, canvas_height
          - reference_size (square canvas)
          - square_size (square canvas)
          - original_width, original_height

        Raises ValueError if none are found.
        """
        kv = self._parse_parameters_kv(parameters_txt)

        def get_float(key: str) -> Optional[float]:
            if key not in kv:
                return None
            try:
                return float(kv[key])
            except Exception:
                return None

        if "solution_width" in kv and "solution_height" in kv:
            W = get_float("solution_width")
            H = get_float("solution_height")
            if W and H and W > 0 and H > 0:
                return W, H

        if "canvas_width" in kv and "canvas_height" in kv:
            W = get_float("canvas_width")
            H = get_float("canvas_height")
            if W and H and W > 0 and H > 0:
                return W, H

        for k in ("reference_size", "square_size"):
            if k in kv:
                S = get_float(k)
                if S and S > 0:
                    return S, S

        if "original_width" in kv and "original_height" in kv:
            W = get_float("original_width")
            H = get_float("original_height")
            if W and H and W > 0 and H > 0:
                return W, H

        raise ValueError(
            "Could not determine solution canvas size for xy normalisation. "
            "Add one of: "
            "(solution_width & solution_height), (canvas_width & canvas_height), "
            "reference_size, square_size, (original_width & original_height)."
        )

    def _get_solution_canvas_wh(self, parameters_txt: str) -> Tuple[float, float]:
        """
        Extract solution/canvas size (W,H) from parameters.txt.

        Accepted keys:
          - solution_width, solution_height
          - canvas_width, canvas_height
          - reference_size (square canvas)
          - square_size (square canvas)
          - original_width, original_height

        Raises ValueError if none are found.
        """
        kv = self._parse_parameters_kv(parameters_txt)

        def get_float(key: str) -> Optional[float]:
            if key not in kv:
                return None
            try:
                return float(kv[key])
            except Exception:
                return None

        if "solution_width" in kv and "solution_height" in kv:
            W = get_float("solution_width")
            H = get_float("solution_height")
            if W and H and W > 0 and H > 0:
                return W, H

        if "canvas_width" in kv and "canvas_height" in kv:
            W = get_float("canvas_width")
            H = get_float("canvas_height")
            if W and H and W > 0 and H > 0:
                return W, H

        for k in ("reference_size", "square_size"):
            if k in kv:
                S = get_float(k)
                if S and S > 0:
                    return S, S

        if "original_width" in kv and "original_height" in kv:
            W = get_float("original_width")
            H = get_float("original_height")
            if W and H and W > 0 and H > 0:
                return W, H

        raise ValueError(
            "Could not determine solution canvas size for xy normalisation. "
            "Add one of: "
            "(solution_width & solution_height), (canvas_width & canvas_height), "
            "reference_size, square_size, (original_width & original_height)."
        )

    @staticmethod
    def _normalize_xy_by_canvas(xy: torch.Tensor, W: float, H: float) -> torch.Tensor:
        """Normalise xy to [0,1] by dividing by solution canvas (W,H)."""
        if xy.numel() == 0:
            return xy
        out = xy.clone()
        out[:, 0] = out[:, 0] / float(W)
        out[:, 1] = out[:, 1] / float(H)
        return out

    # -----------------------------
    # ZIP reading helpers
    # -----------------------------

    def _read_text(self, member: str) -> str:
        """Read and cache a small text file from inside the ZIP (per process)."""
        if member in self._text_cache:
            return self._text_cache[member]

        zf = self._zip()
        try:
            raw = zf.read(member)
        except KeyError:
            raise FileNotFoundError(f"Missing ZIP member: {member}")

        txt = raw.decode("utf-8", errors="replace")
        self._text_cache[member] = txt
        return txt

    def _read_rgba_png(self, member: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read RGBA PNG from ZIP => rgb [3,H,W], alpha [1,H,W]."""
        zf = self._zip()
        try:
            raw = zf.read(member)
        except KeyError:
            raise FileNotFoundError(f"Missing fragment image member: {member}")

        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert("RGBA")
        w, h = img.size
        size = torch.tensor([w, h], dtype=torch.int64)

        if self._resize is not None:
            img = self._resize(img)

        rgba = self._to_tensor(img)  # [4,H,W] in [0,1]
        rgb = rgba[:3]
        alpha = rgba[3:4]

        if self._norm is not None:
            rgb = self._norm(rgb)

        return rgb, alpha, size

    def _read_optional_rgb(self, member: str) -> Optional[torch.Tensor]:
        """Read an optional RGB image (rebuilt_image/groundtruth) if present."""
        zf = self._zip()
        try:
            raw = zf.read(member)
        except KeyError:
            return None

        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert("RGB")

        if self._resize is not None:
            img = self._resize(img)

        return transforms.ToTensor()(img)

    def _fragment_member_path(
            self,
            split: str,
            puzzle_name: str,
            dataset_name: str,
            puzzle_id6: str,
            num_pieces6: str,
            fragment_id: int
    ) -> str:
        """Build canonical fragment path for a given puzzle and fragment id."""
        frag_id6 = f"{fragment_id:06d}"
        filename = f"{dataset_name}_{puzzle_id6}_{num_pieces6}_{frag_id6}.png"
        return f"{split}/{puzzle_name}/fragments/{filename}"

    @staticmethod
    def _rotate_chw_about_center_pixelgrid(
            x: torch.Tensor,
            angle_deg_ccw: float,
            mode: str = "bilinear",
            padding_mode: str = "zeros",
    ) -> torch.Tensor:
        """
        Rotate [C,H,W] by angle_deg_ccw about the true image centre using an explicit pixel grid.
        This avoids aspect-ratio / affine_grid pitfalls.
        """
        assert x.ndim == 3, f"Expected CHW, got {tuple(x.shape)}"
        C, H, W = x.shape
        if abs(angle_deg_ccw) < 1e-8:
            return x

        device = x.device
        dtype = x.dtype

        # pixel-centre convention
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        a = math.radians(angle_deg_ccw)
        ca, sa = math.cos(a), math.sin(a)

        # grid of destination pixel coords
        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

        # shift to centre, rotate (inverse mapping for sampling), shift back
        x0 = xx - cx
        y0 = yy - cy

        # grid_sample expects: for each output location, where to sample from input.
        # To rotate the image CCW, we sample from input using the inverse rotation (-a).
        # That corresponds to:
        xs_in = ca * x0 + sa * y0 + cx
        ys_in = -sa * x0 + ca * y0 + cy

        # normalize to [-1,1] for align_corners=False
        # align_corners=False maps x in [0,W-1] to [-1+1/W, 1-1/W] effectively.
        # The correct normalization is: x_n = (2*(x + 0.5)/W) - 1
        x_n = (2.0 * (xs_in + 0.5) / W) - 1.0
        y_n = (2.0 * (ys_in + 0.5) / H) - 1.0

        grid = torch.stack([x_n, y_n], dim=-1).unsqueeze(0)  # [1,H,W,2]

        xb = x.unsqueeze(0)  # [1,C,H,W]
        yb = F.grid_sample(
            xb,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )
        return yb.squeeze(0)

    @staticmethod
    def _unbake_rotations_any(
            rgb_list,
            alpha_list,
            rot_deg: torch.Tensor,
            baked_ccw_sign: float = +1.0,
    ):
        if rot_deg.numel() == 0:
            return rgb_list, alpha_list

        out_rgb, out_alpha = [], []
        for i in range(len(rgb_list)):
            r = float(rot_deg[i].item()) * float(baked_ccw_sign)

            rgba = torch.cat([rgb_list[i], alpha_list[i]], dim=0)  # [4,H,W]
            rgba_u = UnifiedPuzzleSetZipDataset._rotate_chw_about_center_pixelgrid(
                rgba, angle_deg_ccw=-r, mode="bilinear", padding_mode="zeros"
            )

            out_rgb.append(rgba_u[:3])
            out_alpha.append(rgba_u[3:4].clamp(0.0, 1.0))

        return out_rgb, out_alpha

    # -----------------------------
    # Per-puzzle metadata caching
    # -----------------------------

    def _get_puzzle_meta(self, puzzle_prefix: str, puzzle_name: str) -> Dict[str, Any]:
        """
        Load+parse per-puzzle metadata once per process and cache it.

        Cached items:
          - parameters_txt
          - dataset_name, puzzle_id6, num_pieces6
          - fragment_ids, xy (possibly normalised), rot
          - canvas_W, canvas_H (if normalize_xy01=True)
        """
        if puzzle_prefix in self._puzzle_meta_cache:
            return self._puzzle_meta_cache[puzzle_prefix]

        parameters_txt = self._read_text(f"{puzzle_prefix}/parameters.txt")
        fragments_txt = self._read_text(f"{puzzle_prefix}/fragments.txt")

        dataset_name = self._dataset_name_from_puzzle_name(puzzle_name)
        puzzle_id6 = self._puzzle_id6_from_puzzle_name(puzzle_name)
        num_pieces6 = self._num_pieces6_from_parameters(parameters_txt)

        fragment_ids, xy, rot = self._parse_fragments_txt(fragments_txt)

        canvas_W = canvas_H = None
        if self.normalize_xy01:
            canvas_W, canvas_H = self._get_solution_canvas_wh(parameters_txt)
            xy = self._normalize_xy_by_canvas(xy, canvas_W, canvas_H)

        meta = {
            "parameters_txt": parameters_txt,
            "dataset_name": dataset_name,
            "puzzle_id6": puzzle_id6,
            "num_pieces6": num_pieces6,
            "fragment_ids": fragment_ids,
            "xy": xy,
            "rot": rot,
            "canvas_W": canvas_W,
            "canvas_H": canvas_H,
        }
        self._puzzle_meta_cache[puzzle_prefix] = meta
        return meta

    # -----------------------------
    # CocoTiles augmentations
    # -----------------------------

    @staticmethod
    def _rot90_tensor(x: torch.Tensor, k: int) -> torch.Tensor:
        """Rotate a CHW tensor by k*90 degrees counter-clockwise."""
        k = int(k) % 4
        if k == 0:
            return x
        return torch.rot90(x, k=k, dims=(-2, -1))

    @staticmethod
    def _random_border_crop_and_pad(rgba: torch.Tensor, max_crop: int) -> torch.Tensor:
        """
        Crop a random number of pixels from each side (<= max_crop), then pad back to original size.
        Operates on RGBA so RGB and alpha remain consistent.
        """
        _, H, W = rgba.shape
        if max_crop <= 0 or H < 4 or W < 4:
            return rgba

        t = int(torch.randint(0, max_crop + 1, (1,)).item())
        b = int(torch.randint(0, max_crop + 1, (1,)).item())
        l = int(torch.randint(0, max_crop + 1, (1,)).item())
        r = int(torch.randint(0, max_crop + 1, (1,)).item())

        if (t + b) >= H - 2 or (l + r) >= W - 2:
            return rgba

        cropped = rgba[:, t:H - b, l:W - r]
        pad = (l, r, t, b)  # left, right, top, bottom
        return F.pad(cropped, pad, mode="constant", value=0.0)

    @staticmethod
    def _erode_alpha(alpha: torch.Tensor, k: int) -> torch.Tensor:
        """Morphological erosion on alpha using max-pooling on inverted alpha."""
        if k <= 1:
            return alpha
        if k % 2 == 0:
            k += 1
        inv = 1.0 - alpha
        inv_dil = F.max_pool2d(inv, kernel_size=k, stride=1, padding=k // 2)
        return (1.0 - inv_dil).clamp(0.0, 1.0)

    @staticmethod
    def _add_noise(rgb: torch.Tensor, alpha: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add Gaussian noise to RGB, masked by alpha to avoid noisy background."""
        if sigma <= 0:
            return rgb
        noise = torch.randn_like(rgb) * sigma
        return rgb + noise * alpha

    def _apply_cocotiles_augment(
            self,
            rgb_list: List[torch.Tensor],
            alpha_list: List[torch.Tensor],
            rot: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Apply synthetic augmentation to CocoTiles fragments and update rotation labels.
        """
        mode = self.augment_mode

        if mode == "Simple":
            return rgb_list, alpha_list, torch.zeros_like(rot)

        # Moderate/Hard: choose per-fragment rotation multiples of 90°
        ks = torch.randint(0, 4, (len(rgb_list),), dtype=torch.int64)
        rot_out = (rot + ks.float() * 90.0) % 360.0

        out_rgb: List[torch.Tensor] = []
        out_alpha: List[torch.Tensor] = []

        for i in range(len(rgb_list)):
            rgb_i = rgb_list[i]
            a_i = alpha_list[i]

            k = int(ks[i].item())
            rgb_i = self._rot90_tensor(rgb_i, k)
            a_i = self._rot90_tensor(a_i, k)

            if mode == "Hard":
                rgba = torch.cat([rgb_i, a_i], dim=0)  # [4,H,W]

                # Border damage
                max_crop_px = max(1, int(self.hard_max_crop_frac * rgba.shape[-1]))
                rgba = self._random_border_crop_and_pad(rgba, max_crop=max_crop_px)

                rgb_i = rgba[:3]
                a_i = rgba[3:4]

                # Alpha erosion + noise
                a_i = self._erode_alpha(a_i, k=self.hard_erode_kernel)
                rgb_i = self._add_noise(rgb_i, a_i, sigma=self.hard_noise_sigma)

                # Safety clamps (esp. useful if rgb is ImageNet-normalised)
                rgb_i = rgb_i.clamp(-5.0, 5.0)
                a_i = a_i.clamp(0.0, 1.0)

            out_rgb.append(rgb_i)
            out_alpha.append(a_i)

        return out_rgb, out_alpha, rot_out

    # -----------------------------
    # Dataset API
    # -----------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        puzzle_name = self.puzzle_names[idx]
        puzzle_prefix = f"{self.split}/{puzzle_name}"

        # Read+parse per-puzzle metadata from cache (fast after first access in each worker)
        pm = self._get_puzzle_meta(puzzle_prefix, puzzle_name)

        parameters_txt: str = pm["parameters_txt"]
        dataset_name: str = pm["dataset_name"]
        puzzle_id6: str = pm["puzzle_id6"]
        num_pieces6: str = pm["num_pieces6"]
        fragment_ids: List[int] = pm["fragment_ids"]
        xy: torch.Tensor = pm["xy"]
        rot: torch.Tensor = pm["rot"]
        canvas_W = pm["canvas_W"]
        canvas_H = pm["canvas_H"]

        # Load all fragment images (dominant cost; unavoidable for pixel data)
        rgb_list: List[torch.Tensor] = []
        rdb_size_list: List[torch.Tensor] = []
        alpha_list: List[torch.Tensor] = []
        member_list: List[str] = []

        for fid in fragment_ids:
            member = self._fragment_member_path(self.split, puzzle_name, dataset_name, puzzle_id6, num_pieces6, fid)
            rgb, alpha, size = self._read_rgba_png(member)
            rgb_list.append(rgb)
            rdb_size_list.append(size)
            alpha_list.append(alpha)
            member_list.append(member)

        # Apply augmentations BEFORE returning, so consumers see already-augmented tensors + updated labels
        is_dafne = (dataset_name.lower() == "dafne")
        is_cocotiles = (dataset_name.lower() == "cocotiles")

        if self.augment_mode == "None":
            pass

        elif is_dafne:
            if self.augment_mode in ("Simple", "Moderate", "Hard"):
                raise ValueError(
                    "augment_mode='Moderate' or 'Hard' is invalid for Dafne: rotations are already present in the dataset."
                )

        elif is_cocotiles:
            rgb_list, alpha_list, rot = self._apply_cocotiles_augment(rgb_list, alpha_list, rot)

        else:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Only 'CocoTiles' and 'Dafne' are supported.")

        # ------------------------------------------------------------
        # Shuffle fragments (ids, images, sizes, alpha, xy, rotation)
        # ------------------------------------------------------------
        N = len(rgb_list)
        if N > 1:
            perm = torch.randperm(N)  # random order each call

            # lists (images/sizes/members)
            rgb_list = [rgb_list[i] for i in perm.tolist()]
            alpha_list = [alpha_list[i] for i in perm.tolist()]
            rdb_size_list = [rdb_size_list[i] for i in perm.tolist()]
            member_list = [member_list[i] for i in perm.tolist()]
            fragment_ids = [fragment_ids[i] for i in perm.tolist()]

            # tensors (labels)
            xy = xy[perm]
            rot = rot[perm]

        out: Dict[str, Any] = {
            "puzzle_key": puzzle_prefix,
            "fragment_ids": torch.tensor(fragment_ids, dtype=torch.int64),
            "rgb": rgb_list,
            "rgb_size": rdb_size_list,
            "alpha": alpha_list,
            "xy": xy,  # [N,2] in [0,1] if normalize_xy01=True
            "rotation_deg": rot,
            "parameters_txt": parameters_txt,
            "meta": {
                "members": member_list,
                "dataset_name": dataset_name,
                "augment_mode": self.augment_mode,

                # Hard-mode parameters
                "hard_max_crop_frac": self.hard_max_crop_frac,
                "hard_erode_kernel": self.hard_erode_kernel,
                "hard_noise_sigma": self.hard_noise_sigma,

                # XY normalisation information
                "xy_normalised": self.normalize_xy01,
                "xy_norm_type": "divide-by-solution-canvas" if self.normalize_xy01 else "none",
                "xy_canvas_W": canvas_W,
                "xy_canvas_H": canvas_H,
            },
        }

        if self.return_optional_images:
            out["rebuilt_image"] = self._read_optional_rgb(f"{puzzle_prefix}/rebuilt_image.png")
            out["groundtruth_image"] = self._read_optional_rgb(f"{puzzle_prefix}/groundtruth.png")

        return out

def collate_puzzle_sets(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Batch puzzles as a list-of-puzzles (variable number of fragments per puzzle).
    """
    out = {
        "puzzle_key": [b["puzzle_key"] for b in batch],
        "fragment_ids": [b["fragment_ids"] for b in batch],
        "rgb": [b["rgb"] for b in batch],
        "rgb_size": [b["rgb_size"] for b in batch],
        "alpha": [b["alpha"] for b in batch],
        "xy": [b["xy"] for b in batch],
        "rotation_deg": [b["rotation_deg"] for b in batch],
        "parameters_txt": [b["parameters_txt"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }
    if "rebuilt_image" in batch[0]:
        out["rebuilt_image"] = [b.get("rebuilt_image") for b in batch]
    if "groundtruth_image" in batch[0]:
        out["groundtruth_image"] = [b.get("groundtruth_image") for b in batch]
    return out


def make_unified_puzzle_dataloader_zip(
    zip_path: str,
    split: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    fixed_size: Optional[Tuple[int, int]] = None,
    normalize_rgb: bool = False,
    return_optional_images: bool = False,

    # Augmentation settings
    augment_mode: str = "None",

    # Hard-mode parameters
    hard_max_crop_frac: float = 0.08,
    hard_erode_kernel: int = 5,
    hard_noise_sigma: float = 0.05,

    # XY normalisation
    normalize_xy01: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the unified ZIP dataset.
    """
    ds = UnifiedPuzzleSetZipDataset(
        zip_path=zip_path,
        split=split,
        fixed_size=fixed_size,
        normalize_rgb=normalize_rgb,
        return_optional_images=return_optional_images,
        augment_mode=augment_mode,
        hard_max_crop_frac=hard_max_crop_frac,
        hard_erode_kernel=hard_erode_kernel,
        hard_noise_sigma=hard_noise_sigma,
        normalize_xy01=normalize_xy01,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_puzzle_sets,
        drop_last=False,
    )

train_loader = make_unified_puzzle_dataloader_zip(
    zip_path=DATASET_ZIP,
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUMBER_WORKERS,
    fixed_size=FRAGMENT_FIXED_SIZE,
    normalize_rgb=NORMALIZE_RGB,
    return_optional_images=False,
    augment_mode=DATASET_MODE
)

val_loader = make_unified_puzzle_dataloader_zip(
    zip_path=DATASET_ZIP,
    split="val",
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUMBER_WORKERS,
    fixed_size=FRAGMENT_FIXED_SIZE,
    normalize_rgb=NORMALIZE_RGB,
    return_optional_images=False,
)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_iterator = iter(cycle(train_loader))
print(f'> Size of training dataset {len(train_loader.dataset)}')


# ==========================================
# 1. 深度学习特征提取模块 (Params < 15M)
# ==========================================
class EdgeFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练 MobileNetV2 (约 3.4M 参数，完全符合要求)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # 截取前7层，保留纹理和局部图案，丢弃高级语义
        self.backbone = nn.Sequential(*list(mobilenet.features.children())[:7])

        # 冻结参数，满足“不得进行模型微调”的约束
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 切换为评估模式
        self.eval()

    def extract_ribbon(self, image, contour_segment, ribbon_length=128, ribbon_width=16):
        """
        将图像边缘沿法线方向展平为矩形带。

        参数:
            image: 原始图像 (H, W, 3)
            contour_segment: 轮廓段点集，形状为 (N, 1, 2) 或 (N, 2)
            ribbon_length: 展平后的矩形长度 (L)
            ribbon_width: 向碎片内部采样的深度 (W)

        返回:
            ribbon: 展平后的矩形图像块，形状为 (ribbon_width, ribbon_length, 3)
        """
        # 1. 整理轮廓点坐标
        pts = contour_segment.reshape(-1, 2)
        if len(pts) < 3:
            # 异常处理：如果轮廓点极少，直接返回零矩阵或做简单处理
            return np.zeros((ribbon_width, ribbon_length, 3), dtype=np.uint8)

        x = pts[:, 0]
        y = pts[:, 1]

        # 2. B-样条曲线平滑与等距重采样
        # s=0表示强制通过所有点，k=min(3, len-1)为样条阶数
        try:
            tck, u = splprep([x, y], s=0.0, k=min(3, len(pts) - 1))
            # 生成 ribbon_length 个均匀分布的参数 t
            u_new = np.linspace(0, 1.0, ribbon_length)
            # 计算新的均匀分布点
            x_new, y_new = splev(u_new, tck)
        except Exception as e:
            # 如果样条拟合失败（如点重合），退化为线性插值
            x_new = np.interp(np.linspace(0, 1, ribbon_length), np.linspace(0, 1, len(x)), x)
            y_new = np.interp(np.linspace(0, 1, ribbon_length), np.linspace(0, 1, len(y)), y)

        # 3. 计算切向量和法向量
        # 使用 numpy.gradient 计算一阶导数（切线方向 dx, dy）
        dx = np.gradient(x_new)
        dy = np.gradient(y_new)

        # 归一化切向量
        magnitude = np.sqrt(dx ** 2 + dy ** 2) + 1e-7  # 加 1e-7 防止除零
        dx /= magnitude
        dy /= magnitude

        # 计算法向量。
        # 假设：cv2.findContours(RETR_EXTERNAL) 返回的是逆时针(CCW)轮廓。
        # 在图像坐标系中(x向右，y向下)，逆时针行进时，左侧(内部)的法向量是 (dy, -dx)。
        # 如果你发现提取出来的带子是全黑的(取到了背景)，将其改为 (-dy, dx) 即可。
        nx = dy
        ny = -dx

        # 4. 构建二维采样网格
        # 生成深度方向的索引 [0, 1, ..., ribbon_width-1]，形状为 (W, 1)
        w_idx = np.arange(ribbon_width)[:, np.newaxis]

        # 利用 Numpy 的广播机制(Broadcasting)生成 (W, L) 的网格坐标
        # map_x[w, l] = x_new[l] + w * nx[l]
        map_x = x_new + w_idx * nx
        map_y = y_new + w_idx * ny

        # 5. 使用 cv2.remap 进行亚像素级的重采样拉伸
        # 使用 cv2.INTER_LINEAR (双线性插值) 保证图像平滑
        # borderMode=cv2.BORDER_CONSTANT 越界部分填黑 (0,0,0)
        ribbon = cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return ribbon

    def forward(self, ribbon_img):
        """输入展平图像，输出 1D L2归一化特征描述符"""
        x = self.transform(ribbon_img).unsqueeze(0)  # Shape: (1, 3, W, L)
        with torch.no_grad():
            features = self.backbone(x)  # Shape: (1, C, H', W')
        pooled = torch.mean(features, dim=[2, 3])  # 全局平均池化: (1, C)
        descriptor = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return descriptor.squeeze(0)


# ==========================================
# 2. 拼图碎片定义模块
# ==========================================
class PuzzlePiece:
    def __init__(self, piece_id, image):
        self.piece_id = piece_id
        self.image = image
        self.height, self.width = image.shape[:2]

        # 提取几何轮廓
        self.contours = self._extract_contours()
        # 计算轮廓的曲率方差 (用于自适应权重)
        self.curvature_variance = self._compute_curvature_variance()

        # 当前位姿状态
        self.x = 0.0
        self.y = 0.0
        self.rotation = 0.0

        # 预计算掩码 (Mask)：非纯黑背景区域为 255，背景为 0
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, 50, 150)
        mask = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=2)
        self.mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)


    def _extract_contours(self):
        """使用二值化和道格拉斯-普克算法提取轮廓"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            epsilon = 0.005 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            return approx
        return np.array([])

    def _compute_curvature_variance(self):
        """
        [启发式特征] 计算边缘转角的方差。
        COCO 直边方差趋近 0；DAFNE 曲边方差较大。
        """
        if len(self.contours) < 3: return 0.0
        pts = self.contours.reshape(-1, 2)
        # 计算相邻边的夹角
        angles = []
        for i in range(len(pts)):
            p_prev = pts[i - 1]
            p_curr = pts[i]
            p_next = pts[(i + 1) % len(pts)]
            # 向量计算
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            # 夹角余弦
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angles.append(angle)
        return np.var(angles)  # 返回角度方差


# ==========================================
# 3. 核心求解器与自适应打分模块
# ==========================================
def transform_contour(contour, center, dx, dy, dr):
    """将轮廓点集按指定的旋转和平移量进行坐标变换"""
    if len(contour) == 0:
        return contour

    pts = contour.reshape(-1, 2).astype(np.float32)
    # 1. 获取旋转矩阵 (注意 OpenCV 中 dr 为角度)
    M = cv2.getRotationMatrix2D(center, dr, 1.0)

    # 2. 将全局平移量加入变换矩阵
    M[0, 2] += dx
    M[1, 2] += dy

    # 3. 齐次坐标变换 (N, 2) -> (N, 3) -> 矩阵相乘 -> (N, 2)
    ones = np.ones((pts.shape[0], 1))
    pts_homo = np.hstack([pts, ones])
    transformed_pts = M.dot(pts_homo.T).T

    return transformed_pts


def extract_longest_contiguous_segment(indices, max_len):
    """
    处理环形数组的连续段提取，解决重合边缘跨越起点(索引0)的问题。
    """
    if len(indices) == 0:
        return []

    idx = np.sort(indices)
    # 计算相邻索引的差值
    diffs = np.diff(idx)
    # 差值大于1的地方就是断点
    jumps = np.where(diffs > 1)[0]

    if len(jumps) == 0:
        return idx.tolist()  # 本身就是连续的一段

    # 根据断点将数组分割成多个连续的子段
    segments = np.split(idx, jumps + 1)

    # 核心：检查首尾段是否在物理上是相连的 (即跨越了 0 和 max_len-1)
    if segments[0][0] == 0 and segments[-1][-1] == max_len - 1:
        # 将最后一段和第一段拼接合并
        wrap_segment = np.concatenate((segments[-1], segments[0]))
        segments[-1] = wrap_segment
        segments.pop(0)

    # 返回其中最长的一段
    longest_segment = max(segments, key=len)
    return longest_segment.tolist()

def rotate_point(pt, center, angle_deg):
    """辅助函数：计算点 pt 绕 center 旋转 angle_deg 后的坐标"""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, y = pt[0] - center[0], pt[1] - center[1]
    nx = x * cos_t - y * sin_t
    ny = x * sin_t + y * cos_t
    return (nx + center[0], ny + center[1])


def get_line_segments(piece):
    """针对 COCO：利用多边形逼近提取线段边缘"""
    segments = []
    # 使用较大的 epsilon 强制提取主要的多边形直线边缘
    epsilon = 0.02 * cv2.arcLength(piece.contours, True)
    approx = cv2.approxPolyDP(piece.contours, epsilon, True)

    pts = approx.reshape(-1, 2)
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]

        # 计算线段中心、长度和角度
        center = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        length = np.linalg.norm(p1 - p2)
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

        # 忽略太短的噪点边缘
        if length > piece.width * 0.1:
            segments.append({
                'center': center, 'length': length, 'angle': angle
            })
    return segments


def get_inflection_points(piece):
    """针对 DAFNE：提取高曲率的拐点及局部切线"""
    points = []
    # 使用较小的 epsilon 保留曲线的显著转折点
    epsilon = 0.005 * cv2.arcLength(piece.contours, True)
    approx = cv2.approxPolyDP(piece.contours, epsilon, True)

    pts = approx.reshape(-1, 2)
    n = len(pts)
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]

        # 计算局部切线方向：使用前后相邻点的连线作为切线近似
        tangent_angle = math.degrees(math.atan2(p_next[1] - p_prev[1], p_next[0] - p_prev[0]))

        points.append({
            'pt': (p_curr[0], p_curr[1]),
            'tangent': tangent_angle
        })
    return points


class PuzzleSolver:
    def __init__(self, feature_extractor):
        self.pieces = []
        self.feature_extractor = feature_extractor
        self.MAX_TOLERATED_OVERLAP = 50  # 允许的最大重叠像素数

    def load_pieces(self, folder_path):
        """加载拼图碎片"""
        self.pieces = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                piece_id = os.path.splitext(filename)[0]
                self.pieces.append(PuzzlePiece(piece_id, img))

    def compute_adaptive_weights(self, piece_a, piece_b):
        """
        动态权重分配：根据曲率方差判断是 COCO 还是 DAFNE
            这里需要更新 有全局变量直接表示是哪个数据集
        """
        avg_var = (piece_a.curvature_variance + piece_b.curvature_variance) / 2.0
        # Sigmoid 映射：方差越大(DAFNE)，alpha(几何权重)越大
        alpha = 1.0 / (1.0 + math.exp(-1.0 * (avg_var - 2.0)))
        beta = 1.0 - alpha
        return alpha, beta

    def compute_overlap_penalty(self, piece_a, piece_b, dx, dy, dr):
        """
        优化版重叠面积计算（解决尺寸截断问题）
        输入：
            piece_a: 已放置的参考碎片
            piece_b: 待匹配的碎片
            dx, dy: 碎片B相对A的平移量
            dr: 碎片B相对A的旋转角度（度）
        输出：
            overlap_area: 重叠像素数
        """
        # 1. 基础校验：掩码为空直接返回严重重叠
        mask_a = piece_a.mask
        mask_b = piece_b.mask
        if mask_a is None or mask_b is None:
            return self.MAX_TOLERATED_OVERLAP + 1  # 视为超阈值重叠
        if np.sum(mask_a) == 0 or np.sum(mask_b) == 0:
            return self.MAX_TOLERATED_OVERLAP + 1

        h_a, w_a = mask_a.shape
        h_b, w_b = mask_b.shape

        # 2. 步骤1：构建碎片B的变换矩阵（旋转+平移）
        # 2.1 绕B自身中心旋转
        center_b = (w_b / 2.0, h_b / 2.0)  # 用浮点数避免整数截断
        M_rot = cv2.getRotationMatrix2D(center_b, dr, 1.0)
        # 2.2 叠加平移量（dx/dy是B相对A的平移）
        M_rot[0, 2] += dx
        M_rot[1, 2] += dy
        M = M_rot  # 最终变换矩阵

        # 3. 步骤2：计算碎片B变换后的完整包围盒（避免截断）
        # 3.1 获取B的四个角点（原始坐标）
        corners_b = np.array([
            [0, 0],  # 左上角
            [w_b, 0],  # 右上角
            [w_b, h_b],  # 右下角
            [0, h_b]  # 左下角
        ], dtype=np.float32).reshape(-1, 1, 2)  # 适配cv2.transform输入格式

        # 3.2 对B的角点做变换，得到变换后的位置
        transformed_corners_b = cv2.transform(corners_b, M)
        transformed_corners_b = transformed_corners_b.reshape(-1, 2)  # 展平为Nx2

        # 3.3 计算变换后B的包围盒范围
        min_x_b = np.min(transformed_corners_b[:, 0])
        max_x_b = np.max(transformed_corners_b[:, 0])
        min_y_b = np.min(transformed_corners_b[:, 1])
        max_y_b = np.max(transformed_corners_b[:, 1])

        # 4. 步骤3：计算碎片A的包围盒范围（A的原始位置）
        # A的左上角是(0,0)，右下角是(w_a, h_a)
        min_x_a, max_x_a = 0, w_a
        min_y_a, max_y_a = 0, h_a

        # 5. 步骤4：计算两个碎片的联合包围盒（统一坐标系）
        min_x = int(np.floor(min(min_x_a, min_x_b)))
        max_x = int(np.ceil(max(max_x_a, max_x_b)))
        min_y = int(np.floor(min(min_y_a, min_y_b)))
        max_y = int(np.ceil(max(max_y_a, max_y_b)))

        # 6. 步骤5：调整变换矩阵，让联合包围盒从(0,0)开始（避免负坐标）
        # 偏移量：将联合包围盒的最小坐标移到(0,0)
        offset_x = -min_x
        offset_y = -min_y

        # 6.1 调整B的变换矩阵（增加偏移量）
        M_adjusted = M.copy()
        M_adjusted[0, 2] += offset_x
        M_adjusted[1, 2] += offset_y

        # 6.2 计算A在联合坐标系中的平移矩阵（A的原始位置 + 偏移量）
        M_a = np.array([
            [1, 0, offset_x],  # x轴偏移
            [0, 1, offset_y]  # y轴偏移
        ], dtype=np.float32)

        # 7. 步骤6：对两个掩码做变换，统一到联合包围盒坐标系
        # 7.1 变换B的掩码（旋转+平移+偏移）
        warped_mask_b = cv2.warpAffine(
            mask_b,
            M_adjusted,
            (max_x - min_x, max_y - min_y),  # 联合包围盒尺寸
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # 7.2 变换A的掩码（仅偏移，无旋转）
        warped_mask_a = cv2.warpAffine(
            mask_a,
            M_a,
            (max_x - min_x, max_y - min_y),  # 联合包围盒尺寸
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # 8. 步骤7：计算重叠像素数
        overlap_mask = cv2.bitwise_and(warped_mask_a, warped_mask_b)
        overlap_area = np.count_nonzero(overlap_mask)

        return overlap_area

    def compute_chamfer_distance(self, contour_a, contour_b, dx, dy, dr):
        # 1. 对contour_b做位姿变换（dx, dy, dr）
        M = cv2.getRotationMatrix2D((0, 0), dr, 1.0)
        M[:, 2] += [dx, dy]
        contour_b_transformed = cv2.transform(contour_b, M)
        # 2. 构建KDTree计算最近邻距离
        pts_a = contour_a.reshape(-1, 2)
        pts_b = contour_b_transformed.reshape(-1, 2)
        tree_a = cKDTree(pts_a)
        tree_b = cKDTree(pts_b)
        dist_a, _ = tree_a.query(pts_b)
        dist_b, _ = tree_b.query(pts_a)
        # 3. 倒角距离 = 双向平均距离
        return (np.mean(dist_a) + np.mean(dist_b)) / 2.0

    def compute_appearance_similarity(self, piece_a, piece_b, dx, dy, dr):
        """利用 MobileNet 提取相交边缘的特征并计算余弦相似度"""
        """利用几何变换精确提取重合边缘段，并计算 MobileNet 视觉特征相似度"""

        # 1. 如果轮廓数据异常，直接返回最低分
        if len(piece_a.contours) == 0 or len(piece_b.contours) == 0:
            return 0.0

        # 2. 将碎片 B 的轮廓变换到 A 的坐标系下
        center_b = (piece_b.width / 2.0, piece_b.height / 2.0)
        transformed_cb = transform_contour(piece_b.contours, center_b, dx, dy, dr)

        # 3. 构建 KD 树以实现微秒级的最近邻搜索
        tree_b = cKDTree(transformed_cb)

        # 4. 寻找 A 中距离 B 的边缘小于阈值 (如 15 像素) 的点
        # distances: A中每个点到B最近点的距离; indices_in_b: 对应的B中的点索引
        pts_a = piece_a.contours.reshape(-1, 2)
        distances, indices_in_b = tree_b.query(pts_a)

        MATCH_THRESHOLD = 15.0  # 容忍 15 像素的对齐误差
        valid_mask = distances < MATCH_THRESHOLD
        valid_indices_a = np.where(valid_mask)[0]

        # 如果重合边缘太短 (例如不足 10 个像素点)，说明这不是有效的拼接缝
        if len(valid_indices_a) < 10:
            return 0.0

        # 5. 提取连续的边缘段 (解决环形越界问题)
        longest_seq_a = extract_longest_contiguous_segment(valid_indices_a, len(pts_a))

        # 如果最长连续段依然太短，直接否决
        if len(longest_seq_a) < 10:
            return 0.0

        # 6. 获取真正的 contour_segment (形状必须保持 (N, 1, 2) 以适配后续提取)
        # piece_a 直接按连续索引切片
        contour_segment_a = piece_a.contours[longest_seq_a]

        # piece_b 按 A 映射过去的索引切片，保证了点对点的完美顺序匹配
        matched_indices_b = indices_in_b[longest_seq_a]
        contour_segment_b = piece_b.contours[matched_indices_b]

        # 2. 提取展平图像带
        ribbon_a = self.feature_extractor.extract_ribbon(piece_a.image, contour_segment_a)
        ribbon_b = self.feature_extractor.extract_ribbon(piece_b.image, contour_segment_b)

        # 3. 提取特征描述符
        feat_a = self.feature_extractor(ribbon_a)
        feat_b = self.feature_extractor(ribbon_b)

        # 4. 计算余弦相似度并归一化到 [0, 1]
        sim = torch.dot(feat_a, feat_b).item()
        return (sim + 1.0) / 2.0

    def score_pose_hypothesis(self, piece_a, piece_b, dx, dy, dr):
        """综合打分函数"""
        # 1. 硬性约束：重叠惩罚
        overlap = self.compute_overlap_penalty(piece_a, piece_b, dx, dy, dr)
        if overlap > self.MAX_TOLERATED_OVERLAP:
            return -9999.0

            # 2. 自适应权重
        alpha, beta = self.compute_adaptive_weights(piece_a, piece_b)

        # 3. 几何得分 (距离越小，得分越接近 1)
        dist = self.compute_chamfer_distance(piece_a.contours, piece_b.contours, dx, dy, dr)
        s_geo = math.exp(-0.5 * dist)

        # 4. 外观得分
        # print(piece_a.piece_id, piece_b.image, dx, dy, dr)
        s_app = self.compute_appearance_similarity(piece_a, piece_b, dx, dy, dr)

        # 5. 总分
        return alpha * s_geo + beta * s_app

    def generate_pose_candidates(self, piece_a, piece_b):
        """
        基于几何特征对齐，生成有限且高质量的候选位姿 (dx, dy, dr) 列表
        """
        candidates = []
        center_b = (piece_b.width / 2.0, piece_b.height / 2.0)

        global IS_COCO
        if IS_COCO:
            # ==========================================
            # MS-COCO 模式：线段匹配
            # ==========================================
            edges_a = get_line_segments(piece_a)
            edges_b = get_line_segments(piece_b)

            for ea in edges_a:
                for eb in edges_b:
                    # 启发式剪枝：只匹配长度相近的边缘 (允许 25% 的误差)
                    len_diff = abs(ea['length'] - eb['length'])
                    max_len = max(ea['length'], eb['length'])
                    if len_diff / max_len > 0.25:
                        continue

                    # 计算旋转角 (使两条边方向相反)
                    dr = (ea['angle'] - eb['angle'] + 180.0) % 360.0

                    # 将 B 的边缘中心点绕 B 的图像中心旋转 dr
                    center_b_rot = rotate_point(eb['center'], center_b, dr)

                    # 计算平移量，使旋转后的 B 边缘中心对齐到 A 边缘中心
                    dx = ea['center'][0] - center_b_rot[0]
                    dy = ea['center'][1] - center_b_rot[1]

                    candidates.append((dx, dy, dr))
        else:
            # ==========================================
            # DAFNE 模式：拐点(曲率极值点)匹配
            # ==========================================
            pts_a = get_inflection_points(piece_a)
            pts_b = get_inflection_points(piece_b)

            # 为了防止 DAFNE 点太多导致候选爆炸，可以限制最多测试组合数
            # 真实场景可按角点尖锐程度（内角大小）进行排序截断
            for pa in pts_a:
                for pb in pts_b:
                    # 计算旋转角 (使局部切线反向)
                    dr = (pa['tangent'] - pb['tangent'] + 180.0) % 360.0

                    # 将 B 的拐点绕 B 的图像中心旋转 dr
                    pb_rot = rotate_point(pb['pt'], center_b, dr)

                    # 计算平移量
                    dx = pa['pt'][0] - pb_rot[0]
                    dy = pa['pt'][1] - pb_rot[1]

                    candidates.append((dx, dy, dr))

                    # 考虑到 DAFNE 不规则，增加一个法线反向的候选 (切线同向，法线相反)
                    dr_alt = (pa['tangent'] - pb['tangent']) % 360.0
                    pb_rot_alt = rotate_point(pb['pt'], center_b, dr_alt)
                    candidates.append((
                        pa['pt'][0] - pb_rot_alt[0],
                        pa['pt'][1] - pb_rot_alt[1],
                        dr_alt
                    ))

        # 去重：过滤掉物理上极其相似的冗余候选位姿
        # 这有助于极大地加快后续 score_pose_hypothesis 的速度
        unique_candidates = []
        for cand in candidates:
            is_duplicate = False
            for u_cand in unique_candidates:
                # 如果平移差小于 5 像素，且旋转差小于 5 度，视为重复
                if (abs(cand[0] - u_cand[0]) < 5 and
                        abs(cand[1] - u_cand[1]) < 5 and
                        min(abs(cand[2] - u_cand[2]), 360 - abs(cand[2] - u_cand[2])) < 5):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_candidates.append(cand)

        if len(unique_candidates) == 0:
            return [(0,0,0)]
        return unique_candidates

    def solve(self):
        """全局拼接：基于贪心策略 (建议后续升级为束搜索 Beam Search)"""
        if not self.pieces: return

        # 基准锚点
        anchor = self.pieces[0]
        anchor.x, anchor.y, anchor.rotation = 0.0, 0.0, 0.0
        placed_pieces = [anchor]
        unplaced_pieces = self.pieces[1:]

        while unplaced_pieces:
            best_score = -float('inf')
            best_match = None
            best_pose = (0, 0, 0)
            if(len(unplaced_pieces))==1:
                print(1)
            for placed in placed_pieces:
                for unplaced in unplaced_pieces:
                    # 获取候选位姿
                    if unplaced.piece_id == "CocoTiles_005001_000016_000001":
                        print(1)
                    candidates = self.generate_pose_candidates(placed, unplaced)
                    for dx, dy, dr in candidates:
                        score = self.score_pose_hypothesis(placed, unplaced, dx, dy, dr)
                        if score > best_score:
                            best_score = score
                            best_match = unplaced
                            best_pose = (placed.x + dx, placed.y + dy, (placed.rotation + dr) % 360)

            if best_match:
                best_match.x, best_match.y, best_match.rotation = best_pose
                placed_pieces.append(best_match)
                unplaced_pieces.remove(best_match)
                print(f"已拼接碎片: {best_match.piece_id}, 得分: {best_score:.4f}，剩余{len(unplaced_pieces)}碎片")

    def export_poses(self, output_filepath):
        """
                按纯文本格式导出位姿，格式为:
                [图片ID] [x位置] [y位置] [旋转角度]
                """
        if not self.pieces:
            return

        # 1. 坐标归一化：找到全局最小的 x 和 y，确保所有导出的坐标都是正数
        min_x = min(p.x for p in self.pieces)
        min_y = min(p.y for p in self.pieces)

        # 为了和示例数据完美对齐（示例中坐标可能是中心点坐标，如 30.0）
        # 如果你的 piece.x 代表左上角，且想避免贴着 0 边界，可以加上一个 offset（可选）
        # 这里我们先直接平移到最小值为 0 的相对坐标系，或者你也可以平移到最小值为 piece.width/2
        offset_x = 0.0
        offset_y = 0.0

        # 2. 按照图片 ID 数字大小进行排序 (保证输出顺序是 0, 1, 2... 15)
        # 假设文件名/piece_id 之前读取的是字符串 "0", "1", "10" 等
        sorted_pieces = sorted(
            self.pieces,
            key=lambda p: int(p.piece_id) if p.piece_id.isdigit() else p.piece_id
        )

        # 3. 写入文件
        with open(output_filepath, 'w') as f:
            for piece in sorted_pieces:
                # 平移坐标
                final_x = piece.x - min_x + offset_x
                final_y = piece.y - min_y + offset_y

                # 确保旋转角度在 0 ~ 360 之间
                final_rot = piece.rotation % 360.0

                # 按照要求的格式格式化字符串，保留 1 位小数
                line = f"{piece.piece_id} {final_x:.1f} {final_y:.1f} {final_rot:.1f}\n"
                f.write(line)

        print(f"位姿数据已按评测格式导出至: {output_filepath}")


def render_assembled_puzzle(solver, output_filepath="assembled_result.png", padding=50):
    """
    根据每个碎片的最终位姿 (x, y, rotation) 渲染完整的拼接结果图。

    参数:
        solver: 运行完 solve() 之后的 PuzzleSolver 实例
        output_filepath: 渲染后保存的文件路径
        padding: 画布四周的留白像素
    """
    if not solver.pieces:
        print("没有可供渲染的碎片！")
        return

    # 1. 第一遍遍历：旋转所有碎片，并计算全局画布的边界
    min_gx, min_gy = float('inf'), float('inf')
    max_gx, max_gy = -float('inf'), -float('inf')

    pieces_data = []  # 缓存旋转后的图像和位置信息，避免二次计算

    for piece in solver.pieces:
        img = piece.image
        mask = piece.mask
        h, w = img.shape[:2]

        # --- 图像无损旋转处理 ---
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, piece.rotation, 1.0)

        # 计算旋转后的新图像尺寸 (Bounding Box)
        cos_a = np.abs(M[0, 0])
        sin_a = np.abs(M[0, 1])
        new_w = int((h * sin_a) + (w * cos_a))
        new_h = int((h * cos_a) + (w * sin_a))

        # 调整旋转矩阵的平移部分，防止图像旋转后超出边界被裁剪
        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]

        # 执行旋转
        rot_img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
        rot_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST)

        # --- 计算在全局坐标系中的物理位置 ---
        # 假设 piece.x 和 piece.y 代表原图左上角在全局坐标系的绝对位置
        # 旋转是绕原图中心进行的，因此全局中心点保持不变
        gx_center = piece.x + w / 2.0
        gy_center = piece.y + h / 2.0

        # 旋转后的图像在全局坐标系下的左上角和右下角
        g_left = gx_center - new_w / 2.0
        g_right = gx_center + new_w / 2.0
        g_top = gy_center - new_h / 2.0
        g_bottom = gy_center + new_h / 2.0

        # 更新全局画布的极值
        min_gx = min(min_gx, g_left)
        max_gx = max(max_gx, g_right)
        min_gy = min(min_gy, g_top)
        max_gy = max(max_gy, g_bottom)

        pieces_data.append({
            'piece_id': piece.piece_id,
            'rot_img': rot_img,
            'rot_mask': rot_mask,
            'g_left': g_left,
            'g_top': g_top,
            'new_w': new_w,
            'new_h': new_h
        })

    # 2. 创建全局画布 (带有 Padding)
    canvas_w = int(max_gx - min_gx) + 2 * padding
    canvas_h = int(max_gy - min_gy) + 2 * padding
    # 创建纯黑背景 (如果想用纯白背景，可以将 0 改为 255)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 3. 第二遍遍历：将每个碎片拼贴到画布上
    for pd in pieces_data:
        if pd["piece_id"] not in ("CocoTiles_005001_000016_000000", "CocoTiles_005001_000016_000002"):
            continue
        # 计算在画布上的实际绘制起点 (减去极小值以处理负坐标，并加上留白)
        x_start = int(pd['g_left'] - min_gx) + padding
        y_start = int(pd['g_top'] - min_gy) + padding
        x_end = x_start + pd['new_w']
        y_end = y_start + pd['new_h']

        # 提取画布上的兴趣区域 (ROI)
        roi = canvas[y_start:y_end, x_start:x_end]

        # 获取当前碎片的图像和掩码
        fg_img = pd['rot_img']
        mask = pd['rot_mask']
        mask_inv = cv2.bitwise_not(mask)

        # 使用位运算进行完美无缝的 Alpha 贴图
        # 把 ROI 中需要放碎片的地方抠黑
        canvas_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # 把碎片中不需要的黑色背景抠掉
        piece_fg = cv2.bitwise_and(fg_img, fg_img, mask=mask)
        # 两者相加贴到画布上
        dst = cv2.add(canvas_bg, piece_fg)
        canvas[y_start:y_end, x_start:x_end] = dst

    # 4. 保存文件并使用 Matplotlib 展示 (适配 Jupyter Notebook)
    cv2.imwrite(output_filepath, canvas)
    print(f"拼图结果已保存至: {output_filepath}")

    # 转换为 RGB 格式用于 Matplotlib 正确显示色彩
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 12))
    plt.imshow(canvas_rgb)
    plt.title("Final Assembled Puzzle")
    plt.axis('off')
    plt.show()

# ==========================================
# 4. 主运行逻辑
# ==========================================
if __name__ == "__main__":
    print("初始化特征提取器 (MobileNetV2)...")
    extractor = EdgeFeatureExtractor()
    solver = PuzzleSolver(extractor)

    dummy_folder = "..\\dataset\\CocoTiles_005001\\fragments"
    if os.path.exists(dummy_folder):
        start_time = time.time()
        print(f"加载数据集: {dummy_folder}")
        solver.load_pieces(dummy_folder)
        print(f"已载入图片{len(solver.pieces)}张")
        print("开始全局拼接搜索...")
        solver.solve()
        print("开始验证")
        solver.export_poses("predictions.json")
        print("渲染并展示最终拼接图...")
        render_assembled_puzzle(solver, "final_assembled.png")
        print(f"拼接完成！耗时: {time.time() - start_time:.2f} 秒")
    else:
        print(f"未找到文件夹: {dummy_folder}，请检查路径。")



