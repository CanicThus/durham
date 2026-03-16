import io
import os
import json
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


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
        index_name: str = "dataset_index.json",        # always at ZIP root

        # Augmentation settings
        augment_mode: str = "None",                    # "None" | "Simple" | "Moderate" | "Hard"

        # Hard-mode parameters (only used when augment_mode="Hard")
        hard_max_crop_frac: float = 0.08,              # fraction of width cropped per side
        hard_erode_kernel: int = 5,                    # alpha erosion kernel size
        hard_noise_sigma: float = 0.05,                # Gaussian noise std (RGB)

        # Coordinate normalisation
        normalize_xy01: bool = True,                   # xy := (x/W, y/H) using solution canvas size
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
        xy = torch.tensor([[t[1], t[2]] for t in rows], dtype=torch.float32) if rows else torch.empty((0, 2), dtype=torch.float32)
        rot = torch.tensor([t[3] for t in rows], dtype=torch.float32) if rows else torch.empty((0,), dtype=torch.float32)
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
        w,h = img.size
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
        xs_in =  ca * x0 + sa * y0 + cx
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
    ) :
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
            if self.augment_mode in ( "Simple", "Moderate", "Hard"):
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


# ----------------------------
# Utilities: padding + masking
# ----------------------------

def pad_stack_sequences(
    seqs: List[torch.Tensor],
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of tensors along dim=0 (sequence length) to max length.
    seqs: list of [Ni, ...]
    Returns:
      padded: [B, Nmax, ...]
      key_padding_mask: [B, Nmax] (True where PAD, False where valid)
    """
    assert len(seqs) > 0
    B = len(seqs)
    lengths = [s.shape[0] for s in seqs]
    Nmax = max(lengths)

    out_shape = (B, Nmax) + tuple(seqs[0].shape[1:])
    padded = seqs[0].new_full(out_shape, pad_value)

    key_padding_mask = torch.ones((B, Nmax), dtype=torch.bool, device=seqs[0].device)  # True=pad
    for i, s in enumerate(seqs):
        n = s.shape[0]
        if n == 0:
            continue
        padded[i, :n] = s
        key_padding_mask[i, :n] = False  # valid
    return padded, key_padding_mask


def batch_fragments_from_collate(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Convert collated list-of-puzzles into padded tensors on device.

    Inputs (from collate):
      rgb:   List[B] of List[Ni] of [3,H,W]
      alpha: List[B] of List[Ni] of [1,H,W]
      xy:    List[B] of [Ni,2]
      rot:   List[B] of [Ni]

    Outputs:
      imgs: [B,Nmax,4,H,W]
      imgs_size: [wh]
      xy:   [B,Nmax,2]
      rot:  [B,Nmax]
      pad_mask: [B,Nmax] True=pad
      meta: List
    """
    B = len(batch["rgb"])
    imgs_list = []
    sizes_list =[]
    xy_list = []
    rot_list = []
    meta_list = []

    for b in range(B):
        rgb_list = batch["rgb"][b]
        alpha_list = batch["alpha"][b]
        size_list = batch["rgb_size"][b]

        if len(rgb_list) == 0:
            H = W = 1
            imgs_list.append(torch.empty((0, 4, H, W), dtype=torch.float32, device=device))
            sizes_list.append(torch.empty((0, 2), dtype=torch.int64, device=device))
            xy_list.append(torch.empty((0, 2), dtype=torch.float32, device=device))
            rot_list.append(torch.empty((0,), dtype=torch.float32, device=device))
            continue

        frags = [torch.cat([rgb_list[i], alpha_list[i]], dim=0) for i in range(len(rgb_list))]  # [4,H,W]
        imgs = torch.stack(frags, dim=0)  # [Ni,4,H,W]
        imgs_list.append(imgs.to(device, non_blocking=True))
        # sizes: stack to [Ni,2] and move to device
        sizes = torch.stack(size_list, dim=0).to(device, non_blocking=True)  # [Ni,2] (W,H)
        sizes_list.append(sizes)

        xy_list.append(batch["xy"][b].to(device, non_blocking=True))
        rot_list.append(batch["rotation_deg"][b].to(device, non_blocking=True))
        meta_list.append(batch["meta"][b])

    imgs_padded, pad_mask = pad_stack_sequences(imgs_list, pad_value=0.0)
    xy_padded, _ = pad_stack_sequences(xy_list, pad_value=0.0)
    rot_padded, _ = pad_stack_sequences(rot_list, pad_value=0.0)
    sizes_padded, _ = pad_stack_sequences(sizes_list, pad_value=0)  # [B,Nmax,2]
    
    return {
        "imgs": imgs_padded,
        "img_sizes": sizes_padded,
        "xy": xy_padded,
        "rot": rot_padded,
        "pad_mask": pad_mask,
        "meta":meta_list
    }
