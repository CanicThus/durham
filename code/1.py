import os, io, re, zipfile, tarfile, json, math, time
from pathlib import Path
import torch
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


DATASET_ZIP=DAFNE_ZIP
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

from UnifiedPuzzleSetZipDataset import make_unified_puzzle_dataloader_zip, batch_fragments_from_collate

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




