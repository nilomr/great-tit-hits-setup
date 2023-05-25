# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from config import DIRS, build_projdir
from PIL import Image
from pykanto.utils.io import load_dataset, with_pbar

from greti.io import load_train_df, save_ml_data
from greti.utils import get_dt

# ──── SETTINGS ─────────────────────────────────────────────────────────────────

# Create project directories and load dataset
dataset_name = "great-tit-hits"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)
dataset = load_dataset(DIRS.DATASET, DIRS)

# ──── MAIN ────────────────────────────────────────────────────────────────────
seed = 42
np.random.seed(seed)

# order dataset.files by index:
if all(dataset.files.sort_index().index == dataset.data.index):
    dataset.files.sort_index(inplace=True)


# ──── EXPORT TRAINING SET ─────────────────────────────────────────────────────

# Create output directories
img_out_dir = DIRS.ML / "all-images"
img_out_dir.mkdir(exist_ok=True)

# Export images
to_export = dataset.files.spectrogram.tolist()
save_ml_data(img_out_dir, to_export)

# count the number of images in the output directory and make sure none are
# missing and none are empty:
assert len(list(img_out_dir.glob("*.jpg"))) == len(
    to_export
), "Some images are missing"

assert all(
    [img.stat().st_size > 0 for img in with_pbar(img_out_dir.glob("*.jpg"))]
), "Some files are empty"


# ──── EXPORT DATAFRAME ─────────────────────────────────────────────────────────

# copy split_df to a new dataframe:
df = dataset.files[["spectrogram"]].copy()
# rename spectrogram to path:
df.rename(columns={"spectrogram": "path"}, inplace=True)

# replace the spectrogram paths with the paths to the images:
df["path"] = dataset.files["spectrogram"].map(
    lambda x: str(img_out_dir / x.stem) + ".jpg"
)

# open an image from split_df.path[0] to check that it worked:
Image.open(df.path[700])

# save the dataframe to csv:
out_dir_df = DIRS.ML / "great-tit-inference.csv"
df.to_csv(out_dir_df)
load_train_df(DIRS, out_dir_df)
