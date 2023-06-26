# ──── DESCRIPTION ─────────────────────────────────────────────────────────────

"""
Creates necessary directories for the dataset.
Zips and splits song data files into <5gb chunks to meet OSF requirements.
Copies relevant files and directories to a bare repo used to share the dataset.
"""


# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from config import DIRS, build_projdir
from pykanto.utils.io import load_dataset
from tqdm import tqdm

# ──── SETTINGS ────────────────────────────────────────────────────────────────

dataset_name = "great-tit-hits"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)

root = DIRS.PROJECT.parent / "great-tit-hits"
data = root / "data"
metadata = data / "metadata"
songs = data / "songs"

for d in [metadata, songs]:
    d.mkdir(parents=True, exist_ok=True)

# ──── DATA FILES ─────────────────────────────────────────────────────────────

# Zip and split song data files in <5gb chunks to meet OSF requirements

zfile = shutil.make_archive(
    f"{songs}/song-files",
    "zip",
    root_dir=songs,
    base_dir=DIRS.SEGMENTED,
    verbose=True,
)

MAX_SIZE = 4 * 1024 * 1024 * 1024  # 4GB in bytes


def split_zip(filename, MAX_SIZE, remove=True):
    with zipfile.ZipFile(filename, "r") as zfile:
        size = 0
        parts = 1
        part_filename = f"{filename}.part{parts}"
        with zipfile.ZipFile(
            part_filename, "w", compression=zfile.compression
        ) as part:
            for info in tqdm(zfile.infolist(), desc="Splitting"):
                if size + info.file_size > MAX_SIZE:
                    parts += 1
                    size = 0
                    part_filename = f"{filename}.part{parts}"
                    part = zipfile.ZipFile(
                        part_filename, "w", compression=zfile.compression
                    )
                part.writestr(info, zfile.read(info.filename))
                size += info.file_size
    if remove:
        os.remove(filename)


split_zip(f"{songs}/song-files.zip", MAX_SIZE, remove=True)

# To join back together:
# cat song-files.zip.part* > sf.zip && zip -FF sf.zip --out song-files.zip


# ──── METADATA ───────────────────────────────────────────────────────────────

to_copy = [
    DIRS.MAIN,
    DIRS.MORPHOMETRICS,
    DIRS.NESTBOXES,
    DIRS.PERIMETER.parent,
    DIRS.ML / "output" / "feature_vectors.csv",
    DIRS.DATASET.parent / (DIRS.DATASET_ID + ".csv"),
]

# copy these files and directories to the metadata folder:
for f in to_copy:
    shutil.copytree(f, metadata / f.name) if f.is_dir() else shutil.copy(
        f, metadata
    )
