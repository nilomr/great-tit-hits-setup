# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations
import random
from config import DIRS, build_projdir
from pykanto.utils.io import load_dataset

# ──── SETTINGS ─────────────────────────────────────────────────────────────────


# Create project directories and load dataset
dataset_name = "full-dataset-test"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)
dataset = load_dataset(DIRS.DATASET, DIRS)

# Plot some randomly sampled songs
for i in random.sample(range(len(dataset.data.index)), 5):
    dataset.plot(dataset.data.index[i], segmented=True)
