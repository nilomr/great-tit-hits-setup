# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from config import DIRS, build_projdir
from PIL import Image
from pykanto.utils.io import load_dataset, with_pbar

from greti.io import load_train_df, save_ml_data
from greti.plot import plot_class_count_histogram
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
# Add datetime of 1h-long recording to each song (to split below)
dataset.data["datetime"] = dataset.data.index.map(get_dt)

plot_class_count_histogram(dataset)

# Calculate class size
class_counts = dataset.data.groupby("class_id").size()

# Get class_ids with at least 10 samples and large classes with more than 100
# samples
classes_for_training = class_counts[class_counts >= 10]
large_classes = classes_for_training[classes_for_training > 100]

# Subsample data for large classues
subsampled_data = dataset.data.loc[
    dataset.data["class_id"].isin(large_classes.index)
]
subsampled_data = subsampled_data.groupby("class_id").apply(
    lambda x: x.sample(n=100, random_state=seed) if len(x) > 100 else x
)
subsampled_data.index = subsampled_data.index.droplevel(0)

# Get counts of each class_id in subsampled data
subsampled_counts = subsampled_data["class_id"].value_counts()

# Find classes present in classes_for_training but not in subsampled_counts
classes_to_add = classes_for_training[
    ~classes_for_training.index.isin(subsampled_counts.index)
]

# Add missing classes to subsampled_data
missing_classes_data = dataset.data.loc[
    dataset.data["class_id"].isin(classes_to_add.index)
]
all_data = pd.concat([subsampled_data, missing_classes_data])

# Plot histogram of counts by class_id using pandas
all_data.groupby("class_id").size().plot.hist(bins=30, log=True, figsize=(8, 4))

# Perform 70:30 split within each class_id based on datetime
train_data = pd.DataFrame()
test_data = pd.DataFrame()

for class_id, group_data in with_pbar(all_data.groupby("class_id")):
    sorted_data = group_data.sort_values("datetime")
    split_idx = int(len(sorted_data) * 0.7)
    train_data = pd.concat([train_data, sorted_data[:split_idx]])
    test_data = pd.concat([test_data, sorted_data[split_idx:]])

# now remove all columns except for the class_id, and rename it to "label"
train_data = train_data[["class_id"]].rename(columns={"class_id": "label"})
test_data = test_data[["class_id"]].rename(columns={"class_id": "label"})

# concatenate the train and test data, adding a column "split" to indicate
# which split the data belongs to (train or validation)
train_data["split"] = "train"
test_data["split"] = "validation"
split_df = pd.concat([train_data, test_data]).sort_values(by=["label", "split"])

# print first 20 rows:
split_df.head(20)

val_indices = split_df[split_df["split"] == "validation"].index
split_df.loc[val_indices, "split"] = "validation"
split_df.loc[val_indices, ["is_query", "is_gallery"]] = True

# change false in ["is_query", "is_gallery"] to na
split_df.loc[:, ["is_query", "is_gallery"]] = split_df.loc[
    :, ["is_query", "is_gallery"]
].replace({False: pd.NA})

# convert the label column to int
label_map = {label: i for i, label in enumerate(split_df.label.unique())}
split_df.label = split_df.label.map(label_map)
split_df.groupby(["label", "split"]).count()


# Save to a dictionary to be used in the ML pipeline
label_map_r = {v: k for k, v in label_map.items()}
DIRS.ML.mkdir(exist_ok=True)
with open(DIRS.ML / "label_map.json", "w") as f:
    json.dump(label_map_r, f)

# order dataset.files by ID:
dataset.files.sort_values(by="ID", inplace=True)

# add a column "path" with the path to the spectrogram from dataset.files to
# split_df, joining based on their shared index (not ID):
split_df = split_df.join(dataset.files["spectrogram"], on=split_df.index)
split_df.rename(columns={"spectrogram": "path"}, inplace=True)


# Minimal checks to make sure everything is ok:
assert (
    len(split_df[split_df.index.map(lambda x: isinstance(x, int))]) == 0
), "Some index is a number, not a string"

assert (
    split_df.groupby(["label", "split"]).count().path > 1
).all(), "Some label split has less than two images"


# ──── EXPORT TRAINING SET ─────────────────────────────────────────────────────

# Create output directories
img_out_dir = DIRS.ML / "images"
img_out_dir.mkdir(exist_ok=True)

# Export images
to_export = split_df.path.tolist()
save_ml_data(img_out_dir, to_export)

# count the number of images in the output directory and make sure none are
# missing and none are empty:
assert len(list(img_out_dir.glob("*.jpg"))) == len(
    split_df
), "Some images are missing"

assert all(
    [img.stat().st_size > 0 for img in with_pbar(img_out_dir.glob("*.jpg"))]
), "Some files are empty"


# ──── EXPORT DATAFRAME ─────────────────────────────────────────────────────────

# copy split_df to a new dataframe:
df = split_df.copy()

# replace the spectrogram paths with the paths to the images:
df["path"] = split_df["path"].map(lambda x: str(img_out_dir / x.stem) + ".jpg")

# open an image from split_df.path[0] to check that it worked:
Image.open(df.path[700])

# save the dataframe to csv:

out_dir_df = DIRS.ML / "great-tit-train.csv"
df.to_csv(out_dir_df)
load_train_df(DIRS, out_dir_df)
