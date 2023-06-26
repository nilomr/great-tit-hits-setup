# ──── DESCRIPTION ──────────────────────────────────────────────────────────────
"""
This script is used to review the dataset and remove any noise and labelling
mishaps. It is run manually and carefully, and it should only run once and
only after the dataset has been labelled.
"""

# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from bokeh.palettes import Category20_20
from config import DIRS, build_projdir
from pykanto.utils.io import load_dataset

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


def plot_repertoire_distribution(classes_per_id):
    """
    Plot the frequency distribution of repertoire sizes.

    Args:
        classes_per_id (pandas.Series): Series containing the number of
        different classes per ID.

    Returns:
        None
    """
    # Create a list of unique class counts and their frequencies
    unique_counts, frequencies = (
        classes_per_id.value_counts().sort_index().reset_index().values.T
    )

    # Plot histogram
    plt.bar(unique_counts, frequencies, color="lightblue")

    plt.xlabel("Repertoire size (n song types)", fontsize=12)
    plt.ylabel("Count (n birds)", fontsize=12)
    plt.title("Frequency distribution of repertoire sizes", fontsize=14)

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Label each integer on the x-axis
    xticks = range(unique_counts.min(), unique_counts.max() + 1)
    plt.xticks(xticks, xticks)

    plt.show()


# ──── SETTINGS ────────────────────────────────────────────────────────────────

# Create project directories and load dataset
dataset_name = "great-tit-hits"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)
dataset = load_dataset(DIRS.DATASET, DIRS)


# ──── MAIN ────────────────────────────────────────────────────────────────────

# Plot some randomly sampled songs
for i in random.sample(range(len(dataset.data.index)), 5):
    dataset.plot(dataset.data.index[i], segmented=True)

# How many segmented notes?
n_notes = sum(len(i) for i in dataset.data.onsets.values)
print(f"There are {n_notes} in the dataset")

# Open app
dataset.open_label_app(palette=Category20_20, max_n_labs=20)

# Save manually reviewed class labels to a separate file
dataset = dataset.reload()
dataset.data["class_label"].to_csv(
    dataset.DIRS.DATASET.parent
    / f'labels_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
)

np.unique(
    dataset.data.loc[
        dataset.data.class_label == dataset.data.class_label, "class_label"
    ]
)

# count number of files with a label
dataset.data.class_label.count()

# save dataset to csv
dataset.to_csv(dataset.DIRS.DATASET.parent)

# ──── # CAREFUL THERE ─────────────────────────────────────────────────────────

proceed = input("Are you sure you want to proceed? (y/n)")
if proceed == "y":
    raise PermissionError("This should be run manually and carefully")
    # Set all checks to false:
    dataset.files.loc[dataset.files.voc_check == True, "voc_check"] = False
    # Set all class labels to nan
    dataset.data.loc[:, "class_label"] = np.nan
    # save dataset
    dataset.save_to_disk()

    # set all checks for 20211SW119 to False
    dataset.files.loc[dataset.files.ID == "20221EX38", "voc_check"] = False
    # set all class labels for 20221EX38 to nan
    dataset.data.loc[dataset.data.ID == "20221EX38", "class_label"] = np.nan


# ──── CLEAN DATASET ───────────────────────────────────────────────────────────

# Now, remove all birds with ss < 15 and audio segments marked as noise
# The 15 threshold is entirely arbitrary, imposed by the clustering and
# labelling process. But we have to cut somewhere.

# find index of all rows where class_label is nan or where class_label is '-1':
nan_idx = dataset.data.loc[
    (dataset.data.class_label != dataset.data.class_label)
].index
noise_idx = dataset.data.loc[(dataset.data.class_label == "-1")].index
to_remove = np.concatenate((nan_idx, noise_idx))

# find to_remove files in dataset.DIRS.SEGMENTED and delete them:
segpaths = list(dataset.DIRS.SEGMENTED.rglob("*.wav"))
segpaths = sorted(segpaths, key=lambda x: x.stem)

for segpath in segpaths:
    if segpath.stem in to_remove:
        segpath.unlink()

# Also delete spectrograms and average units files:
trm = dataset.files.loc[to_remove, ["spectrogram"]].values
trm = [item for sublist in trm for item in sublist]
for path in trm:
    if path.exists():
        path.unlink()

# Remove rows from data and files
dataset.data.drop(to_remove, inplace=True)
dataset.files.drop(to_remove, inplace=True)

# Count and print n ids
print(f"Number of IDs: {dataset.data.ID.nunique()}")

# Rename 'class' to 'class_id'
dataset.data.rename(columns={"class": "class_id"}, inplace=True)

# Save dataset
dataset.save_to_disk()
