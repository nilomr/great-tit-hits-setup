# ──── DESCRIPTION ─────────────────────────────────────────────────────────────
"""
This script performs various data wrangling and analysis tasks on the dataset.
It cleans and reorders the columns, exports feature vectors derived from the
metric learning model as CSV, removes bad files, updates JSONs, and extracts
various statistics from the data. Finally, it exports the statistics to a
markdown table.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import DIRS, build_projdir
from pykanto.utils.io import load_dataset

from greti.io import convert_arrays_to_lists, remove_bad_files, update_jsons
from greti.plot import fix_aspect_ratio
from greti.wrangle import add_datetime

# ──── SETTINGS ────────────────────────────────────────────────────────────────

print("Running this will take a while - haven't had time to optimize it yet.")
# Create project directories and load dataset
dataset_name = "great-tit-hits"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)
dataset = load_dataset(DIRS.DATASET, DIRS)

# import broods data (DIRS.BROODS) from csv:
main_df = pd.read_csv(DIRS.MAIN)
broods_df = pd.read_csv(DIRS.BROODS)
birds_df = pd.read_csv(DIRS.MORPHOMETRICS)

# ──── CLEAN/REORDER DATASET COLUMNS ───────────────────────────────────────────

dataset.data = dataset.data[
    [
        "ID",
        "class_id",
        "start",
        "end",
        "length_s",
        "bit_rate",
        "sample_rate",
        "lower_freq",
        "upper_freq",
        "max_amplitude",
        "onsets",
        "offsets",
        "silence_durations",
        "unit_durations",
    ]
]

dataset.data = add_datetime(dataset.data)

# ──── EXPORT FEATURE VECTORS AS CSV ───────────────────────────────────────────

print("Reading in feature vectors and exporting as csv...")

feat_v_dir = DIRS.ML / "output" / "feature_vectors.npy"
feat_v = np.load(feat_v_dir)
if dataset.data.shape[0] != feat_v.shape[0]:
    raise ValueError(
        "Number of rows in dataset.data does not match number of "
        "rows in feature vector array"
    )

# import labels
labels = pd.read_csv(DIRS.ML / "labels.csv", index_col=0)
if not np.array_equal(labels.index, dataset.data.index):
    raise ValueError("Index of labels.csv does not match index of dataset.data")

# Export
feat_v_df = pd.DataFrame(feat_v, index=dataset.data.index)
feat_v_df.to_csv(DIRS.ML / "output" / "feature_vectors.csv", index=False)


# ──── SAVE DATASET ────────────────────────────────────────────────────────────

dataset_csv_dir = DIRS.DATASET.parent / (DIRS.DATASET_ID + ".csv")
dataset.data = convert_arrays_to_lists(dataset.data)
dataset.data.to_csv(dataset_csv_dir, encoding="utf8", index=True)

# Save dataset
dataset.save_to_disk()


# ──── UPDATE JSONS & REMOVE BAD FILES ────────────────────────────────────────


# Remove bad segment wavs and jsons (noise, too short, etc.) These are the
# segments that have been removed from the dataset after labelling and
# segmentation

update_jsons(dataset)
remove_bad_files(dataset)
dataset.save_to_disk()


# ──── GET SOME STATS ─────────────────────────────────────────────────────────

n_notes = sum(len(i) for i in dataset.data.onsets.values)
n_songs = len(dataset.data)
ids_wc = len(
    dataset.data.loc[
        dataset.data.class_id == dataset.data.class_id, "ID"
    ].unique()
)
classes_per_id = dataset.data.groupby("ID")["class_id"].nunique()
median_class_size = dataset.data.groupby("class_id")["ID"].count().median()
mean_class_size = dataset.data.groupby("class_id")["ID"].count().mean()
sd_class_size = dataset.data.groupby("class_id")["ID"].count().std()


# ──── ADD HOW MANY VOCALISATIONS WERE RECORDED PER BIRD TO MAIN_DF: ──────────

# get the pnum and father columns from main_df and merge with dataset.data on ID
# and pnum:
main_df_pnum_father = main_df[["pnum", "father"]]
dataset_pnum = dataset.data[["ID"]]
dataset_pnum_father = dataset_pnum.merge(
    main_df_pnum_father, left_on="ID", right_on="pnum", how="left"
)

# how many unique ids have a father value in main_df?
dataset_pnum_father[dataset_pnum_father.father.notnull()].ID.nunique()

# count number of rows per ID in dataset.data and add as column to main_df,
# left on ID and right on pnum:
main_df = main_df.merge(
    dataset_pnum.groupby("ID").size().to_frame("n_vocalisations"),
    left_on="pnum",
    right_on="ID",
    how="left",
)
main_df.n_vocalisations = main_df.n_vocalisations.fillna(0).astype(int)


# add repertoire size column:
# for each ID, count how many unique class_ids there are in dataset.data:
main_df = main_df.merge(
    dataset.data.groupby("ID")["class_id"]
    .nunique()
    .to_frame("repertoire_size"),
    left_on="pnum",
    right_on="ID",
    how="left",
)

# save to csv:
main_df.to_csv(DIRS.MAIN, index=False)


# ──── EXTRACT FURTHER STATS ──────────────────────────────────────────────────

# caclulate how many birds (father column) were recorded in more than one year
# (recorded == True):
main_df.query("year in [2020, 2021, 2022]").groupby("father")[
    "recorded"
].sum().value_counts()

# same but filtering only birds that have more than 0 vocalisations:
main_df.query("year in [2020, 2021, 2022] & n_vocalisations > 0").groupby(
    "father"
)["recorded"].sum().value_counts()


birds_recorded = len(main_df.query("recorded == True"))
birds_with_data = len(main_df.query("n_vocalisations > 0"))
unique_birds_idd = main_df.query("n_vocalisations > 0").father.nunique()
n_times_recorded = (
    main_df.query("year in [2020, 2021, 2022] & n_vocalisations > 0")
    .groupby("father")["recorded"]
    .sum()
    .value_counts()
)
n_times_recorded = [f"{i} year: {n}" for i, n in n_times_recorded.items()]


# ──── EXPORT TO A MARKDOWN_TABLE ─────────────────────────────────────────────

# Define the statistics and figures
statistics = {
    "Number of Segmented Notes": n_notes,
    "Number of Songs": n_songs,
    "Mean Repertoire Size": classes_per_id.mean(),
    "SD Repertoire Size": classes_per_id.std(),
    "Median Repertoire Size": classes_per_id.median(),
    "Number of Unique Classes": dataset.data.class_id.unique().shape[0],
    "Mean Class Size": mean_class_size,
    "SD Class Size": sd_class_size,
    "Median Class Size": median_class_size,
    "Number of nest sites recorded": birds_recorded,
    "Number of nest sites with data": birds_with_data,
    "Number of unique birds with data that were ID'd": unique_birds_idd,
    "Number of times each bird was recorded": n_times_recorded,
}

# Create a dataframe from the statistics
df = pd.DataFrame(statistics.items(), columns=["Statistic", "Value"])
markdown_table = df.to_markdown(index=False, floatfmt="")
with open(DIRS.DATASET.parent / (DIRS.DATASET_ID + ".md"), "w") as f:
    f.write(markdown_table)
