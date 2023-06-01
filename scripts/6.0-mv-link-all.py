# ──── DESCRIPTION ─────────────────────────────────────────────────────────────

# This script is used to review the dataset and remove any noise and labelling
# mishaps. It is run manually and carefully, and it should only run once and
# only after the dataset has been labelled.


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

# ──── EXPORT TO A MARKDOWN_TABLE ─────────────────────────────────────────────

# Define the statistics and figures
statistics = {
    "Number of Segmented Notes": n_notes,
    "Number of Songs": n_songs,
    "Number of IDs with Class Labels": ids_wc,
    "Mean Repertoire Size": classes_per_id.mean(),
    "SD Repertoire Size": classes_per_id.std(),
    "Median Repertoire Size": classes_per_id.median(),
    "Number of Unique Classes": dataset.data.class_id.unique().shape[0],
    "Mean Class Size": mean_class_size,
    "SD Class Size": sd_class_size,
    "Median Class Size": median_class_size,
}

# Create a dataframe from the statistics
df = pd.DataFrame(statistics.items(), columns=["Statistic", "Value"])
markdown_table = df.to_markdown(index=False, floatfmt="")
with open(DIRS.DATASET.parent / (DIRS.DATASET_ID + ".md"), "w") as f:
    f.write(markdown_table)


# ──── DRAFT! ──────────────────────────────────────────────────────────────────

# calculate how many birds id'd in more than one year:

# import broods data (DIRS.BROODS) from csv:
main_df = pd.read_csv(DIRS.MAIN)
broods_df = pd.read_csv(DIRS.BROODS)
birds_df = pd.read_csv(DIRS.MORPHOMETRICS)

# caclulate how many birds (father column) were recorded in more than one year
# (recorded == True):
main_df.groupby("father")["recorded"].sum().value_counts()


# column 'pnum' in main_df coincides with ID in dataset.data find how many IDs
# in dataset.data also have a 'father' value in in main_df:

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


sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))

# Plot scatter points with loess fit line
sns.regplot(
    data=main_df,
    x="delay",
    y="total_recordings",
    scatter_kws={"alpha": 0.2, "color": "#8FB8DE"},
    line_kws={"color": "red"},
    lowess=True,
    ci=95,
    ax=ax,
)

fix_aspect_ratio(ax, 1)

ax.set_xlabel("Time from first egg to first recording (days)")
ax.set_ylabel("Total recordings (hours)")

plt.show()


dataset_pnum.groupby("ID").size().to_frame("n_rows").sort_values(
    "n_rows", ascending=False
)


# in the broods_df calculate how many birds appear more than once between years
# [2020, 2021, 2022]:
main_df[main_df.year.isin([2020, 2021, 2022])].groupby("father")[
    "father"
].count().value_counts()

# calculate percentage of rows in broods that have a father, by year:
broods_df.groupby("year")["father"].count() / broods_df.groupby("year")[  # noqa
    "father"
].size()  # noqa

# same but with rows where there is an april_lay_date:
broods_df[broods_df.april_lay_date.notnull()].groupby("year")[
    "father"
].count() / broods_df[broods_df.april_lay_date.notnull()].groupby("year")[
    "father"
].size()


# get the longest entry in broods_df.father
broods_df.father.str.len().max()


# TODO 2: programm. copy all relevant files to the new repository,
# great-tit-hits
