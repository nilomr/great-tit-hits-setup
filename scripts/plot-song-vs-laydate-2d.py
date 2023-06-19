import colorsys
import datetime
import typing
from ast import Dict
from locale import normalize
from turtle import st
from typing import Dict, List

import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# add a loess regression line for max_day_diff using seaborn
import seaborn as sns
import snuggs
from config import DIRS, build_projdir
from mpl_toolkits.mplot3d import Axes3D

from greti.plot import fix_aspect_ratio

# ──── MAIN ───────────────────────────────────────────────────────────────────


# Create project directories and load dataset
dataset_name = "great-tit-hits"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)


song_dir = DIRS.DATASET.parent / f"{DIRS.DATASET_ID}.csv"
song_df = pd.read_csv(song_dir, index_col=0, parse_dates=["datetime"])

# Import main dataframe with bird data
main_df = pd.read_csv(DIRS.MAIN)


song_df["year"] = song_df["datetime"].dt.year
song_df["april_day"] = (
    song_df["datetime"] - pd.to_datetime(song_df["year"].astype(str) + "-04-01")
).dt.days + 1


# Calculate the number of rows for each day and ID in song_df
# Group song_df by april_day and ID and count the number of rows in each group
song_counts = (
    song_df.groupby(["april_day", "ID"]).size().reset_index(name="count")
)
song_counts_long = song_counts.melt(
    id_vars=["april_day", "ID"],
    value_vars=["count"],
    var_name="variable",
    value_name="value",
)
merged_df = pd.merge(
    song_counts_long,
    main_df[["april_lay_date", "pnum"]],
    left_on=["ID"],
    right_on=["pnum"],
    how="left",
)

# Calculate the difference between april_day and april_lay_date for each ID
merged_df["day_diff"] = merged_df["april_day"] - merged_df["april_lay_date"]

# count the number of rows for each ID and remove IDs with less than 3 days of data
counts = merged_df.groupby("ID").size()
counts = counts[counts > 2]
merged_df = merged_df[merged_df["ID"].isin(counts.index)]

# remove where day_diff is nan
merged_df = merged_df[~merged_df["day_diff"].isna()]

# 20201O81 is probably the beggining of a second clutch, remove it
merged_df = merged_df[merged_df["ID"] != "20201O81"]

# get the rows with the maximum value for day_diff and arrange them by day_diff
max_day_diff = merged_df.sort_values("day_diff")


# Group merged_df by ID and fit a linear regression to each group
groups = merged_df.groupby("ID")


slopes = {}
for name, group in groups:
    x = group["day_diff"]
    y = group["value"]
    try:
        slope, _ = np.polyfit(x, y, 1)
        slopes[name] = slope
    except:
        slopes[name] = np.nan

# remove entries with NaN slope
slopes = {k: v for k, v in slopes.items() if not np.isnan(v)}
# remove ids with only one entry (slope is NaN) from groups

# normlise the slopes to be between 0 and 1 using numpy
slopes = {k: v / max(slopes.values()) for k, v in slopes.items()}


# create a diverging colormap based on the slopes
cmap = mpcolors.LinearSegmentedColormap.from_list(
    "mycmap", ["#c2244b", "#c2244b", "#1e83cc", "#1e83cc"], N=100
)

# reverse cmap
cmap = cmap.reversed()

# create a dictionary with the color for each ID
color_dict = {}
for name, slope in slopes.items():
    norm = mpcolors.Normalize(vmin=-1, vmax=1)
    color = cmap(norm(slope))
    color_dict[name] = color


bckcol = "#181d21"


fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111)
ax.set_facecolor(bckcol)
plt.gcf().set_facecolor(bckcol)

for i, (name, group) in enumerate(groups):
    x = group["day_diff"]
    y = group["value"]
    slope = slopes[name]
    color = color_dict[name]

    f = np.poly1d(np.polyfit(y, x, 1))
    y_fit = np.linspace(y.min(), y.max(), 100)
    x_fit = f(y_fit)
    ax.plot(
        x_fit,
        y_fit,
        color=color,
        alpha=0.8,
        zorder=i,
        linewidth=2,
        clip_on=False,
    )

# plot the sum of values for each ID
# first, sum the values for each ID
max_day_diff = merged_df.sort_values("value").groupby("ID").tail(1)
# sort by day_diff
max_day_diff = max_day_diff.sort_values("day_diff")

sns.regplot(
    x="day_diff",
    y="value",
    data=merged_df,
    scatter=False,
    color="white",
    lowess=True,
    ax=ax,
)

# add a vertical white line at day_diff = 0
ax.axvline(0, color="white", linewidth=2, zorder=100, alpha=0.8)


# make axes, axes labels, ticks, and axes lines white
for spine in ax.spines.values():
    spine.set_color("white")

ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")

for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_color("white")


# export as svg
plt.savefig(
    DIRS.REPORTS / "figures" / "songs_vs_laydate_2d.svg",
    transparent=True,
    bbox_inches="tight",
)
