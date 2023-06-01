import colorsys
import datetime
import typing
from ast import Dict
from typing import Dict, List

import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# count the number of rows for each ID and remove IDs with only one row
counts = merged_df.groupby("ID").size()
counts = counts[counts > 1]

# remove IDs with only one row from merged_df
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

colors = ["#508cb8", "#a8337b"]

# Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
# set a washed dark grey background color:
bckcol = "#181d21"
ax.set_facecolor(bckcol)
plt.gcf().set_facecolor(bckcol)

for i, (name, group) in enumerate(groups):
    x = group["april_lay_date"]
    y = group["day_diff"]
    z = group["value"]
    slope = slopes[name]
    color = colors[0] if slope > 0 else colors[1]
    # ax.scatter(x, y, z, color=color, alpha=0.5)
    f = np.poly1d(np.polyfit(y, z, 1))
    y_fit = np.linspace(y.min(), y.max(), 100)
    z_fit = f(y_fit)
    x_fit = np.full_like(z_fit, x.mean())
    ax.plot(
        x_fit,
        y_fit,
        z_fit,
        color=color,
        alpha=0.8,
        zorder=i,
        linewidth=2,
        clip_on=False,
    )


x_scale, y_scale, z_scale = (
    0.4,
    0.9,
    1,
)  # set the scaling factors for each axis

ax.get_proj = lambda: np.dot(
    Axes3D.get_proj(ax), np.diag([x_scale, y_scale, z_scale, 1])
)

# rotate to face the camera more, but keep centered on the origin
ax.azim = 140
ax.dist = 8
ax.elev = 25
ax.set_proj_type("ortho")


# fix_aspect_ratio(ax, 1)

ax.set_xlabel("April day")
ax.set_ylabel("Days from first egg")
ax.set_title(
    "Number of songs per day since laying (colored by slope)",
    pad=20,
    color="white",
)
ax.set_zlabel("Number of Songs")


#  set x axis limits between -30 and 30
ax.set_ylim(-25, 25)
# hide y axis below 0
ax.set_xlim(0, None)

# inverse the x axis
ax.invert_xaxis()
ax.invert_yaxis()


bckcol = "#181d21"
ax.set_facecolor(bckcol)
plt.gcf().set_facecolor(bckcol)

# set the tick labels and font sizes for the x, y, and z axes
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_tick_params(labelsize=10, colors="white")
    axis.label.set_color("white")
    axis.label.set_size(12)

# set the major axes to white
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.line.set_color("white")

# remove the background color
for axis in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    axis.fill = False
# remove the background color

ax.grid(False)

# export as svg
plt.savefig(
    DIRS.REPORTS / "figures" / "songs_vs_laydate.svg",
    transparent=True,
    bbox_inches="tight",
)

plt.show()
