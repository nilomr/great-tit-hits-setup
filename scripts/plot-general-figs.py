import geopandas as gpd
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from config import DIRS, build_projdir
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata

from greti.io import load_song_dataframe, prepare_shape_data
from greti.plot import fix_aspect_ratio, plot_nestboxes_and_perimeter


def get_donut_data(song_df, main_df, year: int):
    # find which ID in song_df are not in main_df (pnum):
    missing_ids = (
        song_df[~song_df["ID"].isin(main_df["pnum"])].ID.unique().tolist()
    )
    print(
        f"{len(missing_ids)} PNUM for which there are songs is missing from the brood information"
    )

    outer_d = (
        song_df.drop(song_df[song_df["ID"].isin(missing_ids)].index)
        .query(f"year == {year}")
        .groupby("ID")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    outer_ids, counts = outer_d.ID.tolist(), outer_d.n.tolist()
    print(f"{sum(counts)} songs from {len(outer_ids)} individuals in {year}")

    # Keep only some labels
    labels = [""] * len(counts)
    labels[::10] = counts[::10]
    labels[-1] = counts[-1]

    cmap = plt.colormaps["magma"]

    # Select the rows in main_df corresponding to the IDs in outer_ids
    filtered_df = main_df.loc[main_df["pnum"].isin(outer_ids)]
    num_fledglings = filtered_df["num_fledglings"]
    norm = plt.Normalize(num_fledglings.min(), num_fledglings.max())
    colors = cmap(norm(num_fledglings))

    # palette based on the average length of the song
    song_lengths = song_df.groupby("ID").length_s.mean().loc[outer_ids]
    norm = plt.Normalize(song_lengths.min(), song_lengths.max())
    colors = cmap(norm(song_lengths))

    # Color by number of recordings
    filtered_df = main_df.loc[main_df["pnum"].isin(outer_ids)]
    total_recordings = filtered_df["total_recordings"]
    norm = plt.Normalize(total_recordings.min(), total_recordings.max())
    colors = cmap(norm(total_recordings))

    return counts, labels, colors


def scale_lists(*lists):
    # Calculate the sums of all the input lists
    sums = [sum(lst) for lst in lists]
    # Find the maximum sum
    max_sum = max(sums)
    # Scale each list by dividing each element by the maximum sum
    scaled_lists = [
        [(num / max_sum) - 0.000000001 for num in lst] for lst in lists
    ]

    return scaled_lists


# ──── READ IN THE DATA ───────────────────────────────────────────────────────

# Create project directories and load dataset
dataset_name = "great-tit-hits"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)

# Import song database
song_dir = DIRS.DATASET.parent / f"{DIRS.DATASET_ID}.csv"
song_df = load_song_dataframe(song_dir)
song_df["year"] = song_df["datetime"].dt.year

# Import the bird datasets
main_df = pd.read_csv(DIRS.MAIN)
morpho_df = pd.read_csv(DIRS.MORPHOMETRICS)
broods_df = pd.read_csv(DIRS.BROODS)

# ──── MAIN ───────────────────────────────────────────────────────────────────


# ──── DONUT CHART (SAMPLE SIZES PER INDIVIDUAL) ──────────────────────────────

# counts1, labels1, _ = get_donut_data(song_df, main_df, 2022)
# counts2, labels2, _ = get_donut_data(song_df, main_df, 2021)
# counts3, labels3, _ = get_donut_data(song_df, main_df, 2020)

# counts1, counts2, counts3 = scale_lists(counts1, counts2, counts3)


# fig, ax = plt.subplots()
# ax.axis("equal")
# width = 0.6
# space = 0.05
# radius = 2
# start = 20

# colors = ["#4da9b9", "#3a7b86"]

# pie3, _ = ax.pie(
#     counts3,
#     radius=radius,
#     labels=labels3,
#     colors=colors,
#     wedgeprops={"linewidth": 0},
#     textprops={"fontsize": 7},
#     rotatelabels=True,
#     normalize=False,
#     counterclock=False,
#     startangle=start,
# )
# plt.setp(pie3, width=width)


# colors = ["#b9974d", "#86683a"]
# pie2, _ = ax.pie(
#     counts2,
#     radius=radius - width - space,
#     labeldistance=1 - width,
#     colors=colors,
#     wedgeprops={"linewidth": 0},
#     textprops={"fontsize": 5},
#     normalize=False,
#     counterclock=False,
#     startangle=start,
# )
# plt.setp(pie2, width=width)

# colors = ["#b94d76", "#863a66"]
# pie1, _ = ax.pie(
#     counts1,
#     radius=radius - width - width - space * 2,
#     labeldistance=1 - width - width,
#     colors=colors,
#     wedgeprops={"linewidth": 0},
#     textprops={"fontsize": 5},
#     normalize=False,
#     counterclock=False,
#     startangle=start,
# )
# plt.setp(pie1, width=width)


# # export as svg
# plt.savefig(
#     DIRS.REPORTS / "figures" / "sample_sizes.svg",
#     transparent=True,
#     bbox_inches="tight",
# )

# plt.show()

# ──── DISTRIBUTION OF SONG LENGTHS ───────────────────────────────────────────

color = "#3695a1"
fig, ax = plt.subplots()
ax.hist(song_df.length_s, bins=40, color=color, edgecolor="white", rwidth=0.9)
ax.set_xlabel("Song length (s)")
ax.set_ylabel("Count (log scale)")
ax.set_title("Distribution of song lengths")
ax.set_yscale("log")
ax.set_ylim(0, ax.get_ylim()[1])
ax.set_xlim(0, song_df.length_s.max() + 2)
fix_aspect_ratio(ax, 1 / 4)


# Set y-axis labels to non-scientific notation
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", which="both", length=0)
# export as svg
plt.savefig(
    DIRS.REPORTS / "figures" / "dist_of_song_lengths.svg",
    transparent=True,
    bbox_inches="tight",
)
plt.show()


# TODO: now interval, note duration, etc?

# ──── DISTRIBUTION OF REPERTOIRE SIZES ───────────────────────────────────────


# Count number of different classes per ID, removing 0
classes_per_id = song_df.groupby("ID")["class_id"].nunique()
classes_per_id = classes_per_id[classes_per_id != 0]

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


# ──── DIST OF REPERTOIRE SIZES (POLAR BUBBLES) ───────────────────────────────

# Prepare data
classes_per_id = song_df.groupby("ID")["class_id"].nunique()
classes_per_id = classes_per_id[classes_per_id != 0]
unique_counts, frequencies = (
    classes_per_id.value_counts().sort_index().reset_index().values.T
)
x = np.ones_like(unique_counts)
theta = np.linspace(-3, 2 * np.pi, len(unique_counts), endpoint=False)
color = "#3695a1"
size_constant = 40


# Plot bubble chart in polar coordinates
ax = plt.subplot(111, projection="polar")
ax.scatter(
    theta,
    unique_counts,
    s=frequencies * size_constant,
    alpha=0.8,
    c=color,
    edgecolor="none",
    clip_on=False,
)

# Add point labels
for i, (t, r) in enumerate(zip(theta, unique_counts)):
    ax.text(t, r, f"{r}", ha="center", va="center")

# draw a SMOOTH LINE across all points:
ax.plot(theta, unique_counts, color="black", linewidth=0.5, clip_on=False)

# add a legend for the size of the points
for s in frequencies:
    ax.scatter([], [], c=color, alpha=0.8, s=s * size_constant, label=str(s))
ax.legend(
    scatterpoints=1,
    frameon=False,
    labelspacing=1,
    title="Count (n birds)",
    loc="center left",
    bbox_to_anchor=(1.4, 0.5),
    edgecolor="none",
)

# Calculate the mean and SD and median repertoire size
mean = classes_per_id.mean()
sd = classes_per_id.std()
median = classes_per_id.median()
mode = classes_per_id.mode()[0]

# Add text labels for mean, SD, median, and mode
ax.text(
    1.4, 0.95, f"Mean: {mean:.2f}", ha="right", va="top", transform=ax.transAxes
)
ax.text(
    1.4, 0.90, f"SD: {sd:.2f}", ha="right", va="top", transform=ax.transAxes
)
ax.text(
    1.4,
    0.85,
    f"Median: {median:.2f}",
    ha="right",
    va="top",
    transform=ax.transAxes,
)
ax.text(
    1.4, 0.80, f"Mode: {mode:.2f}", ha="right", va="top", transform=ax.transAxes
)
ax.grid(False)
ax.set_xticks(theta)
ax.set_xticklabels(["{:.2f}".format(t) for t in theta])
ax.set_yticks([])
ax.set_yticklabels([])
ax.invert_yaxis()

# add a title
ax.set_title("Distribution of repertoire sizes", y=1.1)

# export as svg
plt.savefig(
    DIRS.REPORTS / "figures" / "repertoire_sizes_bubble.svg",
    transparent=True,
    bbox_inches="tight",
)
plt.show()


# ──── MAP OF WYTHAM ──────────────────────────────────────────────────────────


perimeter, nestboxes, broods = prepare_shape_data(
    DIRS.PERIMETER, DIRS.NESTBOXES, DIRS.MAIN
)

# get the subset of main_df where year is 2020 onward

bdf = main_df[main_df.year >= 2020]
song_df["nestbox"] = song_df["ID"].str.extract(r"([A-Za-z]+\d+)")
nestbox_counts = bdf.groupby(["nestbox"]).size().reset_index(name="counts")
nestbox_songs = song_df.groupby("nestbox").size().reset_index(name="songs")
nestbox_data = pd.merge(nestbox_counts, nestbox_songs, on="nestbox")
nestbox_data = pd.merge(nestboxes, nestbox_data, on="nestbox", how="left")
nestbox_data = nestbox_data[nestbox_data.type == "G"]

# build a color palette dictionary linearly based on the number of songs recorded at each nestbox
colors = [
    "#086774",
    "#fcae13",
]
norm = mpcolors.LogNorm(vmin=1, vmax=nestbox_data["songs"].max())
cmap = mpcolors.LinearSegmentedColormap.from_list("custom", colors, N=256)
songs_min = nestbox_data["songs"].min()
songs_max = nestbox_data["songs"].max()
color_dict = {
    nestbox_data.loc[i, "nestbox"]: cmap(norm(nestbox_data.loc[i, "songs"]))
    for i in nestbox_data.index
}
nestbox_data["color"] = nestbox_data["nestbox"].map(color_dict)

# build a marker size dictionary linearly based on the number of times recorded at each nestbox
srange = nestbox_data["counts"].max() - nestbox_data["counts"].min()
_min = nestbox_data["counts"].min()
msrange = (50, 400)
nestbox_data["marker_size"] = nestbox_data["counts"].apply(
    lambda x: (x - _min) / srange * (msrange[1] - msrange[0]) + msrange[0]
)
nestbox_data.loc[nestbox_data["counts"].isna(), "marker_size"] = 10


# Plot the nestboxes and perimeter
fig, ax = plt.subplots(figsize=(8, 8))
perimeter.plot(ax=ax, alpha=0.1, edgecolor="k", linewidth=0, color="#2b4957")
ax.scatter(
    nestbox_data.x,
    nestbox_data.y,
    c=nestbox_data.color,
    alpha=0.8,
    linewidth=0,
    s=nestbox_data.marker_size,
)

# Add legend for marker sizes in nestbox_data
for size, label in zip(
    nestbox_data["marker_size"].unique(), nestbox_data["counts"].unique()
):
    ax.scatter(
        [],
        [],
        c="#2b4957",
        alpha=0.8,
        s=size,
        label=int(label) if not np.isnan(label) else 0,
    )
ax.legend(
    title="Years recorded",
    labelspacing=1.5,
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    frameon=False,
    fontsize=10,
    handletextpad=0.5,
    borderpad=0.5,
    title_fontsize=10,
    markerscale=1,
)
# add legend for marker colors in nestbox_data
unique_colors = nestbox_data["color"].unique()
unique_songs = nestbox_data["songs"].unique()
unique_songs = unique_songs[~np.isnan(unique_songs)]

min_color = unique_colors.min()
max_color = unique_colors.max()
middle_color = unique_colors[len(unique_colors) // 2]

min_song = unique_songs.min()
max_song = unique_songs.max()
middle_song = unique_songs[len(unique_songs) // 2]
# create a second axis for the legend
ax2 = ax.twinx()

for color, label in zip([max_color, middle_color], [max_song, min_song]):
    ax2.scatter(
        [],
        [],
        c=color,
        alpha=0.8,
        s=100,
        label=int(label) if not np.isnan(label) else 0,
    )

ax2.legend(
    title="Songs recorded",
    labelspacing=1.5,
    loc="upper right",
    bbox_to_anchor=(1, 1),
    frameon=False,
    fontsize=10,
    handletextpad=0.5,
    borderpad=0.5,
    title_fontsize=10,
    markerscale=1,
)

# set the x and y labels and title for the plot
ax.set_xlabel("Easting", fontsize=12, labelpad=10)
ax.set_ylabel("Northing", fontsize=12, labelpad=10)
ax.set_title("Wytham Woods Nestboxes", fontsize=14, pad=20)
ax.tick_params(axis="both", which="major", pad=2, labelsize=10)

plt.savefig(
    DIRS.REPORTS / "figures" / "wytham_map.svg",
    transparent=True,
    bbox_inches="tight",
)
plt.show()

# ──── MAP OF THE UK ──────────────────────────────────────────────────────────

# Just get it from the internet you idiot,
# no need to do *everything* programmatically

# ──── SONG LENGTH VS IOIS ────────────────────────────────────────────────────


song_df.loc[:, "iois"] = song_df["onsets"].apply(
    lambda x: [j - i for i, j in zip(x[:-1], x[1:])]
)
song_df.loc[:, "iois_mean"] = song_df["iois"].apply(lambda x: np.median(x))


# plot scatterplot between length_s and iois_mean
fig, ax = plt.subplots()
ax.scatter(
    song_df["length_s"],
    song_df["iois_mean"],
    color="#8cb8c9",
    alpha=0.5,
    s=1,
)
ax.set_xlabel("Song length (s)")
ax.set_ylabel("Mean inter-onset interval (s)")
ax.set_title("Relationship between song length and inter-onset interval")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# log y axis
ax.set_yscale("log")
ax.set_xscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
plt.show()


# ──── SIZE AND SONG LENGTH ───────────────────────────────────────────────────


# is there a correlation between song length and body size?

# match morpho_df and song_df based on 'bto_ring' and 'father', respectively


# merge the two dataframes
merged_df = pd.merge(
    main_df,
    morpho_df,
    left_on="father",
    right_on="bto_ring",
    how="left",
)

# only keep the columns we need
merged_df = merged_df[
    [
        "pnum",
        "bto_ring",
        "father",
        "tarsus_length",
        "weight",
        "wing_length",
    ]
]
# remove duplicates
merged_df = merged_df.drop_duplicates()
merged_df.loc[:, "year"] = merged_df["pnum"].astype(str).str[:4]
merged_df = merged_df[merged_df["year"] >= "2020"]


# get the median song length per individual from song_df (duration_s)
sdurs = (
    song_df.groupby("ID")["length_s"]
    .max()
    .reset_index()
    .rename(columns={"ID": "pnum"})
)


# add sdurs to merged_df based on ID
merged_df = pd.merge(
    merged_df,
    sdurs,
    on="pnum",
    how="left",
)
# remove tarsus lengths below 18mm
merged_df = merged_df[merged_df["tarsus_length"] >= 18]

# build a scatter plot of the relationship between tarsus length and song
# duration using sns, andd a linear regression line
fig, ax = plt.subplots()
sns.regplot(
    data=merged_df,
    x="wing_length",
    y="length_s",
    scatter_kws={"color": "#8cb8c9", "alpha": 0.5},
    line_kws={"color": "#3a7b86"},
    ax=ax,
)
ax.set_xlabel("Tarsus length (mm)")
ax.set_ylabel("Song length (s)")
ax.set_title("Relationship between tarsus length and song duration")
fix_aspect_ratio(ax, 9 / 16)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", which="both", length=0)
plt.show()
