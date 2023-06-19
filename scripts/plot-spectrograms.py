# ──── DESCRIPTION ─────────────────────────────────────────────────────────────

# This script is used to review the dataset and remove any noise and labelling
# mishaps. It is run manually and carefully, and it should only run once and
# only after the dataset has been labelled.


# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import itertools
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import DIRS, build_projdir
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pykanto.dataset import KantoData
from pykanto.signal.spectrogram import cut_or_pad_spectrogram
from pykanto.utils.compute import with_pbar
from pykanto.utils.io import load_dataset
from pynndescent import NNDescent
from scipy.spatial.distance import cdist

# pca embedding
from sklearn.decomposition import PCA
from umap import UMAP

# ──── FUNCTION DEFINITIONS ───────────────────────────────────────────────────


def plot_spectrograms(
    specs, dists, labels, bird, cmap="binary_r", figsize=(25, 5)
):
    """
    Plots a row of spectrograms with their distances, labels, and bird names.

    Args:
        specs (list of numpy arrays): The spectrograms to plot.
        dists (numpy array): The distances of the nearest neighbors.
        labels (list of str): The class IDs of the nearest neighbors.
        bird (list of str): The bird names of the nearest neighbors.

    Returns:
        None
    """
    fig, axs = plt.subplots(
        1,
        len(specs),
        figsize=figsize,
        gridspec_kw={"width_ratios": [s.shape[1] for s in specs]},
    )
    # plot the spectrograms and their metadata
    for i, spec in enumerate(specs):
        axs[i].imshow(spec, cmap=cmap, aspect="auto")
        dtext = f"Distance: {dists[i]:.2f}" if dists[i] != 0 else "Query"
        axs[i].set_title(
            dtext + "\n"
            f"Class ID: {labels[i]}"
            "\n"
            f"Father: {bird[i] if not pd.isna(bird[i]) else 'Unknown'}"
        )
        axs[i].tick_params(axis="y", which="both", length=0)
        axs[i].yaxis.set_ticks([])
        axs[i].grid(False)
        axs[i].xaxis.set_ticks([])
        axs[i].xaxis.set_ticklabels([])
    plt.tight_layout()
    # save the figure
    outpath = DIRS.REPORTS / "figures" / "spectrograms" / f"{query_id}.png"
    outpath.parent.mkdir(exist_ok=True)
    plt.savefig(
        outpath,
        transparent=True,
        bbox_inches="tight",
    )
    plt.show()


def find_nearest_neighbors(
    query_id: str,
    features: pd.DataFrame,
    nnindex: NNDescent,
    n_neighbours: int,
    dataset: KantoData,
) -> Tuple[List[np.ndarray], List[int], np.ndarray, List[str], List[str]]:
    """
    Finds the nearest neighbors of a query spectrogram.

    Args:
        query_id: The ID of the query spectrogram.
        features: The median features of the spectrograms.
        nnindex: The nearest neighbor index.
        n_neighbours: The number of nearest neighbors to find.
        dataset: The dataset containing the spectrograms.

    Returns:
        A tuple containing:
        - A list of numpy arrays representing the spectrograms of the nearest neighbors.
        - A list of integers representing the keys of the nearest neighbors in the dataset.
        - A numpy array representing the distances of the nearest neighbors.
        - A list of strings representing the class IDs of the nearest neighbors.
        - A list of strings representing the bird names of the nearest neighbors.
    """

    row_index = features.index.get_loc(query_id)
    f = features.to_numpy()
    current_query = np.asarray(f[row_index], dtype=np.float32).reshape(1, -1)
    indx, dists = nnindex.query(current_query, k=n_neighbours)
    keys = [
        dataset.data[dataset.data.class_id == features.index[j]].index[0]
        for j in indx[0]
    ]
    labels = [dataset.data.at[k, "class_id"] for k in keys]
    bird = [dataset.data.at[k, "father"] for k in keys]

    specs = [np.load(dataset.files.at[k, "spectrogram"]) for k in keys]
    max_len = np.mean([s.shape[1] for s in specs]).astype(int)
    specs = [
        np.pad(
            s,
            (
                (0, 0),
                ((max_len - s.shape[1]) // 2, (max_len - s.shape[1] + 1) // 2),
            ),
            mode="minimum",
        )
        if s.shape[1] < max_len
        else s[:, (s.shape[1] - max_len) // 2 : (s.shape[1] + max_len) // 2]
        for s in specs
    ]

    return specs, keys, dists, labels, bird


# ──── SETTINGS ────────────────────────────────────────────────────────────────

# read in main dataset
dataset_name = "great-tit-hits"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)
dataset = load_dataset(DIRS.DATASET, DIRS)

# add bird rings to dataset where known
main_df = pd.read_csv(DIRS.MAIN)
dataset.data["index"] = dataset.data.index
dataset.data = pd.merge(
    dataset.data,
    main_df[["pnum", "father"]],
    left_on="ID",
    right_on="pnum",
    how="left",
).set_index("index")


# read in feature vectors
featv_dir = DIRS.ML / "output" / "feature_vectors.csv"
df = pd.read_csv(featv_dir)
df.index = dataset.data.index
df.insert(0, "class_id", dataset.data.class_id)

# remove class id col and to numpy array
featv = df.drop(columns=["class_id"]).to_numpy()

# Create index
med_featv = df.groupby("class_id").transform("median").drop_duplicates()
med_featv.index = df.class_id.unique()
medfeats = med_featv.to_numpy()
nnindex = NNDescent(medfeats, metric="euclidean", n_neighbors=30, n_jobs=-1)

n_neighbours = 6

# select 10 randomly from dataset.data.class_id.unique():
query_ids = np.random.choice(dataset.data.class_id.unique(), 10)

for query_id in query_ids:
    specs, keys, dists, labels, bird = find_nearest_neighbors(
        query_id, med_featv, nnindex, n_neighbours, dataset
    )
    plot_spectrograms(
        specs, dists[0], labels, bird, cmap="binary_r", figsize=(30, 5)
    )


# project the feature vectors to 2D
# TODO: write script to test series of parameters on the HPC using the full
# dataset (featv)

embedding_dir = DIRS.ML / "output" / "umap_embedding.npy"

# check if embedding exists
if os.path.exists(embedding_dir):
    # load the embedding
    embedding = np.load(embedding_dir)
else:
    # calculate the embedding
    reducer = UMAP(
        n_neighbors=200, n_components=2, min_dist=0.25, metric="euclidean"
    )
    embedding = reducer.fit_transform(featv)
    # save the embedding
    np.save(embedding_dir, embedding)


pca = PCA(n_components=2)
embedding = pca.fit_transform(featv)


# color the points by year

# get the years
years = [int(i[:4]) for i in df.class_id.unique()]

# get the colors
cmap = plt.cm.get_cmap("viridis", len(years))
colors = [cmap(i) for i in range(len(years))]
colors = dict(zip(years, colors))

# plot the points, do not use a loop:
plt.figure(figsize=(10, 10))
for year in np.unique(years):
    plt.scatter(
        embedding[df.class_id.str.contains(str(year)), 0],
        embedding[df.class_id.str.contains(str(year)), 1],
        c=colors[year],
        s=1,
    )
plt.gca().set_aspect("equal", "datalim")
plt.show()


# ──── FIND AND PLOT DISTRIBUTION OF DISTANCES ────────────────────────────────

# Find the class_ids of the 'father' birds that appear in more than one year
dataset.data["year"] = dataset.data.ID.str[:4]
idd_birds = dataset.data[dataset.data.father.notna()].copy()
idd_birds = idd_birds.groupby("father").filter(
    lambda x: len(x.year.unique()) > 1
)
mult_birds = idd_birds.father.unique()

dfs = []

for year in dataset.data.year.unique():
    year_birds = dataset.data[dataset.data.year == year].father.unique()
    for father in year_birds:
        class_ids = (
            dataset.data[
                (dataset.data.father == father) & (dataset.data.year == year)
            ]
            .class_id.unique()
            .tolist()
        )
        df = pd.DataFrame(
            {"year": [year], "father": [father], "class_ids": [class_ids]}
        )
        dfs.append(df)

# concatenate the dataframes
all_df = pd.concat(dfs, ignore_index=True).dropna()


# find the feature vectors for the class IDs of the fathers that appear in more
# than one year and caclulate the minimum distance between them (across years)

mult_df = all_df[all_df.father.isin(mult_birds)]
dist_dict = {father: [] for father in mult_df.father.unique()}
for father in mult_df.father.unique():
    class_ids = mult_df[mult_df.father == father].class_ids.values
    for i, j in itertools.combinations(class_ids, 2):
        featv1 = med_featv[med_featv.index.isin(i)]
        featv2 = med_featv[med_featv.index.isin(j)]
        dists = cdist(featv1, featv2, metric="euclidean")
        min_dist = np.min(dists)
        dist_dict[father].append(min_dist)


# average the distances for each father
avg_dist_dict = {
    father: np.mean(dist_dict[father]) for father in dist_dict.keys()
}

# Now get the average distance between all the fathers that do not appear in multiple years
not_mult_df = all_df[~all_df.father.isin(mult_birds)]
vs = np.concatenate(not_mult_df.class_ids.values)
featv1 = med_featv[med_featv.index.isin(vs)]
featv2 = med_featv[med_featv.index.isin(vs)]
dists = cdist(featv1, featv2, metric="euclidean")
dists = dists[np.triu_indices(dists.shape[0], k=1)]

# Plot

fig, ax = plt.subplots()
sns.kdeplot(
    avg_dist_dict.values(),
    fill=True,
    color="blue",
    label="mult_birds",
    edgecolor=None,
    common_norm=True,
    cut=0,
)
sns.kdeplot(
    dists,
    fill=True,
    color="red",
    label="not_mult_birds",
    edgecolor=None,
    common_norm=True,
    cut=0,
)
sns.rugplot(
    avg_dist_dict.values(), color="blue", height=0.1, alpha=0.2, linewidth=3
)
sns.rugplot(dists, color="red", height=0.1, alpha=0.002, linewidth=3)
ax.set_ylim(-1, 9)
# set title
ax.set_title(
    "Minimum distance between average feature vectors of the same bird across years (blue)\n"
    "vs. distance between average feature vectors of all other birds (red)"
)
plt.legend()
plt.savefig(
    DIRS.REPORTS / "figures" / "dists_featvecs.svg",
    transparent=True,
    bbox_inches="tight",
)
plt.show()


# Below not used - for reference


# ──── PLOT SCATTER WITH SPECTROGRAMS ─────────────────────────────────────────


# coordinates in featv coincide with rows in df, where indexes are file name
# sand the class_id is the class id. for each class id, get the median of the
# feature vectors, get the row index of the median feature vector or the closest
# feature vector to the median, and using the positions from the embedding, plot
# each the images.


# for ech class id, get the median of the feature vectors, and then get the row
# index that is closest to the median feature vector

# create index for the entire dataset:
nnindex = NNDescent(featv, metric="euclidean", n_neighbors=30, n_jobs=-1)
# get the median of the feature vectors
med_featv = df.groupby("class_id").transform("median").drop_duplicates()
med_featv.index = df.class_id.unique()

# get the row index that is closest to the median feature vector
med_indices = nnindex.query(med_featv.to_numpy(), k=1)
med_indices = med_indices[0].flatten()
mindex = df.iloc[med_indices].index

# get the index for the first row of each class id
first_indices = df.groupby("class_id").head(1).index

# create dictionary with equivalent integers for each row of the dataframe:
# this is needed to get the embedding coordinates for each row
row_to_int = {row: i for i, row in enumerate(df.index)}

# get the int equivalent for each of first_indices
first_int = [row_to_int[i] for i in first_indices]


# get the spectrograms for each in med_indices, centre-crop them to 128x128 and
# plot them

specs = [
    np.load(i)
    for i in with_pbar(
        dataset.files.loc[first_indices].spectrogram,
        desc="Loading spectrograms",
    )
]

# now crop the spectrograms to 128x128
c_specs = [cut_or_pad_spectrogram(spec, 224) for spec in specs]

# plot a couple of them
plt.figure(figsize=(10, 10))
for i, spec in enumerate(c_specs[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(spec, cmap="binary_r")
    plt.axis("off")
plt.show()

# with the med_indices, get the embedding coordinates for each of them
med_coords = embedding[first_int]

# create figure
fig, ax = plt.subplots(figsize=(50, 50))
subset = 5

for i, spec in enumerate(c_specs):
    # create annotation box
    ab = AnnotationBbox(
        OffsetImage(spec, cmap="binary_r", zoom=0.3),
        med_coords[i],
        frameon=False,
        pad=0,
        box_alignment=(0, 0),
    )
    # add to the axes
    ax.add_artist(ab)

# set the limits of the axes
ax.set_xlim(embedding[:, 0].min(), embedding[:, 0].max())
ax.set_ylim(embedding[:, 1].min(), embedding[:, 1].max())
ax.set_aspect("equal", "datalim")
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
# save the figure
plt.savefig("test.png", dpi=100)


# ──── BULD COLOR MAP BASED ON FREQUENCY ──────────────────────────────────────


# group the data by class_id and compute the average spectrogram for each group
grouped_data = dataset.data.groupby("class_id")
specs = []
for class_id, group in with_pbar(grouped_data):
    # get the spectrograms for this group
    specs_group = []
    for i, row in group.iterrows():
        spec = np.load(row["spectrogram"])
        spec = np.mean(spec, axis=1)
        specs_group.append(spec)
    # compute the average spectrogram for this group
    avg_spec = np.mean(specs_group, axis=0)
    specs.append(avg_spec)

# Arrange specs by max frequency
specar = np.array(specs)
max_freqs = np.argmax(specar, axis=1)
specar = specar[np.argsort(max_freqs)]

# img plot of the first array in specs (make up the y axis)
fig, ax = plt.subplots(figsize=(10, 5))
rotated_specar = np.rot90(specar, k=1)
plt.imshow(rotated_specar, cmap="binary_r", aspect="auto")

specs = []
for class_id in with_pbar(dataset.data.class_id.unique()):
    k = dataset.data[dataset.data.class_id == class_id].index
    keys = (k[0], k[-1])
    bs = []
    for k in keys:
        spec = np.load(dataset.files.at[k, "spectrogram"])
        spec = np.mean(spec, axis=1)
        bs.append(spec)
    # average the spectrograms
    bs = np.array(bs)
    avg_spec = np.mean(bs, axis=0)
    specs.append(avg_spec)

# get the index of the frequency bin with the highest value for each spectrogram
max_freqs = np.argmax(specs, axis=1)
# get the median of each array in specs
med_specs = np.mean(specs, axis=1)
