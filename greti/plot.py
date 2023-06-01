from typing import Tuple

import geopandas as gpd
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_class_count_histogram(dataset):
    """
    Plots a histogram of counts by class ID with a log x-axis and customized ticks.
    Args:
        dataset (pandas.DataFrame): The dataset containing the data.
    Returns:
        None
    """

    # Group the data by "class_id" and count occurrences
    counts = dataset.data.groupby("class_id")["class_id"].count()

    # Set a custom color palette with organic colors
    colors = ["#c2d6d6", "#b7d8b0", "#d8d8a8", "#dbb0a8", "#d6c2d6"]
    sns.set_palette(colors)

    plt.figure(figsize=(8, 4))
    plt.hist(
        counts,
        bins=np.logspace(np.log10(counts.min()), np.log10(counts.max()), 30),
        log=False,
    )
    plt.xscale("log")
    plt.xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    plt.minorticks_on()
    plt.gca().xaxis.set_tick_params(
        which="minor", bottom=True, top=False, labelbottom=False
    )
    plt.gca().spines["top"].set_visible(False)
    plt.grid(False)
    plt.xlabel("Counts")
    plt.ylabel("Frequency")
    plt.title("Histogram of Counts by Class ID")
    plt.show()


# ──── FUNCTION DEFINITIONS ───────────────────────────────────────────────────


def fix_aspect_ratio(ax: axes.Axes, ratio: float) -> None:
    """
    Set the aspect ratio of an Axes object to a fixed value.
    Args:
        ax: The Axes object to set the aspect ratio for.
        ratio: The desired aspect ratio.
    Returns:
        None
    """
    xvals, yvals = ax.get_xlim(), ax.get_ylim()
    xrange, yrange = xvals[1] - xvals[0], yvals[1] - yvals[0]

    # Check if either axis is log-scaled
    if ax.get_xscale() == "log":
        xrange = np.log10(xrange)
    if ax.get_yscale() == "log":
        yrange = np.log10(yrange)

    ax.set_aspect(ratio * (xrange / yrange), adjustable="box")


def plot_nestboxes_and_perimeter(
    perimeter: gpd.GeoDataFrame,
    nestboxes: pd.DataFrame,
    fig_size: Tuple[int, int] = (8, 8),
) -> None:
    """
    Plot the nestboxes and perimeter.
    Args:
    - perimeter (geopandas.GeoDataFrame): GeoDataFrame containing the perimeter data.
    - nestboxes (pandas.DataFrame): DataFrame containing the nestboxes data.
    - fig_size (Tuple[int,int]): The size of the figure to create. Default is (8,8).
    Returns:
    - None
    """
    sns.set_style("white")

    # plot the nestboxes and perimeter
    fig, ax = plt.subplots(figsize=fig_size)
    perimeter.plot(ax=ax, alpha=0.5, edgecolor="k", linewidth=0, color="grey")
    plt.scatter(
        nestboxes.x,
        nestboxes.y,
        color="#242424",
        alpha=1,
        linewidth=0,
        s=10,
        label="Not Recorded",
    )
    ax.set_xlabel("Easting", fontsize=12, labelpad=10)
    ax.set_ylabel("Northing", fontsize=12, labelpad=10)
    ax.set_title("Wytham Woods Nestboxes", fontsize=14, pad=20)
    ax.tick_params(axis="both", which="major", pad=2, labelsize=10)
    sns.despine(ax=ax, left=True, bottom=True)
    plt.show()
