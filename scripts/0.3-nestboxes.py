# ──── DESCRIPTION ──────────────────────────────────────────────────────────────


# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations
from pathlib import Path

from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import DIRS

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


def clean_column_names(nestbox_data):
    nestbox_data.columns = nestbox_data.columns.str.lower().str.replace(
        " ", "_"
    )
    nestbox_data.columns = nestbox_data.columns.str.replace(".", "_")
    nestbox_data.columns = nestbox_data.columns.str.replace("(", "_")
    nestbox_data.columns = nestbox_data.columns.str.replace(")", "_")
    nestbox_data.columns = nestbox_data.columns.str.strip("_")
    nestbox_data = nestbox_data.rename(columns={"box": "nestbox"})
    return nestbox_data


def clean_strings(nestbox_data):
    for column in nestbox_data.columns:
        if column in ["nestbox", "section", "type"]:
            continue
        elif nestbox_data[column].dtype == "object":
            nestbox_data.loc[:, column] = nestbox_data[column].apply(
                lambda x: x.strip().casefold().replace(" ", "_")
                if isinstance(x, str)
                else x
            )
    return nestbox_data


def filter_nestbox_data(nestbox_data):
    int_vars = ["habitat_type", "soil_type"]
    for var in int_vars:
        nestbox_data[var] = nestbox_data[var].astype(int)
    for column in nestbox_data.columns:
        if nestbox_data[column].dtype == "object":
            assert nestbox_data[column].apply(lambda x: x != "").all()
    nestbox_data = nestbox_data.sort_values("nestbox")
    nestbox_data = nestbox_data.reset_index(drop=True)

    return nestbox_data


def plot_map(nestbox_data, variables, num_cols, axs):
    for i, variable in enumerate(variables):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]

        sns.scatterplot(
            data=nestbox_data,
            x="x",
            y="y",
            hue=variable,
            size=1,
            legend=False,
            edgecolor=None,
            palette="viridis",
            ax=ax,
        )

        ax.set_title(variable)
        ax.set_aspect("equal", "box")
        ax.axis("off")


def plot_variables(nestbox_data, variables):
    num_plots = len(variables)
    num_cols = 6
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(15, 10),
        squeeze=False,
    )

    plot_map(nestbox_data, variables, num_cols, axs)
    if num_plots < num_cols * num_rows:
        for i in range(num_plots, num_cols * num_rows):
            fig.delaxes(axs[i // num_cols, i % num_cols])

    plt.tight_layout()
    plt.show()


def prepare_shape_data(
    perimeter_path: Path, nestboxes_path: Path, broods_path: Path
) -> Tuple:
    """
    Read data from files and return the prepared data.

    Args:
    - perimeter_path (str): The path to the shapefile of the perimeter.
    - nestboxes_path (str): The path to the CSV file of the nestboxes.
    - broods_path (str): The path to the CSV file of the recorded nestboxes.

    Returns:
    - Tuple containing the following data:
        - perimeter (geopandas.GeoDataFrame): GeoDataFrame containing the perimeter data
        - nestboxes (pandas.DataFrame): DataFrame containing the nestboxes data
        - broods (pandas.DataFrame): DataFrame containing the recorded nestboxes data
    """
    # read in the shapefile
    perimeter = gpd.read_file(perimeter_path).iloc[0:1]

    # import nestbox coordinates
    nestboxes = pd.read_csv(nestboxes_path)

    # import data on recorded nestboxes
    broods = pd.read_csv(broods_path)

    return perimeter, nestboxes, broods


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


# ──── MAIN ─────────────────────────────────────────────────────────────────────

# Load the data
nestbox_data = pd.read_csv(
    DIRS.BIRD_DATA / "ebmp_nestboxes.csv", engine="python"
)

# Clean the main dataset
nestbox_data = (
    clean_column_names(nestbox_data)
    .pipe(filter_nestbox_data)
    .pipe(clean_strings)
)


# ──── SAVE THE CLEAN DATAFRAME ─────────────────────────────────────────────────

nestbox_data.to_csv(DIRS.NESTBOXES, index=False, na_rep="NA")


# ──── EXTRA VISUAL CHECKS ──────────────────────────────────────────────────────

# plot nextbox position (x and y columns) and colour by habitat type:
# Load the data


variables = [
    "poly",
    "edge_edi",
    "altitude_m",
    "aspect",
    "northness",
    "habitat_type",
]


plot_variables(nestbox_data, variables)

perimeter, nestboxes, broods = prepare_shape_data(
    DIRS.PERIMETER, DIRS.NESTBOXES, DIRS.MAIN
)

plot_nestboxes_and_perimeter(perimeter, nestboxes)
