# ──── DESCRIPTION ──────────────────────────────────────────────────────────────


# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import DIRS

from greti.io import prepare_shape_data
from greti.plot import plot_nestboxes_and_perimeter

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


def clean_column_names(nestbox_data):
    """
    Cleans the column names of a pandas DataFrame by converting them to lowercase, replacing spaces and dots with underscores,
    and removing parentheses and trailing underscores. Also renames the 'box' column to 'nestbox'.

    Args:
        nestbox_data (pandas.DataFrame): The DataFrame to clean.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
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
    """
    Cleans the string columns of a pandas DataFrame by converting them to lowercase, replacing spaces with underscores,
    and removing leading and trailing whitespaces.

    Args:
        nestbox_data (pandas.DataFrame): The DataFrame to clean.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
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
    """
    Filters and sorts a pandas DataFrame containing nestbox data. Converts the 'habitat_type' and 'soil_type' columns to integers,
    checks that all string columns are non-empty, sorts the DataFrame by 'nestbox', and resets the index.

    Args:
        nestbox_data (pandas.DataFrame): The DataFrame to filter.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
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
    """
    Plots a scatterplot for each variable in a list of variables, with the x and y coordinates of the nestboxes as the axes,
    and the variable as the color of the points.

    Args:
        nestbox_data (pandas.DataFrame): The DataFrame containing the nestbox data.
        variables (list): A list of variables to plot.
        num_cols (int): The number of columns in the plot grid.
        axs (numpy.ndarray): A 2D array of matplotlib Axes objects.

    Returns:
        None
    """
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
    """
    Plots a grid of scatterplots for a list of variables, with the x and y coordinates of the nestboxes as the axes,
    and the variable as the color of the points.

    Args:
        nestbox_data (pandas.DataFrame): The DataFrame containing the nestbox data.
        variables (list): A list of variables to plot.

    Returns:
        None
    """
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
