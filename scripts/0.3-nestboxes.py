# ──── DESCRIPTION ──────────────────────────────────────────────────────────────


# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plot_variables(nestbox_data, variables)
