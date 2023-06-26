# ──── DESCRIPTION ──────────────────────────────────────────────────────────────


# ──── IMPORTS ──────────────────────────────────────────────────────────────────
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import DIRS

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


def clean_column_names(broods_data):
    broods_data.columns = broods_data.columns.str.lower().str.replace(" ", "_")
    broods_data.columns = broods_data.columns.str.replace(".", "_")
    broods_data = broods_data.loc[:, ~broods_data.columns.str.match("legacy")]
    return broods_data


def clean_dates(broods_data):
    """
    Clean date columns in the broods_data DataFrame by replacing "/" with "-"
    and converting them to datetime format.

    Args:
    broods_data (pd.DataFrame): The DataFrame containing the broods data.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    date_cols = ["lay_date", "clear_date", "expected_hatch_date", "hatch_date"]
    for col in date_cols:
        broods_data.loc[:, col] = broods_data[col].str.replace("/", "-")
        broods_data.loc[:, col] = pd.to_datetime(
            broods_data[col], format="%d-%m-%Y"
        )
    return broods_data


def clean_strings(broods_data):
    """
    Clean string columns in the broods_data DataFrame by stripping whitespace,
    converting to lowercase, and replacing spaces with underscores.

    Args:
    broods_data (pd.DataFrame): The DataFrame containing the broods data.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    for column in broods_data.columns:
        if column in ["pnum", "owner", "nestbox"]:
            continue
        elif broods_data[column].dtype == "object":
            broods_data.loc[:, column] = broods_data[column].apply(
                lambda x: x.strip().casefold().replace(" ", "_")
                if isinstance(x, str)
                else x
            )
    return broods_data


def filter_broods_data(broods_data, first_year: int | None):
    """
    Filter the broods_data DataFrame by year and species, and add a new column
    for the nestbox.

    Args:
    broods_data (pd.DataFrame): The DataFrame containing the broods data.
    first_year (int | None): The earliest year to include in the filtered DataFrame.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    if first_year is not None:
        broods_data = broods_data[broods_data.year >= first_year]
    broods_data = broods_data[~broods_data.species.isna()]
    broods_data.insert(2, "nestbox", broods_data.pnum.apply(lambda x: x[5:]))
    broods_data.april_lay_date = broods_data.april_lay_date.astype(
        pd.Int64Dtype()
    )
    broods_data.state_code = broods_data.state_code.astype(pd.Int64Dtype())
    assert broods_data.lay_date.apply(lambda x: x != "").all()
    assert broods_data.lay_date.apply(lambda x: x is not None).all()
    for column in broods_data.columns:
        if broods_data[column].dtype == "object":
            assert broods_data[column].apply(lambda x: x != "").all()
    return broods_data


def plot_broods_per_year(broods_data):
    """
    Plot the number of unique broods per year.

    Args:
    broods_data (pd.DataFrame): The DataFrame containing the broods data.
    """
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x="year", data=broods_data, ax=ax, color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of broods")
    ax.set_title("Number of unique broods per year")
    plt.show()


def plot_species_proportions(broods_data):
    """
    Plot the proportion of rows that are species == b and species ==g for each year.

    Args:
    broods_data (pd.DataFrame): The DataFrame containing the broods data.
    """
    species_prop = (
        broods_data.groupby(["year", "species"]).size().unstack().fillna(0)
    )
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in species_prop.columns:
        line = sns.lineplot(
            x=species_prop.index,
            y=column,
            data=species_prop,
            ax=ax,
            label=column,
        )
        x_pos = species_prop.index[-1]
        y_pos = species_prop[column][x_pos]
        ax.text(
            x_pos + 0.2, y_pos, column, color=line.get_lines()[-1].get_color()
        )
    ax.set(
        xlabel="Year",
        ylabel="Number of Birds",
        title="Number of Each Species Over Time",
    )
    sns.despine(left=True, bottom=True)
    ax.legend().remove()
    plt.tight_layout()
    plt.show()


def plot_missing_lay_dates(broods_data):
    """
    Plot the number of missing lay date values per year.

    Args:
    broods_data (pd.DataFrame): The DataFrame containing the broods data.
    """
    missing = broods_data.groupby("year")["lay_date"].apply(
        lambda x: x.isna().sum()
    )
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=missing.index, y=missing.values, ax=ax)
    # make line thicker and orange:
    ax.lines[0].set_linewidth(3)
    ax.lines[0].set_color("#b3550e")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Missing Lay Dates")
    ax.set_title("Missing Lay Dates Per Year")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_lay_date_counts(broods_data):
    """
    Plot the number of birds that have lay dates each year.

    Args:
    broods_data (pd.DataFrame): The DataFrame containing the broods data.
    """
    lay_date_count = broods_data.groupby("year")["lay_date"].count()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=lay_date_count.index, y=lay_date_count.values, ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Birds with Lay Dates")
    ax.set_title("Birds with Lay Dates Per Year")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


# ──── MAIN ─────────────────────────────────────────────────────────────────────

# Load the data
broods_data = pd.read_csv(DIRS.BIRD_DATA / "ebmp_broods.csv", engine="python")

# Clean the main dataset
broods_data = (
    clean_column_names(broods_data)
    .pipe(filter_broods_data, 2013)
    .pipe(clean_dates)
    .pipe(clean_strings)
)

# Also load extra file with 2020 data, some of which is missing from the main
# dataset, and add it to the main dataset.
# REVIEW: remove when people get their shit together

broods_data_2020 = pd.read_csv(
    DIRS.BIRD_DATA / "ebmp_broods_2020.csv", engine="python"
)

broods_data_2020 = (
    clean_column_names(broods_data_2020)
    .pipe(filter_broods_data, None)
    .pipe(clean_dates)
    .pipe(clean_strings)
)
# add a 'year column to broods_data_2020 before any other columns:
broods_data_2020.insert(0, "year", 2020)

# substitute the 2020 rows in broods_data with the 2020 rows in
# broods_data_2020:
broods_data = pd.concat(
    [broods_data[broods_data.year != 2020], broods_data_2020], ignore_index=True
)

# ──── SAVE THE CLEAN DATAFRAME ─────────────────────────────────────────────────

broods_data.to_csv(DIRS.BROODS, index=False, na_rep="NA")


# ──── EXTRA VISUAL CHECKS ──────────────────────────────────────────────────────

plot_broods_per_year(broods_data)
plot_species_proportions(broods_data)
plot_missing_lay_dates(broods_data)
plot_lay_date_counts(broods_data)
