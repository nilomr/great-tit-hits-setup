# ──── DESCRIPTION ──────────────────────────────────────────────────────────────


# ──── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import DIRS

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


def clean_morpho_data(morpho_data):
    """
    Cleans the morphometrics data by performing the following operations:
    - Clean column names
    - Remove quotes from pit tags
    - Change empty pit tags to NaN
    - Convert pit_tag_state column to integer allowing NA values
    - Parse date_time and order by date_time
    - Clean string columns
    - Location to uppercase for consistency with other datasets
    - Check that there are no empty strings in the dataframe
    - Rename bto_species_code column to 'species' and only keep the first character of the string
    - Create a new column 'year' from the date_time column

    Args:
        morpho_data (pandas.DataFrame): The morphometrics data to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned morphometrics data.
    """
    # Clean column names
    morpho_data.columns = morpho_data.columns.str.lower().str.replace(" ", "_")

    # Remove quotes from pit tags
    morpho_data.pit_tag = morpho_data.pit_tag.str.replace("'", "")

    # Change empty pit tags to NaN
    morpho_data.pit_tag = morpho_data.pit_tag.replace("", None)

    # Convert pit_tag_state column to integer allowing NA values
    morpho_data.pit_tag_state = morpho_data.pit_tag_state.astype(
        pd.Int64Dtype()
    )

    # Parse date_time and order by date_time
    morpho_data.date_time = pd.to_datetime(
        morpho_data.date_time, format="%d-%m-%Y %H:%M"
    )
    morpho_data = morpho_data.sort_values("date_time").reset_index(drop=True)
    morpho_data = morpho_data.loc[:, ~morpho_data.columns.str.match("unnamed")]

    # Loop over all string columns and clean them
    for column in morpho_data.columns:
        if morpho_data[column].dtype == "object":
            morpho_data[column] = morpho_data[column].apply(
                lambda x: x.strip().casefold().replace(" ", "_")
                if isinstance(x, str)
                else x
            )

    # uppercase location for consistency with other datasets
    morpho_data.location = morpho_data.location.str.upper()

    # Check that there are no empty strings in the dataframe
    for column in morpho_data.columns:
        if morpho_data[column].dtype == "object":
            assert morpho_data[column].apply(lambda x: x != "").all()

    # Rename bto_species_code column to 'species' and only keep the first
    # character of the string;
    morpho_data.rename(columns={"bto_species_code": "species"}, inplace=True)
    morpho_data.species = morpho_data.species.str[0]
    morpho_data = morpho_data[morpho_data.species == "g"]

    # Create a new column 'year' from the date_time column
    morpho_data.insert(10, "year", morpho_data.date_time.dt.year)

    return morpho_data


def plot_unique_birds_per_year(morpho_data):
    """Plots the number of unique birds (based on bto_ring) per year.

    Args:
        morpho_data (pandas.DataFrame): The morphometric data.

    Returns:
        None
    """
    unique_birds = morpho_data.drop_duplicates(subset="bto_ring")
    sns.set_style("whitegrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x="year", data=unique_birds, ax=ax, color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of birds")
    ax.set_title("Number of unique birds per year")
    plt.show()


def print_oldest_birds(morpho_data):
    """Prints the oldest birds in the dataset based on the difference between the
    oldest and first date_time for each bto_ring.

    Args:
        morpho_data (pandas.DataFrame): The morphometric data.

    Returns:
        None
    """
    oldest_bird = morpho_data.groupby("bto_ring").date_time.agg(["min", "max"])
    oldest_bird["age"] = oldest_bird["max"] - oldest_bird["min"]
    oldest_bird["age"] = oldest_bird["age"].dt.days / 365
    print(oldest_bird.sort_values("age", ascending=False).head(10))


# ──── MAIN ─────────────────────────────────────────────────────────────────────


# Load the dataframe
morpho_data = pd.read_csv(
    DIRS.BIRD_DATA / "ebmp_morphometrics.csv", engine="python"
)

# Clean the dataframe
morpho_data = clean_morpho_data(morpho_data)

# Plot the number of unique birds per year
plot_unique_birds_per_year(morpho_data)

# Whats the oldest bird in the dataset?
print_oldest_birds(morpho_data)

# Save the clean dataframe
morpho_data.to_csv(DIRS.MORPHOMETRICS, index=False, na_rep="NA")
