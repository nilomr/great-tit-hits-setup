# ─── DESCRIPTION ─────────────────────────────────────────────────────────────

"""
Code to read and combine relevant information available for each breeding
attempt in a nest box at which we tried to record songs.
"""

# ──── IMPORTS ──────────────────────────────────────────────────────────────────


from __future__ import annotations

from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import DIRS
from matplotlib.patches import Patch

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


def read_broods_data():
    date_cols = ["lay_date", "clear_date", "expected_hatch_date", "hatch_date"]
    broods_data = pd.read_csv(DIRS.BROODS, parse_dates=date_cols)
    return broods_data


def read_morphometrics():
    morpho = pd.read_csv(DIRS.MORPHOMETRICS)
    return morpho


def read_nestboxes():
    nestboxes = pd.read_csv(DIRS.NESTBOXES)
    nestboxes.drop(columns=["section"], inplace=True)
    return nestboxes


def filter_species(morpho, broods, species="g"):
    """Filter the morpho and broods dataframes to keep only the specified
    species.
    Creates an exception for ['20201MP57', '20201O81'], which are known to be
    great tits but marked as a mixed brood.
    """
    morpho = morpho.query(f"species == '{species}'")
    print(
        "broods with pnum ['20201MP57', '20201O81', '20221EX62'] are mixed broods, "
        "change to 'g' based on fieldworker comments"
    )
    broods.loc[
        broods["pnum"].isin(["20201MP57", "20201O81", "20221EX62"]), "species"
    ] = "g"
    broods = broods.query(f"species == '{species}'")

    return morpho, broods


def get_ided_males(broods):
    """Get the IDs of males that are present in the broods dataframe."""
    fathers = broods.father.values
    fathers = list(fathers[~pd.isnull(fathers)])
    return fathers


def add_resident_column(broods, fathers, morpho):
    """Add a 'resident' column to the broods dataframe, indicating whether each male is a resident or not."""
    wytham_born = morpho.query(
        "bto_ring == @fathers and age=='1'"
    ).bto_ring.values
    broods["resident"] = broods.father.apply(
        lambda x: True
        if x in wytham_born
        else np.nan
        if pd.isnull(x)
        else False
    )
    return broods


def plot_proportion_residents(broods):
    """Calculate and plot the proportion of residents in each year.

    Args:
        broods (pandas.DataFrame): A DataFrame with columns 'year' (categorical)
            and 'resident' (boolean).

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The plot object.
    """
    # Calculate the proportion of residents in each year
    proportions = broods.groupby("year")["resident"].mean()

    # Set plot size and style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))

    # Plot the data
    ax = plt.plot(
        proportions.index, proportions.values, linewidth=3, color="royalblue"
    )

    # Set plot title, axis labels, and tick parameters
    plt.title(
        "Proportion of broods by resident dads per year", pad=20, fontsize=20
    )
    plt.xlabel("Year", labelpad=10, fontsize=16)
    plt.ylabel("Proportion residents", labelpad=10, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Show the plot
    plt.show()


def separate_natal_box_column(broods):
    """Separate the natal_box column in the broods dataframe into natal_box and year_born columns."""
    broods[["natal_box", "year_born"]] = broods["location"].apply(
        lambda x: pd.Series((x[5:], x[:4]))
        if isinstance(x, str) and len(x) > 4
        else pd.Series((np.nan, np.nan))
    )
    broods.drop(columns=["location"], inplace=True)
    return broods


def add_natal_box_column(broods, fathers, morpho):
    """Add information about the natal box of each male to the broods dataframe."""
    natal_box = morpho.query("bto_ring == @fathers and age=='1'")[
        ["bto_ring", "location"]
    ]
    broods = (
        broods.merge(
            natal_box, left_on="father", right_on="bto_ring", how="outer"
        )
        .drop_duplicates()
        .drop(columns=["bto_ring"])
    )
    return separate_natal_box_column(broods)


def add_father_age(morpho, fathers, broods):
    """
    Adds the age of the father to each brood in the dataset.

    Parameters:
        morpho (pd.DataFrame): Morphometric data
        fathers (list): List of unique father IDs
        broods (pd.DataFrame): Brood data

    Returns:
        pd.DataFrame: The input brood data with father ages added
    """

    # Select ages of each male:
    ages = morpho.query("bto_ring == @fathers")[["bto_ring", "age", "year"]]
    ages.rename(columns={"age": "bto_age"}, inplace=True)

    # Add ages to broods data so that the age of the father that year is added:
    broods = broods.merge(
        ages,
        left_on=["father", "year"],
        right_on=["bto_ring", "year"],
        how="left",
    )
    broods.drop(columns=["bto_ring"], inplace=True)

    # Where pnums are duplicated, keep the one with the lowest bto_age:
    broods = broods.sort_values("bto_age").drop_duplicates(
        subset=["pnum"], keep="last"
    )

    return broods


def get_bird_age(age_code, year_born):
    """
    Get the age of a bird in years from its age code or year born.
    See: https://www.bto.org/sites/default/files/u17/downloads/about/resources/agecodes.pdf

    Parameters:
        age_code (int): The age code of the bird
        year_born (int or np.nan): The year the bird was born, or np.nan if unknown

    Returns:
        int or str or np.nan: The age of the bird in years, or "unknown" or np.nan if not available
    """

    if pd.isnull(age_code):
        return np.nan

    age_code = int(age_code)

    if pd.isnull(year_born):
        if age_code == 0:
            return np.nan
        elif age_code == 1:
            return 0
        elif age_code == 4:
            return "unknown"
        elif age_code == 5:
            return 1
        elif age_code == 6:
            return "adult"
        else:
            raise ValueError(f"Age code {age_code} not recognised")
    else:
        return "resident"


def add_bird_age(broods):
    """
    Adds the age of each bird in the brood data.

    Parameters:
        broods (pd.DataFrame): Brood data

    Returns:
        pd.DataFrame: The input brood data with bird ages added
    """

    # Add bird ages to brood data:
    broods["age"] = broods.apply(
        lambda x: get_bird_age(x["bto_age"], x["year_born"]), axis=1
    )

    # Any bird with age == 1 was born in the previous year:
    broods.loc[broods.age == 1, "year_born"] = broods.year - 1

    # Arrange by year and pnum:
    broods = broods.sort_values(["year", "pnum"])

    return broods


def get_adult_birds(morpho, broods):
    """
    Extracts non-resident adult birds from broods DataFrame and calculates their
    year of birth if possible.

    Args:
        morpho (pandas.DataFrame): DataFrame with morphometric data for the birds.
        broods (pandas.DataFrame): DataFrame with breeding data for the birds.

    Returns:
        pandas.DataFrame: DataFrame with the bto_ring, age, year, and year_born
        of each adult bird.
    """

    # Get birds where age == 'adult' and find them in the morphometrics data:
    adults = broods.query("age == 'adult'").father.values
    adults = list(adults[~pd.isnull(adults)])
    adults = morpho.query("bto_ring == @adults")

    # For each adult, order it by year and calculate when it was born, where if age
    # ==3 the year born is the current year and if age ==5 the year born is the
    # previous year:
    adults["year_born"] = adults.apply(
        lambda x: x["year"]
        if x["age"] == "3"
        else x["year"] - 1
        if x["age"] == "5"
        else np.nan,
        axis=1,
    )

    # Remove rows where year_born is missing:
    adults = adults[~pd.isnull(adults.year_born)][
        ["bto_ring", "age", "year", "year_born"]
    ]
    return adults


def refine_brood_ages(morpho, fathers, broods):
    """
    Refine ages for 4 and 6 birds where possible.
    """

    adults = get_adult_birds(morpho, broods)

    # complete the year_born column in broods with the year_born column in adults if
    # year_born in broods is missing:
    broods["year_born"] = broods.apply(
        lambda x: adults.query("bto_ring == @x.father").year_born.values[0]
        if pd.isnull(x.year_born)
        and len(adults.query("bto_ring == @x.father")) > 0
        else x.year_born,
        axis=1,
    )

    # calculate age using year_born (year - year_born):
    broods["age"] = broods.apply(
        lambda x: int(x["year"]) - int(x["year_born"])
        if not pd.isnull(x.year_born)
        else x.age,
        axis=1,
    )

    # Update the age column for rows where age is either 0 or -1:
    for value in [0, -1]:
        df = broods.query(f"age == {value}")
        if len(df) > 0:
            print(
                f"Found {len(df)} birds with age == {value}, will set their age to 'unknown'"
            )
            broods.loc[df.index, "age"] = "unknown"

    return broods


def plot_age_classes(broods):
    sns.set_style("whitegrid")
    custom_palette = sns.color_palette(
        ["#FFC107", "#009688"] + sns.color_palette("rocket", n_colors=7)
    )

    plt.figure(figsize=(10, 5))
    sns.countplot(
        x="year",
        hue="age",
        data=broods,
        palette=custom_palette,
        width=1,
    )
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.legend(title="Age", loc="upper right")
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_proportion_recorded(broods):
    # plot histogram of recorded pnums per year as proportion of total broods that had a
    # non - na lay_date:
    # filter out broods with no lay_date:
    broods_nona = broods[broods.lay_date.notna()]

    # get the number of broods recorded per year:
    rec_per_year = broods_nona.groupby("year").recorded.sum()

    # get the total number of broods per year:
    total_per_year = broods_nona.groupby("year").recorded.count()

    # get the proportion of broods recorded per year:
    prop_rec_per_year = rec_per_year / total_per_year

    # plot:
    plt.figure(figsize=(10, 5))
    # all bars to grey:
    sns.barplot(
        x=prop_rec_per_year.index, y=prop_rec_per_year.values, color="#9E9E9E"
    )
    plt.xlabel("Year")
    plt.ylabel("Proportion of broods recorded")
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_delay_distribution(broods):
    """Plots a distribution of the delay between lay_date and first recording
    for different years.

    Args:
        broods (pandas.DataFrame): A DataFrame with columns 'delay' (numeric)
            and 'year' (categorical).

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The plot object.
    """

    # Set plot size and style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))

    # get only recorded broods:
    df = broods[~broods["recorded"] == False]
    labels = df[~df["year"].isna()]["year"].unique().astype(int)

    # Create KDE plot with filled area and multiple colors
    ax = sns.kdeplot(
        data=df,
        x="delay",
        hue="year",
        fill=True,
        common_norm=False,
        palette=sns.color_palette("icefire", len(labels)),
        alpha=0.5,
        linewidth=0,
        legend=False,
    )

    # create the legend, making sure the order of labels is respected:
    ax.legend(
        handles=[
            Patch(
                color=sns.color_palette("icefire", len(labels))[i],
                label=labels[i],
            )
            for i in range(len(labels))
        ],
        title="Year",
        loc="upper left",
        bbox_to_anchor=(0, 1),
        frameon=False,
    )

    # Set plot title, axis labels, and tick parameters
    ax.set_title(
        "Distribution of delay between lay date and first recording", pad=20
    )
    ax.set_xlabel("Delay (days)", labelpad=10)
    ax.set_ylabel("Density", labelpad=10)
    ax.tick_params(axis="both", which="major", pad=10)

    # Increase all font size
    plt.rcParams.update({"font.size": 16})

    plt.tight_layout()
    plt.show()


def get_pnum_first_last_recordings(
    olddirs: list[Path], extension: str = ".WAV"
) -> Dict[str, Dict[str, dt]]:
    d = {}
    for box in olddirs:
        box.stem
        fs = list(box.glob(f"*{extension}"))
        fs.sort()
        f, l = fs[0].stem, fs[-1].stem

        d[box.stem] = {
            k: dt.strptime(datime, "%Y%m%d_%H%M%S").date()
            for k, datime in zip(["first", "last"], [f, l])
        }

    return d


def count_files(dir: Path, only_empty: bool = False) -> int:
    """
    Counts the number of .WAV files in a given directory.

    Args:
        dir (pathlib.Path): The directory to search for .WAV files.
        only_empty (bool): Whether to count only empty files. Default is False.

    Returns:
        int: The number of .WAV files in the directory that meet the criteria.
    """
    files = [
        f.name.split(".")[0]
        for f in dir.glob("*")
        if f.is_file()
        and f.suffix == ".WAV"
        and (f.stat().st_size > -1 if not only_empty else f.stat().st_size == 0)
    ]
    return len(files)


def add_missing_data_info(broods, raw_data_dir: Path) -> int:
    """
    Get information about the number of hours recorded and the amount of missing
    files due to recorder malfunction.

    Args:
        raw_data_dir (pathlib.Path): A Path object pointing to the root
            directory of raw data.

    Returns:
        Tuple[int, str]: A tuple containing the total number of hours recorded
            and a string indicating the amount of missing files as a percentage
            of the total number of recordings.
    """
    pnumdirs = [
        f for d in raw_data_dir.glob("*") for f in d.glob("*") if f.is_dir()
    ]

    recording_dates = {
        p.name: {
            "total": count_files(p),
            "missing": count_files(p, only_empty=True),
        }
        for p in pnumdirs
    }
    missing = sum(
        [
            recording_dates[p]["missing"]
            for p in recording_dates.keys()
            if recording_dates[p]["missing"] > 0
        ]
    )

    # add missing data info to broods:
    # 0 if the pnum is not in the recording dates dict:

    broods.loc[:, ["total_recordings", "missing_recordings"]] = [
        (
            recording_dates[p]["total"],
            recording_dates[p]["missing"],
        )
        if p in recording_dates.keys()
        else (0, 0)
        for p in broods.pnum
    ]

    total = sum([recording_dates[p]["total"] for p in recording_dates.keys()])
    print(
        f"Missing {missing} out of {total} recordings ({missing/total*100:.2f}%)"
    )
    print(
        f"{total // 24 // 365} years, {(total // 24) % 365 // 30} months, "
        f"{(total // 24) % 365 % 30} days"
        f" of recordings ({total} hours)"
    )

    return broods


def add_if_recorded(broods: pd.DataFrame, raw_data_dir: Path) -> pd.DataFrame:
    """
    Add whether a recording device was placed in each nestbox and return the updated broods dataframe.

    Args:
        broods (pd.DataFrame): A pandas dataframe containing brood data.
        raw_data_dir (pathlib.Path): A Path object pointing to the root directory of raw data.

    Returns:
        pd.DataFrame: A pandas dataframe with an additional column indicating whether each nestbox was recorded.
    """
    # Get a list of all the pnums in the raw data folder:
    rec_pnum = [
        f.name
        for d in raw_data_dir.glob("*")
        for f in d.glob("*")
        if f.is_dir()
    ]

    # Add a column to broods where each row is True if the pnum is in the list of recorded pnums:
    broods.loc[:, "recorded"] = [
        True if p in rec_pnum else False for p in broods.pnum
    ]

    return broods


def get_recording_hours(
    pnumdirs: List[Path],
) -> Dict[str, Dict[str, List[int | bool]]]:
    """
    Returns a dictionary containing information about each recording for each
    pnum.

    Args:
        pnumdirs: A list of Path objects, each representing a pnum.

    Returns:
        A dictionary where the keys are pnum names and
        the values are dictionaries with two keys: "hour" and "missing". The
        value associated with "hour" is a list of integers representing the hour
        at which each recording was made. The value associated with "missing" is
        a list of booleans indicating whether each recording is empty or not.
    """
    recording_hours = {}
    for p in pnumdirs:
        recording_hours[p.name] = {
            "hour": [int(f.name.split("_")[1][:2]) for f in p.glob("*.WAV")],
            "missing": [f.stat().st_size == 0 for f in p.glob("*.WAV")],
        }
    return recording_hours


def get_missing_hours(
    recording_hours: Dict[str, Dict[str, List[int | bool]]],
    start: int,
    end: int,
) -> Dict[int, float]:
    """
    Calculates the proportion of missing recordings for each hour within a given
    range of hours.

    Args:
        recording_hours: A dictionary where the keys are pnums and the values
            are dictionaries with two keys: "hour" and "missing". The value
            associated with "hour" is a list of integers representing the hour
            at which each recording was made. The value associated with
            "missing" is a list of booleans indicating whether each recording is
            empty or not.
        start: An integer representing the first hour to include in the output.
        end: An integer representing the last hour to include in the output.

    Returns:
        A dictionary where the keys are integers representing the hour and the
        values are floats representing the proportion of missing recordings for
        that hour.
    """
    missing_dic = {}
    total_h = {}
    for pnum in recording_hours:
        for h, m in zip(
            recording_hours[pnum]["hour"], recording_hours[pnum]["missing"]
        ):
            if h not in missing_dic:
                missing_dic[h] = []
                total_h[h] = 0
            missing_dic[h].append(m)
            total_h[h] += 1
    missing_hours = {
        h: sum(missing_dic[h]) / total_h[h] for h in missing_dic.keys()
    }
    return {k: v for k, v in missing_hours.items() if k >= start and k <= end}


def plot_missing_by_hour(missing_hours: dict) -> None:
    """
    Plots the distribution of missing recordings across the day.

    Args:
    - missing_hours (dict): A dictionary with the proportion of missing recordings for each hour of the day.

    Returns:
    - None.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        missing_hours.keys(),
        missing_hours.values(),
        color="k",
        width=0.5,
        align="center",
    )
    for i, v in enumerate(missing_hours.values()):
        ax.text(
            i + 3,
            v + 0.05,
            f"{v:.3f}%",
            ha="center",
            va="bottom",
            color="k",
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion missing", labelpad=10)
    ax.set_xlabel("Hour of the day", labelpad=10)
    ax.set_title("Distribution of missing recordings across the day", pad=20)
    ax.set_xticks(list(missing_hours.keys()))
    ax.set_xticklabels(list(missing_hours.keys()))
    plt.tight_layout()
    plt.show()


def plot_0byte_files(path: Path) -> None:
    pnumdirs = [f for d in path.glob("*") for f in d.glob("*") if f.is_dir()]
    recording_hours = get_recording_hours(pnumdirs)
    missing_hours = get_missing_hours(recording_hours, 3, 8)
    plot_missing_by_hour(missing_hours)


def add_recording_info(pnumdirs, broods):
    """
    Adds recording information to broods dataframe including the date of first
    and last recording, the delay from the lay_date of first recording, and the
    lay_date of the pnum.

    Args:
    pnumdirs (list): A list of directories containing the recordings for each pnum
    broods (pandas.DataFrame): A dataframe containing information about the broods, including pnum.

    Returns:
    pandas.DataFrame: A copy of the original dataframe with additional columns:
        - first_recorded: the date of the first recording
        - last_recorded: the date of the last recording
        - lay_date: the date the first egg was laid
        - delay: the delay in days between the lay_date and the first recording
    """
    # get the dates of the first and last recordings for each pnum
    rec_dates = get_pnum_first_last_recordings(pnumdirs)

    # create a dataframe from the dictionary of recording dates, convert to datetime
    fldf = pd.DataFrame.from_dict(rec_dates, orient="index").apply(
        pd.to_datetime
    )

    # rename columns to avoid conflict when merging
    fldf.rename(
        columns={"first": "first_recorded", "last": "last_recorded"},
        inplace=True,
    )

    # merge recording information with broods dataframe
    broods = broods.merge(
        fldf, left_on="pnum", right_index=True, how="left", suffixes=("", "")
    )

    # calculate delay in days from lay date to first recording
    broods["delay"] = (broods["first_recorded"] - broods["lay_date"]).dt.days

    return broods.copy()


# ──── COMBINE DATASETS ────────────────────────────────────────────────────────

# Read in datasets
broods, morpho, nestboxes = (
    read_broods_data(),
    read_morphometrics(),
    read_nestboxes(),
)

# ──── RESIDENT / IMMIGRANT INFORMATION ────────────────────────────────────────

morpho, broods = filter_species(morpho, broods, species="g")
fathers = get_ided_males(broods)
broods = add_resident_column(broods, fathers, morpho)
broods = add_natal_box_column(broods, fathers, morpho)
plot_proportion_residents(broods)

# ──── BIRD AGES ───────────────────────────────────────────────────────────────

# Add the ages of birds from the ringing data to the broods data, refining them
# where possible. Certainty varies from near absolute for birds born in wytham
# to low if they weren't and were not caught before completing their first
# moult. This is reflected in the data by assigning 'unknown' in the age column,
# or 'adult' in cases where the bird is at least known to be older tha 1 year.

broods = add_father_age(morpho, fathers, broods)
broods = add_bird_age(broods)
broods = refine_brood_ages(morpho, fathers, broods)
plot_age_classes(broods)

# Get fathers in broods that appear more than once:
fathers = broods.father.dropna().value_counts()
fathers = fathers[fathers > 1].index.values

# are there any duplicate pnums?
broods[broods.duplicated(subset=["pnum"], keep=False)]

# ──── COORDINATES ─────────────────────────────────────────────────────────────

# Add coordinates for each nestbox using the nestboxes df we imported already:
broods = broods.merge(nestboxes, on="nestbox", how="left")

# get unique nestboxes with missing coordinates (x,y) in broods:
missing_coords = broods.query("x.isnull()").nestbox.unique()
print(f"Found {len(missing_coords)} nestboxes with missing coordinates")

# ──── ADD RECORDING INFO ──────────────────────────────────────────────────────

add_if_recorded(broods, DIRS.RAW_DATA.parent)
plot_proportion_recorded(broods)
add_missing_data_info(broods, DIRS.RAW_DATA.parent)

# plot distribution of missing recordings across the day:
plot_0byte_files(DIRS.RAW_DATA.parent)

# Add when it was recorded, between which dates, and how much delay from the
# lay_date for that pnum in the broods dataframe:
pnumdirs = [
    f for d in DIRS.RAW_DATA.parent.glob("*") for f in d.glob("*") if f.is_dir()
]
broods = add_recording_info(pnumdirs, broods)
plot_delay_distribution(broods)


# ──── SAVE DERIVED DATASET ────────────────────────────────────────────────────

# Remove unwanted columns and save

# Remove unnecessary columns:
todrop = [
    "section",
    "species",
    "owner",
    "mixed_species",
    "identified_adults",
    "chick_ids",
    "experiment_codes",
    "num_eggs_manipulated",
    "comments",
    "num_linked_pulli_records",
    "num_linked_adult_records",
    "expected_hatch_date",
    "lay_date_uncertainty",
    "state_code",
    "closed",
    "dead_ringed_chick_ids",
]

broods.drop(columns=todrop, inplace=True)
broods.sort_values("pnum", inplace=True)
broods.to_csv(
    DIRS.BROODS.parent / "main.csv",
    index=False,
)

# filter to only include 2020, 2021, and 2022 data:
print("Quick data check (2020-onwards):")
broods = broods.query("year >= 2020")

print(f"{len(broods)} rows in dataset")
print(
    f"Out of those, {len(broods.query('resident == True'))} "
    "were born in the population"
)
# Print proportion of broods that have IDs / were born in whytham:
print(f"{len(broods[broods['father'].notna()])} birds have ID")
print(
    f"{len(broods[broods['father'].notna()].query('recorded == True'))} "
    "of those with ID were recorded, as well as "
    f"{len(broods[~broods['father'].notna()].query('recorded == True'))} without ID"
)
