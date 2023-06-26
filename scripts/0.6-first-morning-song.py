# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import ray
import seaborn as sns
from config import build_projdir
from matplotlib import dates as mdates
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.paths import get_file_paths

from greti.wrangle import add_sunrise_columns

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


@ray.remote
def process_file(file: Path) -> Tuple[str, str, dt] | None:
    """Process an XML file and extract the start time of the first song.
    Assumes that sample rate is 48000 Hz.

    Args:
        file: The path of the XML file to be processed.

    Returns:
        A tuple with the ID of the XML data, the stem of the file name and the start
        time of the first song in the file. If the file cannot be parsed or the
        "FIRST" label is not found, returns None.

    Raises:
        Any error that occurs while parsing the XML file.
    """
    sr = 48000
    try:
        xmldata = parse_sonic_visualiser_xml(file)
    except:
        return None
    if "FIRST" not in xmldata.label:
        return None

    t = xmldata.start_times[xmldata.label.index("FIRST")] / sr
    begin = dt.strptime(file.stem, "%Y%m%d_%H%M%S")
    start_time = begin + timedelta(seconds=t)

    return (xmldata.ID, file.stem, start_time)


def process_files(xml_filepaths: List[Path]) -> Dict[str, List[dt]]:
    """Process a list of XML files and populate a dictionary with the start times of
    the first song for each XML data.

    Args:
        xml_filepaths: A list of paths to XML files to be processed.

    Returns:
        A dictionary where the keys are the IDs of the XML data and the values are
        lists of start times of the first song for that XML data, in chronological
        order.

    Raises:
        Any error that occurs while processing the XML files.
    """
    timesdict: Dict[str, List[dt]] = {}
    ray.init(ignore_reinit_error=True)

    # Create a list of remote function calls
    results = [process_file.remote(file) for file in xml_filepaths]

    # Iterate over the results and populate the timesdict
    for result in ray.get(results):
        if result is not None:
            xmldata_id, file_stem, start_time = result

            if xmldata_id not in timesdict:
                # Create a new list of start times for this ID
                timesdict[xmldata_id] = [start_time]
            else:
                # Append to the existing list of start times for this ID
                timesdict[xmldata_id].append(start_time)

    # Sort the lists of start times for each ID in chronological order
    for id in timesdict:
        timesdict[id].sort()

    return timesdict


def extract_times(ids_times_dict: dict) -> list:
    """
    Extracts all values from the dictionary and returns a flattened list of values.

    Args:
    ids_times_dict: A dictionary of ids and corresponding lists of times.

    Returns:
    A flattened list of all times in the dictionary.
    """
    return [t for v in ids_times_dict.values() for t in v]


def extract_ids_and_years(ids_times_dict: dict) -> tuple:
    """
    Extracts keys and years from the dictionary of the same length as times.

    Args:
    ids_times_dict: A dictionary of ids and corresponding lists of times.

    Returns:
    A tuple containing a list of keys from the dictionary and a list of years.
    """
    ids = [k for k, v in ids_times_dict.items() for t in v]
    years = [i[:4] for i in ids]
    return ids, years


def create_dataframe(ids: list, years: list, times: list) -> pd.DataFrame:
    """
    Creates a pandas DataFrame from the times, ids, and years.

    Args:
    ids: A list of ids for each recording.
    years: A list of years for each recording.
    times: A flattened list of all times in the dictionary.

    Returns:
    A pandas DataFrame with columns for ids, years, and times.
    """
    df = pd.DataFrame({"pnum": ids, "year": years, "timedate": times})
    return df


def plot_kde(df, x_col, hue_col, labels=None, cmap="magma"):
    alpha = 0.5
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({"font.size": 16})
    sns.set_style("white")
    if labels is None:
        labels = df[~df[hue_col].isna()][hue_col].unique().astype(int)
    ax = sns.kdeplot(
        data=df,
        x=x_col,
        hue=hue_col,
        fill=True,
        common_norm=False,
        palette=sns.color_palette(cmap, len(labels)),
        linewidth=0,
        alpha=alpha,
        legend=False,
    )
    # add x label:
    plt.xlabel(x_col.capitalize() + " (minutes)", labelpad=10)
    # add y label:
    plt.ylabel("Density", labelpad=10)

    # Place legend in upper left corner inside the plot, and remove box around it
    # create the legend, making sure the order of labels is respected:
    ax.legend(
        handles=[
            Patch(
                color=sns.color_palette(cmap, len(labels))[i],
                label=labels[i],
                alpha=alpha,
            )
            for i in range(len(labels))
        ],
        title=hue_col.capitalize(),
        loc="upper right",
        frameon=False,
    )
    # return the plot so that we can continue to edit it later:
    return ax


def plot_first_song_sunrise(df, year):
    orange = "#d47604"
    blue = "#24888f"
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({"font.size": 16})
    df_year = df[df["year"] == year]
    ax = sns.scatterplot(
        data=df_year, x="date", y="time_min", alpha=0.5, color=blue
    )
    sns.lineplot(
        data=df_year,
        x="date",
        y="time_min",
        estimator="mean",
        ax=ax,
        color=blue,
    )
    sns.scatterplot(
        data=df_year, x="date", y="sunrise_min", alpha=0.5, color=orange, ax=ax
    )
    sns.lineplot(
        data=df_year,
        x="date",
        y="sunrise_min",
        estimator="mean",
        color=orange,
        ax=ax,
    )

    unique_dates = df_year["date"].unique()
    mindate = unique_dates.min()
    maxdate = unique_dates.max()
    plt.xticks(ticks=pd.date_range(mindate, maxdate, freq="5D"))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x / 60)}:{int(x % 60):02}")
    )

    plt.figtext(
        0.47, 0.96, "Sunrise", fontsize="large", color=orange, ha="right"
    )
    plt.figtext(
        0.53,
        0.96,
        "time of first song",
        fontsize="large",
        color=blue,
        ha="left",
    )
    plt.figtext(0.50, 0.96, " and ", fontsize="large", color="k", ha="center")
    # add year to upper right corner:
    plt.figtext(0.98, 0.98, year, fontsize="large", color="k", ha="right")

    plt.xlabel("Date", labelpad=10)
    plt.ylabel("Time", labelpad=10)
    plt.tick_params(axis="both", which="major", pad=10)
    plt.ylim(200, 400)
    sns.despine(left=True, bottom=True, right=True, top=True)
    return ax


def plot_time_vs_date_ind(df: pd.DataFrame, year: str) -> plt.Axes:
    """
    Plots the time of first song vs date for a given year in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        year (str): The year for which the data needs to be plotted.

    Returns:
        plt.Axes: The matplotlib axes object containing the plot.
    """
    # Filter data for the given year
    df_year = df[df["year"] == year]

    # Compute the difference between the first and last time for each pnum
    df_diff = (
        df_year.groupby("pnum")["time_min"]
        .agg(["first", "last"])
        .reset_index()
        .assign(diff=lambda x: x["last"] - x["first"])
    )

    # Assign colors to each pnum based on the difference between first and last time
    colors = sns.color_palette("RdBu", n_colors=len(df_diff))
    pnums = df_diff.sort_values("diff", ascending=False)["pnum"].tolist()
    color_dict = dict(zip(pnums, colors))

    # Assign alpha and linewidth to each pnum based on the count of time_min values
    alpha_dict = df_year.groupby("pnum")["time_min"].count() / 10
    # normalise alpha values between 0.2 and 1:
    alpha_dict = (alpha_dict - alpha_dict.min()) / (
        alpha_dict.max() - alpha_dict.min()
    )
    alpha_dict = alpha_dict * 0.8 + 0.2

    size_dict = df_year.groupby("pnum")["time_min"].count() / 2

    # Create the figure
    plt.rcParams.update({"font.size": 16})
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.rcParams.update({"font.size": 16})

    # Plot the data for each pnum
    for i, (pnum, data) in enumerate(df_year.groupby("pnum")):
        plt.plot(
            data["date"],
            data["time_min"],
            color=color_dict[pnum],
            alpha=alpha_dict[pnum],
            linewidth=size_dict[pnum],
        )

    # Set the x and y labels and add a title
    plt.xlabel("Date", labelpad=10)
    plt.ylabel("Time", labelpad=10)
    plt.title("Time of First Song", pad=10)

    # Format the y-axis to display time in HH:MM format
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x/60):02d}:{int(x%60):02d}")
    )

    # Set the tick parameters and format the x-axis to display dates in MM-DD format
    plt.tick_params(axis="both", which="major", pad=10)
    unique_dates = df_year["date"].unique()
    mindate = unique_dates.min()
    maxdate = unique_dates.max()
    plt.xticks(ticks=pd.date_range(mindate, maxdate, freq="5D"))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.figtext(0.98, 0.98, year, fontsize="large", color="k", ha="right")

    # Remove the spines from the plot
    sns.despine(left=True, bottom=True, right=True, top=True)

    # Return the axes object
    return ax


# ──── SETTINGS ─────────────────────────────────────────────────────────────────

# Create project directories
DIRS = build_projdir("GRETI_2022")

# Find files and annotations
xml_filepaths = get_file_paths(DIRS.RAW_DATA.parent, [".xml"], verbose=True)


# ──── FIND TIME OF FIRST SONG ──────────────────────────────────────────────────

# Using parse_sonic_visualiser_xml from pykanto to parse the xml files, find the
# time of the day of the segment lebelled 'FIRST':

timesdict = process_files(xml_filepaths)
times = extract_times(timesdict)
ids, years = extract_ids_and_years(timesdict)
df = create_dataframe(ids, years, times)
df = add_sunrise_columns(df)

if df.duplicated(subset=["pnum", "date"]).any():
    raise ValueError("There are duplicate date-pnum pairs in the dataframe!")

# save the dataframe to a csv file:
df.to_csv(DIRS.BIRD_DATA / "times.csv", index=False)


# ──── PLOT (SANITY CHECK) ──────────────────────────────────────────────────────

# get median diff_time by ids, taking the first value of years and disregarding
# other columns:
df_median = (
    df[["pnum", "year", "diff_time"]]
    .groupby(["pnum", "year"])
    .median()
    .reset_index()
)

# get global median / means
global_median = df_median["diff_time"].median()
mean_of_means = df_median.groupby("pnum")["diff_time"].mean().mean()
print(f"Global median time to sunrise: {global_median:.2f}")
print(f"Gloabl mean of mean time to sunrise: {mean_of_means:.2f}")


# Plot distribution of time of day for all ids, aggregated by year:
ax = plot_kde(df, "time_min", "year")
# zoom in between 2 am and 8am:
plt.xlim(120, 480)
# add x ticks every 30 minutes:
plt.xticks(ticks=range(120, 480, 60))
# format x ticks as hours:minutes
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{int(x/60):02d}:{int(x%60):02d}")
)
# add title
plt.title("Time of Day", pad=10)
plt.xlabel("Time (UTC)", labelpad=10)
plt.tight_layout()

# save the plot:
plt.savefig(DIRS.FIGURES / "time_of_day.png", dpi=300)


# Plot distribution of time since sunrise for all ids
ax = plot_kde(df_median, "diff_time", "year")
# zoom in between -60 and 60 minutes:
plt.xlim(-60, 60)
# add x ticks every 10:
plt.xticks(ticks=range(-60, 60, 20))
# add title
plt.title("Time From Sunrise", pad=10)
plt.xlabel("Time difference (minutes)", labelpad=10)
plt.tight_layout()

# save the plot:
plt.savefig(DIRS.FIGURES / "time_from_sunrise.png", dpi=300)


# Plot distribution of time of day and sunrise across each year:
for year in ["2021", "2022"]:
    ax = plot_first_song_sunrise(df, year)
    # save plot making sure it is not cropped:
    plt.savefig(
        DIRS.FIGURES / f"first_song_sunrise_{year}.png",
        dpi=300,
        bbox_inches="tight",
    )

# Plot the time of first song vs date for each year, by individual:
for year in ["2021", "2022"]:
    ax = plot_time_vs_date_ind(df, year)
    # save plot making sure it is not cropped:
    plt.savefig(
        DIRS.FIGURES / f"time_vs_date_ind_{year}.png",
        dpi=300,
        bbox_inches="tight",
    )
