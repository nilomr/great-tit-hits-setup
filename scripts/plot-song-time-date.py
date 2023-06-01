import colorsys
import datetime
import typing
from ast import Dict
from typing import Dict, List

import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from config import DIRS, build_projdir
from mpl_toolkits.mplot3d import Axes3D
from pytz import timezone


def preprocess_data(song_df):
    """
    Preprocesses the song DataFrame by adding new columns for year, April day,
    and time, and adds centered day (by median) and times (by median).

    Parameters:
        song_df (pandas.DataFrame): The DataFrame containing the song data.

    Returns:
    pandas.DataFrame: The preprocessed DataFrame.
    """
    # Add columns for year, April day, and time
    song_df["year"] = song_df["datetime"].dt.year
    song_df["april_day"] = (
        song_df["datetime"]
        - pd.to_datetime(song_df["year"].astype(str) + "-04-01")
    ).dt.days + 1  # April 1 = 1
    song_df["time"] = (
        song_df["datetime"].dt.hour * 60 + song_df["datetime"].dt.minute
    )  # Time in minutes since midnight

    # Group the DataFrame by year
    grouped = song_df.groupby("year")

    # Subtract the mean (median for time, 2020 was more long-tailed) value of each variable for each year from the corresponding values
    song_df["april_day_centred"] = grouped["april_day"].transform(
        lambda x: x - int(x.median())
    )
    song_df["time_centred"] = grouped["time"].transform(
        lambda x: x - x.median()
    )
    return song_df


def back_transform_dates(df: pd.DataFrame) -> Dict[int, str]:
    """
    Given a pandas DataFrame with an 'april_day_centred' column, returns a new DataFrame with a column of back-transformed dates.
    """
    # Get the median april day
    median_april_day = df["april_day"].median()

    # Create a dictionary of the back-transformed april_day_centred values from
    df["april_day_back"] = df["april_day_centred"] + median_april_day

    # Create a dictionary of the back-transformed april_day_centred values df
    datedf = (
        df[["april_day_centred", "april_day_back"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(by="april_day_centred")
    )

    # Convert april_day_back to day of the year and then to a date (values before
    # the first of April are negative):
    datedf["april_day_back"] = datedf["april_day_back"].apply(
        lambda x: datetime.datetime(2020, 4, 1) + datetime.timedelta(days=x)
    )

    # Change the format of the date to 'Apr 1' or 'Apr 2' etc.
    datedf["april_day_back"] = datedf["april_day_back"].apply(
        lambda x: x.strftime("%b %d")
    )

    datedict = datedf.set_index("april_day_centred").to_dict()["april_day_back"]

    # Get the minimum and maximum keys in the dictionary
    min_key = min(datedict)
    max_key = max(datedict)

    # Convert the date string to a datetime object
    date_format = "%b %d"
    start_date = datetime.datetime.strptime(datedict[min_key], date_format)

    # Create a new dictionary with the filled in dates
    filled_dates = {}
    current_key = min_key
    current_date = start_date

    while current_key <= max_key:
        if current_key in datedict:
            filled_dates[current_key] = datedict[current_key]
        else:
            filled_dates[current_key] = current_date.strftime(date_format)
        current_key += 1
        current_date += datetime.timedelta(days=1)

    return filled_dates


def create_intervals(
    song_df, interval_size
) -> typing.Tuple[pd.DataFrame, Dict[float, str], Dict[int, str]]:
    """
    Creates time intervals of a specified size and assigns each song to an interval.

    Parameters:
        song_df (pandas.DataFrame): The DataFrame containing the song data.
        interval_size (int): The size of the time intervals in minutes.

    Returns:

        pandas.DataFrame: The DataFrame with an additional "interval" column
            containing the interval labels.

        pandas.DataFrame: The DataFrame containing the intervals and their
            corresponding times.

    """
    # Create labels for the intervals
    interval_labels = [
        f"{i}:{i+interval_size}"
        for i in range(
            int(song_df["time_centred"].min()),
            int(song_df["time_centred"].max()),
            interval_size,
        )
    ]

    # Assign each song to an interval using pd.cut()
    song_df["interval"] = pd.cut(
        song_df["time_centred"],
        bins=pd.interval_range(
            start=song_df["time_centred"].min(),
            end=song_df["time_centred"].max(),
            freq=interval_size,
        ),
        labels=interval_labels,
    )

    return song_df, back_transform_times(song_df), back_transform_dates(song_df)


def back_transform_times(song_df) -> Dict[float, str]:
    """
    Returns a dictionary containing the intervals and their corresponding times.
    It's a bit convoluted but it works. NOTE: uses the median time across years.
    """
    median_time = song_df["time"].median()
    song_df["t_centred_mins"] = song_df["interval"].apply(
        lambda x: int(x.mid) + median_time
    )
    int_in_mins = song_df[["interval", "t_centred_mins"]]
    int_in_mins = int_in_mins.drop_duplicates().sort_values(by="interval")
    int_in_mins["interval_int"] = list(range(len(int_in_mins)))
    int_in_mins["time"] = int_in_mins["t_centred_mins"].apply(
        lambda x: f"{int(x // 60)}:{int(x % 60):02d}"
    )
    int_in_mins = int_in_mins.reset_index(drop=True).dropna()

    # interpolate interval_key so that there are values for every minute in time

    df = int_in_mins.copy()
    df["time"] = pd.to_datetime(df["time"])
    new_df = pd.DataFrame(
        {
            "time_c": pd.date_range(
                start=df["time"].min(), end=df["time"].max(), freq="1min"
            )
        }
    )
    interval_mapping = {
        interval: i for i, interval in enumerate(df["interval"])
    }
    df["interval_num"] = df["interval"].map(interval_mapping)
    merged_df = pd.merge(
        new_df, df, how="left", left_on="time_c", right_on="time"
    )
    merged_df["interval_fl"] = merged_df["interval_int"].interpolate()
    merged_df["time_c"] = merged_df["time_c"].dt.strftime("%H:%M")

    # create a dictionary from the interval_key
    interval_dict = dict(
        zip(
            merged_df["interval_fl"],
            merged_df["time_c"],
        )
    )

    return interval_dict


def interpolate_values(result, resolution: float = 0.2):
    """
    Reindexes the DataFrame to create any missing rows, interpolates to fill in the missing values,
    calculates a smoothed average of the number of songs per interval for each day,
    and returns the pivoted DataFrame with interpolated intervals.

    Parameters:
        result (pandas.DataFrame): The DataFrame containing the number of songs
            per interval for each day.
        resolution (float): The resolution of the interpolation (default: 0.2,
        in fraction of an interval)

    Returns:
        pandas.DataFrame: The pivoted DataFrame with interpolated intervals.
    """
    # Reindex the DataFrame to create any missing rows
    idx = pd.Index(range(result.index.min(), result.index.max() + 1))
    result = result.reindex(idx)
    result.index.name = "april_day_centred"

    # Interpolate to fill in the missing values
    result = result.interpolate()

    # For each day calculate a smoothed average of the number of songs per interval,
    # and get the coordinates of this line
    smoothed = result.apply(lambda x: x.rolling(3, center=True).mean(), axis=0)

    # Unpivot the columns using melt()
    melted = smoothed.reset_index().melt(
        id_vars="april_day_centred", var_name="interval", value_name="nsongs"
    )

    piv_df = melted.pivot(
        index="interval", columns="april_day_centred", values="nsongs"
    )

    # convert the "interval" column to an integer using a dictionary
    interval_labels = piv_df.index.unique()

    piv_df.index = piv_df.index.map(
        {interval: i for i, interval in enumerate(interval_labels)}
    )

    # Now interpolate interval so that the curve is a bit smoother
    idx = pd.Index(
        np.arange(
            piv_df.index.min(), piv_df.index.max() + resolution, resolution
        )
    )
    piv_df = piv_df.reindex(idx).interpolate(method="cubic")

    piv_df.index.name = "interval"

    long_df = piv_df.reset_index().melt(
        id_vars="interval", var_name="april_day_centred", value_name="nsongs"
    )

    return long_df


def mute_colors(colors: List[str], amount: float = 5):
    """
    Takes a list of hex colors and returns a list of muted hex colors.
    """
    muted_colors = []
    for color in colors:
        rgb = mpcolors.to_rgb(color)
        hsv = colorsys.rgb_to_hsv(*rgb)
        muted_hsv = (hsv[0], hsv[1] / amount, hsv[2])
        muted_rgb = colorsys.hsv_to_rgb(*muted_hsv)
        muted_hex = mpcolors.to_hex(muted_rgb)
        muted_colors.append(muted_hex)
    return muted_colors


# ──── MAIN ───────────────────────────────────────────────────────────────────

# Create project directories and load dataset
dataset_name = "great-tit-hits"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)

song_dir = DIRS.DATASET.parent / f"{DIRS.DATASET_ID}.csv"
song_df = pd.read_csv(song_dir, index_col=0, parse_dates=["datetime"])


song_df = preprocess_data(song_df)
# Define the n-minute intervals
interval_size = 8  # lower to increase resolution
song_df, times_key, dates_key = create_intervals(song_df, interval_size)

# get number of songs per interval for each day
result = song_df.groupby(["april_day_centred", "interval"]).size()
result = result.unstack()
long_df = interpolate_values(
    result, resolution=0.1
)  # change to increase resolution - 0.2 is good, integers fast

# Remove rows where long_df["interval"] is outside the range 0-30
interval_range = (4, 20)
trimmed_df = long_df.loc[
    (long_df["interval"] >= interval_range[0])
    & (long_df["interval"] <= interval_range[1])
]
trimmed_df = trimmed_df.dropna()
trimmed_df["nsongs"] = trimmed_df["nsongs"].astype(int)

# Group by day for sequential plotting
groups = trimmed_df.groupby(["april_day_centred"])

# Select variable to color by
var = "nsongs"
smin = trimmed_df[var].min()
smin = smin + abs(smin) + 2
smax = trimmed_df[var].max()

# Prepare data for mean time of day line
max_nsongs = groups.apply(lambda x: x.loc[x[var].idxmax()])
max_nsongs = max_nsongs.reset_index(drop=True)
p_interval = max_nsongs["interval"].rolling(5, center=True).mean()
apr_day = max_nsongs["april_day_centred"]


# Prepare data for sunrise line
dates = song_df["datetime"].dt.date.unique()
dates = pd.date_range(dates.min(), dates.max()).date
tz = timezone("UTC")
wytham_latlong = (51.769602, -1.327018)
wytham = LocationInfo(
    "Wytham", "England", "UTC", wytham_latlong[0], wytham_latlong[1]
)
sunrise_times = [
    sun(wytham.observer, date, tzinfo=tz)["sunrise"] for date in dates
]

dates_a = [date.strftime("%b %d") for date in dates]
dates_sun = dict(zip(dates_a, sunrise_times))
dates_sun = {k: v.strftime("%H:%M") for k, v in dates_sun.items()}
dates_sub = {k: v for k, v in dates_sun.items() if k in dates_key.values()}
rdk = {v: k for k, v in dates_key.items()}
dates_sub = {rdk[k]: v for k, v in dates_sub.items()}
dates_sub = {k: dates_sub[k] for k in sorted(dates_sub)}
rtk = {v: k for k, v in times_key.items()}
sun_dict = {k: rtk[v] for k, v in dates_sub.items()}
apr_day_sun, sun_interval = zip(*sun_dict.items())


# Plot
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
#  set a washed dark grey background color:
bckcol = "#181d21"
ax.set_facecolor(bckcol)
plt.gcf().set_facecolor(bckcol)

# Define the colors for the custom colormap
mincolor = "#29024e"
colors = [
    mincolor,
    mincolor,
    "#5300a1",
    "#9600a1",
    "#d0008e",
    "#f45d43",
    "#fcae13",
]
colors = mute_colors(colors, amount=1.7)
norm = mpcolors.LogNorm(vmin=1, vmax=smax)
cmap = mpcolors.LinearSegmentedColormap.from_list("custom", colors, N=256)
colors = {k: cmap(norm(k)) for k in sorted(trimmed_df[var].unique())}
colors[0] = colors[1]


for group in groups:
    x = group[1]["april_day_centred"].values
    y = group[1]["interval"].values
    z = group[1]["nsongs"].values

    N = len(z)
    for i in range(N - 1):
        ax.plot(
            x[i : i + 2],
            y[i : i + 2],
            z[i : i + 2],
            color=colors[z[i]],
            clip_on=False,
            linewidth=2,
            zorder=-i,
        )

# Plot mean line
ax.plot(
    apr_day,
    p_interval,
    np.ones(len(p_interval)),
    color="white",
    linewidth=2,
    zorder=-100,
    clip_on=False,
)

# plot sun line
ax.plot(
    apr_day_sun,
    sun_interval,
    np.ones(len(sun_interval)),
    color="yellow",
    linewidth=2,
    zorder=-100,
    clip_on=False,
)


x_scale, y_scale, z_scale = (
    1.7,
    0.5,
    0.6,
)  # set the scaling factors for each axis

ax.get_proj = lambda: np.dot(
    Axes3D.get_proj(ax), np.diag([x_scale, y_scale, z_scale, 1])
)

# rotate to face the camera more, but keep centered on the origin
ax.azim = 140
ax.dist = 6
ax.elev = 25
ax.set_proj_type("ortho")

# ax.view_init(elev=20, azim=45)
ax.set_ylim(interval_range)
ax.set_xlabel("Date", labelpad=20)
ax.set_ylabel("Time of Day")
ax.set_zlabel("Number of Songs")

# set the ticks and labels
yticks = {k: v for k, v in times_key.items() if v.endswith("0")}
yticks = list(yticks.keys())[::2]
ylabels = [times_key[k] for k in yticks]
# get ticks every 5 days from the dates_key
xticks = list(dates_key.keys())[::5]
xlabels = [dates_key[k] for k in xticks]

# ticks and labels for the z axis every 500 starting at 0
zticks = np.arange(0, smax, 500)
zlabels = [str(i) for i in zticks]


# now set these as the yticks and labels
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, rotation=0)
ax.set_ylim(interval_range)

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=0)

ax.set_zticks(zticks)
ax.set_zticklabels(zlabels, rotation=0)


# set the tick labels and font sizes for the x, y, and z axes
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_tick_params(labelsize=10, colors="white")
    axis.label.set_color("white")
    axis.label.set_size(12)

# set the major axes to white
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.line.set_color("white")

# remove the background color
for axis in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    axis.fill = False

# add vertical colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, shrink=0.4, aspect=10, pad=0)
# change labels to non scientific notation:
cbar.set_label("Number of Songs", rotation=270, labelpad=15, color="white")
cbar.ax.tick_params(labelsize=10, colors="white")


# inverse the x-axis
ax.invert_xaxis()
ax.invert_yaxis()

# remove all grid lines except for the x-axis
ax.grid(False)

# export as svg
plt.savefig(
    DIRS.REPORTS / "figures" / "time-of-song-slices.svg",
    transparent=True,
    bbox_inches="tight",
)
plt.show()
