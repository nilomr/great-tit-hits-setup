# ──── ADD DATETIME: ────────────────────────────────────────────────────────────


from datetime import datetime

import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from pytz import timezone


def add_datetime(dataset: pd.DataFrame, sample_rate: int = 48000):
    """
    Add datetime column to the dataset based on index values.
    Args:
        dataset (pd.DataFrame): The dataset containing the index values.
    Returns:
        pd.DataFrame: The dataset with the datetime column added.
    Raises:
        ValueError: If the dataset is empty or does not contain the required
        columns.
    """

    # Define the sample rate

    if dataset.empty or "datetime" in dataset.columns:
        raise ValueError("Invalid dataset")

    # Split the index string into separate columns
    dataset[["date", "time"]] = (
        dataset.index.to_series().str.split("_", expand=True).iloc[:, 1:3]
    )

    # Convert the date column to datetime objects
    dataset["date"] = pd.to_datetime(dataset["date"], format="%Y%m%d")

    # Extract time from string and store as datetime time
    dataset["time"] = (
        dataset["time"]
        .str.extract(r"(\d{2})(\d{2})(\d{2})", expand=True)
        .apply(lambda x: ":".join(x), axis=1)
    )

    # Convert the start frame to start time in seconds
    dataset["start_s"] = pd.to_numeric(dataset["start"]) / sample_rate

    # Add time + start_s to get start time object, and combine with date
    dataset.insert(
        2,
        "datetime",
        (dataset["date"].astype(str) + " " + dataset["time"]).apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        )
        + pd.to_timedelta(dataset["start_s"], unit="s"),
    )

    # Remove date, time, and start_s columns
    dataset.drop(columns=["date", "time", "start_s"], inplace=True)

    return dataset


def add_sunrise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns for sunrise time, sunrise time in minutes, and difference from
    sunrise time.

    Args:
        df: A pandas DataFrame with columns for ids, years, and times.

    Returns:
        The original DataFrame with added columns for sunrise time, sunrise
        time in minutes, and difference from sunrise time.
    """
    # create date column from timedate column:
    df.loc[:, "date"] = df["timedate"].apply(lambda x: x.date())

    # create times_min column from timedate column:
    df.loc[:, "time_min"] = df["timedate"].apply(
        lambda x: x.hour * 60 + x.minute
    )

    # create a timezone object for the location of the recordings:
    tz = timezone("UTC")
    wytham_latlong = (51.769602, -1.327018)
    # create a location object for the location of the recordings:
    wytham = LocationInfo(
        "Wytham", "England", "UTC", wytham_latlong[0], wytham_latlong[1]
    )

    # create a list of sunrise times for each date in the dataframe:
    sunrise_times = [
        sun(wytham.observer, date, tzinfo=tz)["sunrise"] for date in df["date"]
    ]
    # create a list of sunrise times in minutes:
    sunrise_min = [t.hour * 60 + t.minute for t in sunrise_times]

    # add sunrise and sunrise_min to the dataframe, without time zone:
    df.loc[:, "sunrise"] = sunrise_times
    df.loc[:, "sunrise_min"] = sunrise_min

    # calculate the time since sunrise in minutes:
    df.loc[:, "diff_time"] = df["time_min"] - df["sunrise_min"]

    return df
