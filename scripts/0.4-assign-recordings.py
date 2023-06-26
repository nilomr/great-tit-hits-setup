# ─── DESCRIPTION ─────────────────────────────────────────────────────────────

"""
Code to read and combine relevant information available for each breeding
attempt in a nest box at which we tried to record songs.
"""

# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import warnings
from datetime import datetime as dt
from pathlib import Path
from typing import Dict

import pandas as pd
from config import DIRS
from pykanto.utils.compute import with_pbar

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


def read_broods_data():
    """
    Reads the breeding data from a CSV file and returns a pandas DataFrame.

    Returns:
        A pandas DataFrame containing the breeding data.
    """
    date_cols = ["lay_date", "clear_date", "expected_hatch_date", "hatch_date"]
    broods_data = pd.read_csv(DIRS.BROODS, parse_dates=date_cols)
    return broods_data


def get_pnums_from_file():
    """
    Reads the pnums from a CSV file and returns a dictionary mapping box codes
    to pnums.

    Returns:
        A dictionary mapping box codes to pnums.
    """
    if "segmented" not in str(DIRS.RAW_DATA):
        return None

    warnings.warn(
        "Trying to read existing pnums from file (raw data folder missing)"
    )

    # TODO: filter df by recorded == True before getting values!
    # Otherwise will get last attempt if > 1 attempts
    d = dict(pd.read_csv(bird_data_outdir)[["pnum", "box"]].values)
    d_pnum = {v: k for k, v in d.items()}
    print(f"Found {len(d_pnum)} pnums")

    return d_pnum


def get_box_first_last_recordings(
    olddirs: list[Path], extension: str = ".WAV"
) -> Dict[str, Dict[str, datetime]]:
    """
    Given a list of directories containing recordings, returns a dictionary mapping
    box codes to the datetime of their first and last recordings.

    Args:
        olddirs: A list of directories containing recordings.
        extension: The file extension of the recordings.

    Returns:
        A dictionary mapping box codes to the datetime of their first and last
        recordings.
    """
    d = {}
    for box in olddirs:
        box.stem
        fs = list(box.glob(f"*{extension}"))
        fs.sort()
        f, l = fs[0].stem, fs[-1].stem

        d[box.stem] = {
            k: dt.strptime(datime, "%Y%m%d_%H%M%S")
            for k, datime in zip(["first", "last"], [f, l])
        }

    # Works with box code or pnum
    d = {(k if len(k) <= 5 else k[5:]): v for k, v in d.items()}

    return d


def find_pnum_for_nestboxes(
    d: Dict[str, Dict[str, datetime]], year: int
) -> Dict[str, str]:
    """
    Given a dictionary mapping box codes to the datetime of their first and last
    recordings, and a year, returns a dictionary mapping box codes to their
    corresponding pnums.

    Args:
        d: A dictionary mapping box codes to the datetime of their first and
            last recordings.
        year: The year of the breeding data to use.

    Returns:
        A dictionary mapping box codes to their corresponding pnums.
    """
    broods_data = read_broods_data().query("year == @year").copy()
    d_pnum = {}
    missing_pnums = []
    for nestbox, dates in d.items():
        nestboxes = broods_data.query("nestbox == @nestbox").copy()
        if len(nestboxes) > 1:
            nestboxes.sort_values("clear_date", inplace=True)
            k = 0
            while (
                nestboxes.at[nestboxes.index[k], "clear_date"] < dates["last"]
            ):
                k += 1
            pnum = nestboxes.at[nestboxes.index[k], "pnum"]
        elif len(nestboxes) == 1:
            pnum = nestboxes["pnum"].values[0]
        else:
            missing_pnums.append(nestbox)
            pnum = f"{year}1{nestbox}"
        d_pnum[nestbox] = pnum

    if len(missing_pnums) > 0:
        print(
            f"Missing pnums for {missing_pnums}, we generated them "
            "but you might want to check why this is"
        )

    if len(d) != len(d_pnum):
        raise IndexError("Number of boxes does not match number of pnums")

    return d_pnum


def get_pnums_from_raw_data(
    olddirs: list[Path], extension: str = ".WAV"
) -> Dict[str, str]:
    """
    Given a list of directories containing recordings, returns a dictionary mapping
    box codes to their corresponding pnums.

    Args:
        olddirs: A list of directories containing recordings.
        extension: The file extension of the recordings.

    Returns:
        A dictionary mapping box codes to their corresponding pnums.
    """
    d_box_dates = get_box_first_last_recordings(olddirs, extension)
    years = set([v["first"].year for v in d_box_dates.values()])
    if len(years) > 1:
        raise ValueError("More than one year in data")
    year = list(years)[0]

    d_pnum = find_pnum_for_nestboxes(d_box_dates, year)

    return d_pnum


def get_pnums(olddirs: list[Path]):
    """
    Given a list of directories containing recordings, returns a dictionary mapping
    box codes to their corresponding pnums.

    Args:
        olddirs: A list of directories containing recordings.

    Returns:
        A dictionary mapping box codes to their corresponding pnums.
    """
    pnums = get_pnums_from_file()
    if pnums is not None:
        return pnums

    return get_pnums_from_raw_data(olddirs)


def rename_raw_data_folders(
    raw_data_dir: Path,
    dir_mapping: Dict[str, str],
    preview: bool = True,
) -> None:
    """
    Rename raw data folders based on a dictionary mapping old directory names to
    new ones.

    Args:
        raw_data_dir: The path to the root directory containing the raw data folders.
        dir_mapping: A dictionary mapping old directory names to new ones.
        preview: Whether to preview the output before renaming the folders.

    Returns:
        None
    """
    raw_data_dir = Path(raw_data_dir)
    if "segmented" in str(raw_data_dir):
        raise ValueError("Can only rename folders in /raw directory")

    if preview:
        old_dirs = list(raw_data_dir.glob("*"))
        for old_dir in old_dirs:
            if old_dir.stem in dir_mapping:
                new_dir = raw_data_dir / dir_mapping[old_dir.stem]
            else:
                warnings.warn(
                    f"Directory {old_dir} not in dir_mapping, skipping."
                )

        print(f"Would rename {old_dir.name} to {new_dir.name}")
        warnings.warn(
            "If that looks ok, proceed with caution and set "
            "'preview=False' to run this function."
        )

    else:
        old_dirs = list(raw_data_dir.glob("*"))
        for old_dir in with_pbar(old_dirs):
            if old_dir.stem in dir_mapping:
                new_dir = raw_data_dir / dir_mapping[old_dir.stem]
                old_dir.rename(new_dir)


# ──── ASSIGN RECORDIGNS TO BREEDING ATTEMPTS ───────────────────────────────────

# Every year songs recorded at a nest are saved to a `year/boxnumber` folder.
# Once all the breeding data are in, we can link the nestboxes with their
# records. To make it easier to later combine data from multiple years, we will
# rename the folders from their simple names to their 'Pnum', which includes the
# year and the breeding attempt.

olddirs = [f for f in DIRS.RAW_DATA.iterdir() if f.is_dir()]
# NOTE: stop if olddirs already have pnums
d_pnum = get_pnums(olddirs)

rename_raw_data_folders(
    DIRS.RAW_DATA,
    d_pnum,
    preview=True,
)
