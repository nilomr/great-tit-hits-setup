# ──── IMPORTS ──────────────────────────────────────────────────────────────────

import math
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pyrootutils
from config import DIRS
from pykanto.utils.io import copy_xml_files, with_pbar


# ──── FUNCTION DEFITNITIONS ───────────────────────────────────────────────────


def count_segments(sr, xmlfiles):
    point_count = 0
    for xml_file in with_pbar(xmlfiles):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        point_count += len(root.findall(".//point"))
    print(f"Total count of entries: {point_count}")

    total_duration_f = sum(
        int(point.get("duration"))  # Get duration value as integer
        for xml_file in xmlfiles
        for point in ET.parse(xml_file).getroot().findall(".//point")
    )

    print(f"Total duration: {timedelta(seconds=total_duration_f / sr)}")


# ──── MAIN ─────────────────────────────────────────────────────────────────────

# Where are the project and its data?
PROJECT_ROOT = pyrootutils.find_root()

# dataset name
DATASET_ID = "GRETI_2022"
sr = 48000

# Wjere can we find the .xml annotation files?

# DATA_LOCATION = Path("/media/nilomr/SONGDATA/wytham-great-tit"
# DATA_LOCATION = Path("/data/zool-songbird/shil5293/data/wytham-great-tit")
data_loc = Path(
    "/media/nilomr/My Passport/SONGDATA/wytham-great-tit/raw/GRETI_2022/"
)

# Set and create destination if it doesn't exist:
destination = DIRS.RAW_DATA
destination.mkdir(parents=True, exist_ok=True)

# Find all .xml files recursively in DATA_LOCATION:
xmlfiles = list(data_loc.glob("*/*.xml"))

# Count total number of segments & duration
count_segments(sr, xmlfiles)

# Now copy them:
copy_xml_files(xmlfiles, destination)
