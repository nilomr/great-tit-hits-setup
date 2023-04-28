# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import ujson as json

sys.path.append("..")
from config import build_projdir
from pykanto.utils.compute import with_pbar
from pykanto.utils.paths import get_file_paths


# NOTE: for 2020 is a bit more involved: first get the time of the first 'GRETI
# LOW QUALITY' segment, rename as 'FIRST', then get the time of the first
# 'FIRST'. complication: need to do this from the original metadata files from
# AviaNZ. Check the code to export the metadata from AviaNZ in pykanto, already
# have a method for this.


def create_data_dict(
    data_file: Path,
    sample_rate: int,
    valid_labels: List[str],
) -> Dict[str, Dict[str, int]]:
    """
    Creates a dictionary with data extracted from a .data file, filtered by
    valid_labels.

    Args:
        data_file (str): The path to the data file.
        sample_rate (int): The sample rate of the data.
        valid_labels (List[str]): A list of valid labels to filter the data by.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary where the keys are segment IDs
            and the values are dictionaries with
            the following keys: 'label', 'start', 'end', 'duration', 'min_freq',
            'max_freq'.
    """
    data_dict = {}
    with open(data_file, "r") as data:
        segments = json.load(data)[1:]
        segments = sorted(segments, key=lambda x: x[0])

        i = 0
        for seg in segments:
            label = seg[4][0]["species"]
            if label not in valid_labels:
                continue
            start, end, min_freq, max_freq = seg[:4]
            start = int(start * sample_rate)
            end = int(end * sample_rate)
            duration = end - start

            data_dict[f"{Path(data_file).stem}_{i}"] = {
                "label": "",
                "start": start,
                "end": end,
                "duration": duration,
                "min_freq": min_freq,
                "max_freq": max_freq,
            }

            i += 1

    return data_dict if i > 0 else {}


def write_xml(data_dict: Dict, out_file: Path):
    sv = ET.Element("sv")
    data = ET.SubElement(sv, "data")
    model = ET.SubElement(
        data,
        "model",
        {
            "id": "10",
            "name": "",
            "sampleRate": "48000",
            "start": "0",
            "end": "0",
            "type": "sparse",
            "dimensions": "2",
            "resolution": "1",
            "notifyOnAdd": "true",
            "dataset": "9",
            "subtype": "box",
            "minimum": "",
            "maximum": "",
            "units": "Hz",
        },
    )
    dataset = ET.SubElement(data, "dataset", {"id": "9", "dimensions": "2"})
    for i, seg in data_dict.items():
        ET.SubElement(
            dataset,
            "point",
            {
                "frame": str(seg["start"]),
                "value": str(seg["min_freq"]),
                "duration": str(seg["duration"]),
                "extent": str(seg["max_freq"] - seg["min_freq"]),
                "label": seg["label"],
            },
        )
    display = ET.SubElement(sv, "display")
    layer = ET.SubElement(
        display,
        "layer",
        {
            "id": "11",
            "type": "boxes",
            "name": "Boxes",
            "model": "10",
            "verticalScale": "0",
            "colourName": "White",
            "colour": "#ffffff",
            "darkBackground": "true",
        },
    )
    tree = ET.ElementTree(sv)
    tree.write(
        out_file,
        encoding="UTF-8",
        xml_declaration=True,
    )


# Create project directories
DIRS = build_projdir("GRETI_2020")

# Find all the .data files
data_files = get_file_paths(DIRS.RAW_DATA, [".data"], verbose=True)

valid_labels = ["GRETI_HQ"]
sampleRate = 48000


for data_file in with_pbar(data_files):
    file_dict = create_data_dict(data_file, sampleRate, valid_labels)
    if not file_dict:
        continue
    out_file = data_file.parent / str(data_file.stem.split(".")[0] + ".xml")
    write_xml(file_dict, out_file)


def get_xml_points(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return len(root.findall("data/dataset/point"))


# Are there any xml files with no points?
xml_files = get_file_paths(DIRS.RAW_DATA, [".xml"], verbose=True)
xml_files_empty = [x for x in xml_files if get_xml_points(x) == 0]
print(f"Found {len(xml_files_empty)} xml files with no points")
