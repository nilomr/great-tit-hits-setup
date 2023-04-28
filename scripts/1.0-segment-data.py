# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

# find any xml files with no points:
from typing import List, Tuple

from config import DIRS, build_projdir
from pykanto.signal.segment import ReadWav, segment_files_parallel
from pykanto.utils.compute import with_pbar
from pykanto.utils.custom import parse_sonic_visualiser_xml
from pykanto.utils.io import make_tarfile
from pykanto.utils.paths import get_file_paths

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


def check_paths(wav_filepaths, xml_filepaths):
    """Check that the paths are as expected"""
    wav_pnums = set([f.parent.parent.name for f in wav_filepaths])
    if not len(wav_pnums) == 3:
        raise ValueError(
            f"Expected 3 pnums, found {len(wav_pnums)}: {wav_pnums}"
        )

    # double check that neither wav_filepaths nor xml_filepaths contain any files
    # with forbidden names (i.e. containing letters):
    import re

    for f in wav_filepaths:
        if re.search(r"[a-zA-Z]", f.stem):
            warnings.warn(
                f"Found file with letters in name: {f}, will remove it"
            )

    for f in xml_filepaths:
        if re.search(r"[a-zA-Z]", f.stem):
            warnings.warn(
                f"Found file with letters in name: {f}, will remove it"
            )

    # now remove any files with letters in the name:
    wav_filepaths = [
        f for f in wav_filepaths if not re.search(r"[a-zA-Z]", f.stem)
    ]
    xml_filepaths = [
        f for f in xml_filepaths if not re.search(r"[a-zA-Z]", f.stem)
    ]
    return wav_filepaths, xml_filepaths


def get_wavs_w_annotation(wav_filepaths, xml_filepaths):
    """Get all pairs of wav and xml files that have a matching file name and
    parent folder"""
    wav_filepaths = set(wav_filepaths)
    xml_filepaths = set(xml_filepaths)
    files_to_segment = []
    for wav in wav_filepaths:
        xml = wav.parent / f"{wav.stem}.xml"
        if xml in xml_filepaths:
            files_to_segment.append((wav, xml))
    return files_to_segment


def get_xml_points(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return len(root.findall("data/dataset/point"))


def check_empty_xml_files(files_to_segment: List[Tuple[Path, Path]]) -> None:
    empty_xmls = []
    for _, xml in with_pbar(files_to_segment):
        l = get_xml_points(xml)
        if l == 0:
            empty_xmls.append(xml)
    if empty_xmls:
        raise IndexError(
            f"There are {len(empty_xmls)} empty xml files! {empty_xmls}"
        )
    else:
        print("All xml files have at least one entry")


# ──── SEGMENT FILES ────────────────────────────────────────────────────────────


# ──── FIND FILES TO SEGMENT ────────────────────────────────────────────────────

# Create project directories
# Create a WYTHAM_GRETIS dir to trick ProjDirs (expects an existing directory)
(DIRS.RAW_DATA.parent / "GREAT-TIT-HITS").mkdir(exist_ok=True)
DIRS = build_projdir("GREAT-TIT-HITS")

# Find files and annotations
wav_filepaths, xml_filepaths = [
    get_file_paths(DIRS.RAW_DATA.parent, [ext], verbose=True)
    for ext in [".WAV", ".xml"]
]

wav_filepaths, xml_filepaths = check_paths(wav_filepaths, xml_filepaths)
files_to_segment = get_wavs_w_annotation(wav_filepaths, xml_filepaths)

# order tuples of (wav, xml) filepaths:
files_to_segment = sorted(files_to_segment, key=lambda x: x[0].parent.name)

# Count all points in all xml files and check that there are no empty ones.
n_points = sum(
    [get_xml_points(xml) for wav, xml in with_pbar(files_to_segment)]
)
print(f"Found {n_points} points in {len(files_to_segment)} files")
check_empty_xml_files(files_to_segment)

# ──── SEGMENT FILES ────────────────────────────────────────────────────────────

# %%

for file in wav_filepaths[:2]:
    m = ReadWav(file).get_metadata()

segment_files_parallel(
    files_to_segment,
    DIRS,
    resample=22050,
    parser_func=parse_sonic_visualiser_xml,
    min_duration=0.5,
    min_freqrange=100,
    min_amplitude=5000,
    labels_to_ignore=["NOISE", "FIRST"],
)

# Compress segmented folder annotations to upload via scp
out_dir = DIRS.SEGMENTED.parent / f"{DIRS.SEGMENTED.name}.tar.gz"
in_dir = DIRS.SEGMENTED
make_tarfile(in_dir, out_dir)
