# ──── IMPORTS ──────────────────────────────────────────────────────────────────

from pathlib import Path
from pykanto.utils.paths import (
    ProjDirs,
    link_project_data,
)
import pyrootutils

# ──── PROJECT SETUP ────────────────────────────────────────────────────────────

# Name of the dataset to use or create
DATASET_ID = "GRETI_2022"

# Where are the project and its data?
PROJECT_ROOT = pyrootutils.find_root()

# Where are the project data?
# This could be an external drive, etc

DATA_LOCATION = Path("/media/nilomr/SONGDATA/wytham-great-tit")
# DATA_LOCATION = Path("/media/nilomr/My Passport/SONGDATA/wytham-great-tit")
# DATA_LOCATION = Path("/data/zool-songbird/shil5293/data/wytham-great-tit")

# Create symlink from project to data if it doesn't exist already:
link_project_data(DATA_LOCATION, PROJECT_ROOT / "data")

# Create a ProjDirs object for the project, including location of raw data to
# segment or the already-segmented data
RAW_DATA = PROJECT_ROOT / "data" / "raw" / DATASET_ID  # .lower()
DIRS = ProjDirs(PROJECT_ROOT, RAW_DATA, DATASET_ID, mkdir=True)

# ──── INPUT AND OUTPUT FILE PATHS ──────────────────────────────────────────────


bird_data = DIRS.RESOURCES / "birds"
DIRS.append("BIRD_DATA", bird_data)
DIRS.append("MORPHOMETRICS", bird_data / "morphometrics.csv")
DIRS.append("BROODS", bird_data / "broods.csv")
DIRS.append("MAIN", bird_data / "main.csv")
DIRS.append("NESTBOXES", bird_data / "nestboxes.csv")


# ──── TEST ─────────────────────────────────────────────────────────────────────


def build_projdir(
    dataset_id: str,
    data_dir: Path = Path("/media/nilomr/SONGDATA/wytham-great-tit"),
) -> ProjDirs:
    """
    Returns a ProjDirs object for the given dataset ID.

    Args:
        dataset_id (str): The name of the dataset to use or create.
        data_dir (Path): The location of the data. This could be an external drive, etc

    Returns:
        ProjDirs: A ProjDirs object for the project, including the location
            of raw data to segment or the already-segmented data.
    """
    # Where are the project and its data?
    PROJECT_ROOT = pyrootutils.find_root()

    # Where are the project data?
    # This could be an external drive, etc

    # Create symlink from project to data if it doesn't exist already:
    link_project_data(data_dir, PROJECT_ROOT / "data")

    # Create a ProjDirs object for the project, including location of raw data to
    # segment or the already-segmented data
    RAW_DATA = PROJECT_ROOT / "data" / "raw" / dataset_id
    dirs = ProjDirs(PROJECT_ROOT, RAW_DATA, dataset_id, mkdir=True)

    # Set up additional input and output file paths
    bird_data = dirs.RESOURCES / "birds"
    dirs.append("BIRD_DATA", bird_data)
    dirs.append("MORPHOMETRICS", bird_data / "morphometrics.csv")
    dirs.append("BROODS", bird_data / "broods.csv")
    dirs.append("MAIN", bird_data / "main.csv")
    dirs.append("NESTBOXES", bird_data / "nestboxes.csv")

    return dirs
