import ast
import json
import warnings
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from PIL import Image
from pykanto.dataset import KantoData
from pykanto.utils.compute import with_pbar
from pykanto.utils.io import save_to_jsons
from pykanto.utils.paths import ProjDirs


def load_train_df(DIRS: ProjDirs, path_to_df: Path) -> pd.DataFrame:
    """
    Load the split dataframe and update the paths based on the new location of
    the images.

    Args:
        DIRS (ProjDirs): Project directories path_to_df (Path): Path to the
        split dataframe.
    Returns:
        pd.DataFrame: The dataframe with updated paths.
    """
    df = pd.read_csv(path_to_df, index_col=0)
    img_out_dir = DIRS.ML / "images"
    df["path"] = df["path"].map(
        lambda x: Path(str(img_out_dir / Path(x).stem) + ".jpg")
    )

    assert all(
        df.path.apply(lambda x: True if x.exists() else False)
    ), "Some paths do not point to images"

    return df


# ──── FUNCTION DEFINITIONS ────────────────────────────────────────────────────


def save_spectrogram_as_image(
    spectrogram: np.ndarray, output_path: Path
) -> None:
    """
    Save a single spectrogram as a JPG image.

    Args:
        spectrogram (np.ndarray): Input spectrogram array. output_path (str):
        Output file path for the JPG image.
    """
    # Normalize the spectrogram
    normalized_spec = (spectrogram - np.min(spectrogram)) / (
        np.max(spectrogram) - np.min(spectrogram)
    )

    # Apply colormap to the normalized spectrogram
    colormap = cm.magma(normalized_spec)

    # Convert the spectrogram to RGB image
    image = Image.fromarray((colormap * 255).astype(np.uint8)).convert("RGB")

    # Save the image as JPG
    image.save(output_path)


def save_ml_data(img_out_dir: Path, paths: List[Path]) -> None:
    """
    Save spectrograms from paths to numpy arrays to JPG images in the output
    directory.

    Args:
        img_out_dir (Path): Output directory for the JPG images. paths
        (List[Path]): List of paths to spectrograms.

    Returns:
    """
    for path in with_pbar(paths):
        spectrogram = np.load(path)
        filename = Path(path).stem + ".jpg"
        save_spectrogram_as_image(spectrogram, img_out_dir / filename)


def update_json_paths(dataset: KantoData):
    """Updates the paths of JSON files in the dataset to a new directory.

    Args:
        dataset: The dataset object containing JSON files.

    Returns:
        A list of updated JSON file paths.

    Raises:
        ValueError: If an old path does not exist or if the paths are already
        updated or if a new path does not exist or if the number of old and new
        paths do not match.
    """
    # Checking if paths are already updated
    if dataset._jsonfiles == [
        dataset.DIRS.SEGMENTED / path.parent.name / path.name
        for path in dataset._jsonfiles
    ]:
        warnings.warn("Paths already updated")
        return None

    new_jsonpaths = []
    # Updating paths
    for path in with_pbar(dataset._jsonfiles, desc="Updating paths"):
        if path.parent.parent.name == dataset.DIRS.SEGMENTED.name:
            new_path = dataset.DIRS.SEGMENTED / path.parent.name / path.name
            if not new_path.exists() and new_path.stem in [dataset.data.index]:
                raise ValueError(f"New path {new_path} does not exist")
            new_jsonpaths.append(new_path)

    # get any old paths that have not been updated
    new_jsonpaths_set = {path.name for path in new_jsonpaths}
    not_updated = [
        path.name
        for path in with_pbar(dataset._jsonfiles, desc="Checking old paths")
        if path.name not in new_jsonpaths_set
    ]

    if len(not_updated) > 0:
        print(f"Some old paths ({len(not_updated)}) have not been updated")

    return new_jsonpaths


# ──── SAVE NEW METADATA TO OLD JSON FILES ─────────────────────────────────────

# Update the JSON file paths for the dataset


# remove any JSON files where path.stem is not in dataset.data.index
def update_jsons(dataset):
    """
    Removes bad segment WAV and JSON files from the dataset.

    Args:
        dataset: A `pykanto.dataset.KantoData` object containing the dataset.

    Returns:
        None.

    Raises:
        None.
    """
    dataset._jsonfiles = [
        jfile
        for jfile in dataset._jsonfiles
        if jfile.stem in dataset.data.index
    ]

    new_jsonpaths = update_json_paths(dataset)
    if new_jsonpaths:
        dataset._jsonfiles = new_jsonpaths

    # Check if the JSON files have already been updated
    testjson = [
        path
        for path in dataset._jsonfiles
        if path.name == dataset.data.index[0] + ".JSON"
    ][0]

    with open(testjson, "r") as f:
        jfile = json.load(f)

    # If the columns in the dataset have changed, update the JSON files
    if len(set(dataset.data.columns) - set(jfile.keys())) > 0:
        # Convert datetime column to string so it can be serialized
        dataset.data.datetime = dataset.data.datetime.astype(str)
        save_to_jsons(dataset)
    else:
        # Otherwise, print a message saying there is no need to update the JSON
        # files
        print("No need to update JSON files.")


# ─── REMOVE BAD SEGMENTS ─────────────────────────────────────────────────────


def remove_bad_files(dataset: KantoData):
    """
    Removes bad files from the dataset.
    Args:
        dataset: A KantoData object representing the dataset.
    Returns:
        None
    """
    idxs = dataset.data.index
    to_rm = [jfile for jfile in dataset._jsonfiles if jfile.stem not in idxs]
    if len(to_rm) == 0:
        print("No files to remove.")
        return

    # wavs to remove
    w_to_rm = [
        jfile.parent.parent / "WAV" / (jfile.stem + ".wav") for jfile in to_rm
    ]

    for jfile in with_pbar(to_rm, desc="Removing bad segments"):
        jfile.unlink()

    for wfile in with_pbar(w_to_rm, desc="Removing bad wavs"):
        if wfile.exists():
            wfile.unlink()
        else:
            print(f"{wfile} does not exist.")


def find_list_cols(df):
    columns_with_format = []
    for column in df.columns:
        if (
            df[column].dtype == "object"
            and df[column]
            .str.contains(r"^\[\d+(\.\d+)?( \d+(\.\d+)?)*\]$")
            .any()
        ):
            columns_with_format.append(column)
    return columns_with_format


def convert_arrays_to_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts NumPy arrays in a pandas DataFrame to lists.
    Args:
        df: A pandas DataFrame.
    Returns:
        A pandas DataFrame with NumPy arrays converted to lists.
    Raises:
        TypeError: If a column cannot be converted to a list.
    Example:
        >>> df = pd.DataFrame({
        ...     'col1': [1, 2, 3],
        ...     'col2': ['[1, 2, 3]', '[4, 5, 6]', '[7, 8, 9]'],
        ...     'col3': [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        ... })
        >>> convert_arrays_to_lists(df)
           col1       col2          col3
        0     1  [1, 2, 3]  [1, 2, 3]
        1     2  [4, 5, 6]  [4, 5, 6]
        2     3  [7, 8, 9]  [7, 8, 9]
    """
    # find columns containing NumPy arrays
    array_cols = df.applymap(lambda x: isinstance(x, np.ndarray)).any()

    # convert NumPy arrays to lists
    for col in array_cols[array_cols].index:
        try:
            df[col] = df[col].apply(lambda x: x.tolist())
        except Exception as exc:
            raise TypeError(f"{col} could not be converted to a list") from exc

    return df


def load_song_dataframe(song_dir: Path) -> pd.DataFrame:
    """
    Loads a song DataFrame from a CSV file. Converts lists in string format
    to numpy arrays.
    Args:
        song_dir: A string representing the path to the CSV file.
    Returns:
        A pandas DataFrame containing the song data.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or does not contain the required columns.
    Example:
        >>> df = load_song_dataframe("path/to/song.csv")
    """
    # Define the required columns and the function to parse lists
    c = ["onsets", "offsets", "silence_durations", "unit_durations"]

    def parse_lists(x):
        try:
            return ast.literal_eval(x)
        except:
            return x

    # Load the CSV file into a DataFrame
    try:
        song_df = pd.read_csv(
            song_dir,
            index_col=0,
            parse_dates=["datetime"],
            converters={col: parse_lists for col in c},
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {song_dir}") from e
    except ValueError as e:
        raise ValueError(
            f"File is empty or does not contain the required columns: {song_dir}"
        ) from e

    # Convert the c columns to NumPy arrays
    song_df[c] = song_df[c].apply(lambda x: np.array(x))

    return song_df


def prepare_shape_data(
    perimeter_path: Path, nestboxes_path: Path, broods_path: Path
) -> Tuple:
    """
    Read data from files and return the prepared data.
    Args:
    - perimeter_path (str): The path to the shapefile of the perimeter.
    - nestboxes_path (str): The path to the CSV file of the nestboxes.
    - broods_path (str): The path to the CSV file of the recorded nestboxes.
    Returns:
    - Tuple containing the following data:
        - perimeter (geopandas.GeoDataFrame): GeoDataFrame containing the perimeter data
        - nestboxes (pandas.DataFrame): DataFrame containing the nestboxes data
        - broods (pandas.DataFrame): DataFrame containing the recorded nestboxes data
    """
    # read in the shapefile
    perimeter = gpd.read_file(perimeter_path).iloc[0:1]

    # import nestbox coordinates
    nestboxes = pd.read_csv(nestboxes_path)

    # import data on recorded nestboxes
    broods = pd.read_csv(broods_path)

    return perimeter, nestboxes, broods
