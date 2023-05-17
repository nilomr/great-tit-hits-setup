from typing import List
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import pandas as pd
from pykanto.utils.io import with_pbar
from pykanto.utils.paths import ProjDirs
from pathlib import Path


def load_train_df(DIRS: ProjDirs, path_to_df: Path) -> pd.DataFrame:
    """
    Load the split dataframe and update the paths based on the new location of
    the images.
    Args:
        DIRS (ProjDirs): Project directories
        path_to_df (Path): Path to the split dataframe.
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


# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


def save_spectrogram_as_image(
    spectrogram: np.ndarray, output_path: Path
) -> None:
    """
    Save a single spectrogram as a JPG image.
    Args:
        spectrogram (np.ndarray): Input spectrogram array.
        output_path (str): Output file path for the JPG image.
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
    Save spectrograms from paths to numpy arrays to JPG images
    in the output directory.
    Args:
        img_out_dir (Path): Output directory for the JPG images.
        paths (List[Path]): List of paths to spectrograms.
    Returns:
    """
    for path in with_pbar(paths):
        spectrogram = np.load(path)
        filename = Path(path).stem + ".jpg"
        save_spectrogram_as_image(spectrogram, img_out_dir / filename)
