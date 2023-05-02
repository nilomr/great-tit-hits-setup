# ──── IMPORTS ──────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import pyrootutils
import ray
import typer
from pykanto.dataset import KantoData
from pykanto.parameters import Parameters
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs

# ──── SETTINGS ─────────────────────────────────────────────────────────────────

app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(
    dataset_id: str = typer.Option(
        ...,
        "--dataset-id",
        "-d",
        help="Name of the dataset to be created",
    ),
    data_folder: str = typer.Option(
        ...,
        "--data-folder",
        "-f",
        help=(
            "Name of the folder containing the segmented data, "
            "assumed to be under ` project_root / 'data' / 'segmented' `"
        ),
    ),
):
    typer.echo(f"Chosen dataset name: {dataset_id}")

    # Ray settings
    if "ip_head" in os.environ:
        typer.echo("ip_head in os.environ")
        redis_password = sys.argv[1]
        ray.init(address=os.environ["ip_head"], _redis_password=redis_password)
        typer.echo(ray.cluster_resources())

    # Create a ProjDirs object for the project, including location of raw data to
    # use
    project_root = pyrootutils.find_root()
    segmented_dir = project_root / "data" / "segmented" / data_folder
    DIRS = ProjDirs(project_root, segmented_dir, dataset_id, mkdir=True)

    # Define parameters
    params = Parameters(
        # Spectrogramming
        window_length=1024,
        hop_length=128,
        n_fft=1024,
        num_mel_bins=224,
        sr=22050,
        top_dB=65,  # top dB to keep
        lowcut=2000,
        highcut=10000,
        # Segmentation
        max_dB=-30,  # Max threshold for segmentation
        dB_delta=5,  # n thresholding steps, in dB
        silence_threshold=0.1,  # Between 0.1 and 0.3 tends to work
        max_unit_length=0.4,  # Maximum unit length allowed
        min_unit_length=0.02,  # Minimum unit length allowed
        min_silence_length=0.001,  # Minimum silence length allowed
        gauss_sigma=3,  # Sigma for gaussian kernel
        # general settings
        dereverb=True,
        song_level=True,
        subset=None,
        verbose=False,
        num_cpus=80,
    )

    # ──── BUILD AND SAVE DATASET ───────────────────────────────────────────────────

    # np.random.seed(123)
    # random.seed(123)
    dataset = KantoData(
        DIRS,
        parameters=params,
        overwrite_dataset=False,
        random_subset=None,
        overwrite_data=False,
    )

    out_dir = DIRS.DATA / "datasets" / dataset_id / f"{dataset_id}.db"
    dataset = load_dataset(out_dir, DIRS)

    dataset.segment_into_units()
    dataset.get_units()
    dataset.cluster_ids(min_sample=15)
    dataset.prepare_interactive_data()
    dataset.to_csv(dataset.DIRS.DATASET.parent)


if __name__ == "__main__":
    app()
