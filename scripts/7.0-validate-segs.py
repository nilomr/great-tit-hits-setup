# ──── DESCRIPTION ─────────────────────────────────────────────────────────────

"""
This script can be used to check the results of the note segmentation procedure
+ calculate the percentage of notes that were correctly segmented.
"""

# ──── IMPORTS ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import tkinter as tk

import matplotlib.pyplot as plt
from config import DIRS, build_projdir
from PIL import Image, ImageTk
from pykanto.utils.io import load_dataset

# ──── # DEFINITIONS ──────────────────────────────────────────────────────────


class ImageLabeler:
    def __init__(self, master, image_paths, element_counts):
        self.master = master
        self.image_paths = image_paths
        self.element_counts = element_counts
        self.current_image_index = 0

        # Create GUI elements
        self.image_label = tk.Label(master)
        self.image_label.pack()
        self.total_label = tk.Label(
            master, text=f"Enter total number of elements"
        )
        self.total_label.pack()
        self.total_entry = tk.Entry(master)
        self.total_entry.pack()
        self.correct_label = tk.Label(
            master, text=f"Enter number of correct elements"
        )
        self.correct_label.pack()
        self.correct_entry = tk.Entry(master)
        self.correct_entry.pack()
        self.continue_button = tk.Button(
            master, text="Continue", command=self.continue_labeling
        )
        self.continue_button.pack()
        self.close_button = tk.Button(
            master, text="Close", command=self.master.destroy
        )
        self.close_button.pack()

        # Load first image
        self.load_image()

    def load_image(self):
        # Load image from file
        image_path = self.image_paths[self.current_image_index]
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)

        # Update image label
        self.image_label.configure(image=photo)
        self.image_label.image = photo

        # Update element label
        total_count = self.element_counts[self.current_image_index]
        self.total_label.configure(
            text=f"Enter total number of elements (original: {total_count})"
        )
        self.correct_label.configure(text=f"Enter number of correct elements:")

    def continue_labeling(self):
        # Get entered element counts
        total_count = int(self.total_entry.get())
        correct_count = int(self.correct_entry.get())

        # Add element counts to list
        self.element_counts[self.current_image_index] = (
            total_count,
            correct_count,
        )

        # Move to next image
        self.current_image_index += 1

        # Check if we've labeled all images
        if self.current_image_index >= len(self.image_paths):
            self.master.destroy()
            return

        # Load next image
        self.load_image()

        # Clear element entry boxes
        self.total_entry.delete(0, tk.END)
        self.correct_entry.delete(0, tk.END)


# ──── MAIN ───────────────────────────────────────────────────────────────────


# Create project directories and load dataset
dataset_name = "great-tit-hits"
(DIRS.RAW_DATA.parent / dataset_name).mkdir(exist_ok=True)
DIRS = build_projdir(dataset_name)
dataset = load_dataset(DIRS.DATASET, DIRS)


# Save a random subset of segmented spectrograms.
sampled_files = dataset.files.sample(n=100, random_state=42)
image_paths = []
element_counts = (
    dataset.data.loc[sampled_files.index]
    .onsets.apply(lambda x: len(x))
    .to_list()
)
outputdir = DIRS.REPORTS / "figures" / "spectrograms" / "checks"
outputdir.mkdir(parents=True, exist_ok=True)

for spec in sampled_files.index:
    fig = plt.figure()
    dataset.plot(spec, segmented=True)
    plt.close(fig)
    plt.savefig(
        outputdir / f"{spec}.jpg",
        bbox_inches="tight",
    )
    image_paths.append(outputdir / f"{spec}.jpg")
    plt.close()


# Create GUI and start
root = tk.Toplevel()
labeler = ImageLabeler(root, image_paths, element_counts)
root.mainloop()


# Print results:
diffs = []
tuples = [tup for tup in labeler.element_counts if isinstance(tup, tuple)]
for tup in tuples:
    diff = abs(tup[0] - tup[1])
    diffs.append(diff)

prop = sum(diffs) / sum([tup[0] for tup in tuples])

print(f"N notes: {sum([tup[0] for tup in tuples])}, Wrong: {sum(diffs)}")
print(f"proportion correct: {(1-prop) * 100}")
