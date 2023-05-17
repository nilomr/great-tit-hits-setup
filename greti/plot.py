import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_class_count_histogram(dataset):
    """
    Plots a histogram of counts by class ID with a log x-axis and customized ticks.
    Args:
        dataset (pandas.DataFrame): The dataset containing the data.
    Returns:
        None
    """

    # Group the data by "class_id" and count occurrences
    counts = dataset.data.groupby("class_id")["class_id"].count()

    # Set a custom color palette with organic colors
    colors = ["#c2d6d6", "#b7d8b0", "#d8d8a8", "#dbb0a8", "#d6c2d6"]
    sns.set_palette(colors)

    plt.figure(figsize=(8, 4))
    plt.hist(
        counts,
        bins=np.logspace(np.log10(counts.min()), np.log10(counts.max()), 30),
        log=False,
    )
    plt.xscale("log")
    plt.xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    plt.minorticks_on()
    plt.gca().xaxis.set_tick_params(
        which="minor", bottom=True, top=False, labelbottom=False
    )
    plt.gca().spines["top"].set_visible(False)
    plt.grid(False)
    plt.xlabel("Counts")
    plt.ylabel("Frequency")
    plt.title("Histogram of Counts by Class ID")
    plt.show()
