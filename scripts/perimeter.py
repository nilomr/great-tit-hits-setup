import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("white")

# read in the shapefile
from config import DIRS

file = DIRS.RESOURCES / "wytham_map" / "perimeter.shp"
perimeter = gpd.read_file(file).iloc[0:1]

# import nestbox coordinates from DIRS.NESTBOXES:
nestboxes = pd.read_csv(DIRS.NESTBOXES)

# import data on recorded nestboxes from DIRS.BROODS:
broods = pd.read_csv(DIRS.MAIN)


# plot the nestboxes and perimeter
fig, ax = plt.subplots(figsize=(8, 8))
perimeter.plot(ax=ax, alpha=0.5, edgecolor="k", linewidth=0, color="grey")
plt.scatter(
    nestboxes.x,
    nestboxes.y,
    color="#242424",
    alpha=1,
    linewidth=0,
    s=10,
    label="Not Recorded",
)
ax.set_xlabel("Easting", fontsize=12, labelpad=10)
ax.set_ylabel("Northing", fontsize=12, labelpad=10)
ax.set_title("Wytham Woods Nestboxes", fontsize=14, pad=20)
ax.tick_params(axis="both", which="major", pad=2, labelsize=10)
sns.despine(ax=ax, left=True, bottom=True)
plt.show()
