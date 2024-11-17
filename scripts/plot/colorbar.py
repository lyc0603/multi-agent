"""
Script to plot colorbar
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from environ.constants import FIGURE_PATH

# Step 1: Create a green-white-red colormap
colors = [(1, 0, 0), (1, 1, 1), (0, 1, 0)]  # Green -> White -> Red
green_white_red_cmap = LinearSegmentedColormap.from_list("GreenWhiteRed", colors, N=256)

# Generate some example data
data = np.random.rand(10, 10) * 2 - 1  # Values between -1 and 1 for better contrast

# Step 2: Plot the data
fig, ax = plt.subplots()
cax = ax.imshow(data, cmap=green_white_red_cmap)

# Add colorbar on the right
cbar = fig.colorbar(cax, ax=ax, orientation="vertical")
cbar.set_label('Probability of "Rise"')  # Optional label for the colorbar

# Show the plot
plt.show()

plt.savefig(f"{FIGURE_PATH}/colorbar.png")
