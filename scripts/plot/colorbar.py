import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Create a custom colormap: Red (low) -> White (middle) -> Green (high)
colors = [(1, 0, 0), (1, 1, 1), (0, 1, 0)]  # Red -> White -> Green
cmap = LinearSegmentedColormap.from_list("RedWhiteGreen", colors, N=256)

# Generate example data between 0 and 1
data = np.random.rand(10, 10)  # Values between 0 and 1

# Use TwoSlopeNorm to center the color at 0.5
norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

# Plot the data
fig, ax = plt.subplots()
cax = ax.imshow(data, cmap=cmap, norm=norm)

# Add colorbar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Probability of "Rise"')

plt.show()


# plt.savefig(f"{FIGURE_PATH}/colorbar.png")
