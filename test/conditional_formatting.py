"""
Script to test the conditional formatting of the excel sheet
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Data from the LaTeX table
data = {
    "Ablation": [
        "Full",
        "w/o Crypto Factor Expert",
        "w/o Technical Expert",
        "w/o Market Factor Expert",
        "w/o News Expert",
        "w/o collab",
    ],
    "Cumulative": [1.5425, 1.1395, 1.1354, 1.3968, 1.3576, 1.4926],
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Conditional formatting: White to Blue
cmap_lighter = LinearSegmentedColormap.from_list("white_to_blue", ["white", "#478ecc"])
norm = plt.Normalize(df["Cumulative"].min(), df["Cumulative"].max())

# Generating a plain LaTeX table
latex_code = df.to_latex(
    index=False,
    caption="Ablation Study Cumulative Results",
    label="tab:ablation_results",
)


# Alternatively, simulate conditional formatting (if required)
def simulate_latex_color_formatting(value):
    rgba = cmap_lighter(norm(value))
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )
    return f"\\cellcolor[HTML]{{{hex_color[1:]}}} {value:.4f}"


# Apply formatting to cumulative column
df["Cumulative"] = df["Cumulative"].apply(simulate_latex_color_formatting)

# Convert the styled DataFrame to LaTeX
latex_code_with_colors = df.to_latex(
    escape=False,
    index=False,
    caption="Ablation Study Cumulative Results with Colors",
    label="tab:ablation_results_with_colors",
)
df
print(latex_code_with_colors)