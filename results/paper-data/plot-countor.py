import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# Function to extract the exp line for a specific structure
def extract_exp_line(file_path, level=3.35):
    # Load the data
    data = np.loadtxt(file_path, delimiter="\t", usecols=(1, 2, 6))
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    
    # Extract points where z is close to the specified level (exp line)
    exp_line_points = []
    for xi, yi, zi in zip(x, y, z):
        if np.isclose(zi, level, atol=0.2):  # You can adjust the tolerance (atol) as needed
            exp_line_points.append((xi, yi))

    return np.array(exp_line_points)

# List of your file paths (modify as necessary)
files = [
    "/Users/zahrakarimi/Desktop/project/plot/data-contour/c22_up_oopccd-7-july.txt",
    "/Users/zahrakarimi/Desktop/project/plot/data-contour/c28_up_oopccd-7-july.txt",
    "/Users/zahrakarimi/Desktop/project/plot/data-contour/c30_up_oopccd-7-july.txt",
    "/Users/zahrakarimi/Desktop/project/plot/data-contour/c32_up_oopccd-7-july.txt",
    "/Users/zahrakarimi/Desktop/project/plot/data-contour/c36_up_oopccd-7-july.txt"
]

# List of corresponding energy levels (exp lines) for each structure
exp_lines = [3.35, 3.38, 3.02, 2.85, 2.30]

# Create subplots for each structure
fig, axes = plt.subplots(1, len(files), figsize=(15, 6))  # Create a grid of subplots

# Loop through each structure and plot
for i, (file, exp_line) in enumerate(zip(files, exp_lines)):
    # Load the data for the current structure
    data = np.loadtxt(file, delimiter="\t", usecols=(1, 2, 6))
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    
    # Create contour plot
    ax = axes[i]
    contour = ax.tricontourf(x, y, z, levels=20, cmap="plasma")
    
    # Extract the structure name from the file path (e.g., 'c22', 'c28', etc.)
    structure_name = os.path.basename(file).split('_')[0]  # This extracts the part before the first underscore
    
    # Set the title to the structure name
    ax.set_title(f"Structure {structure_name}")
    
    # Overlay the specific exp line (contour line)
    ax.tricontour(x, y, z, levels=[exp_line], colors='black', linestyles='dashed', linewidths=2)
    
    # Customize the plot with labels
    ax.set_xlabel("u' (eV)")
    ax.set_ylabel("t (eV)")

    # Create a custom legend handle for the dashed line
    dashed_line = Line2D([0], [0], color='black', linestyle='dashed', label=f"Exp line {exp_line} eV")
    ax.legend(handles=[dashed_line], loc='best')

# Add colorbar for the last subplot (the colorbar is the same for all plots)
fig.colorbar(contour, ax=axes[-1], label='Excitation Energy (eV)')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()