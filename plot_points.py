import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from points_to_fourier import make_fourier_portrait

make_fourier_portrait(n_size=2**10, diameter=2**6)
# File to load points from
points_file = 'portrait_data_fourier.pkl'

# Load points from the file
if os.path.exists(points_file):
    with open(points_file, 'rb') as f:
        points = pickle.load(f)

else:
    print(f"File \"{points_file}\". Please run the previous program to create a file.")
    points = []

if len(points) == 0:
    print("No points to plot.")
else:
    # Convert points list to a numpy array for ease of use
    points_array = np.array(points)

    fig, ax = plt.subplots(figsize=(8, 10))

    # Connect the points with lines
    ax.plot(points_array[:, 0], points_array[:, 1], linestyle='-', color='black', linewidth=2)

    # Optional: Set the limits of the plot based on your points
    ax.set_xlim(np.min(points_array[:, 0]) - 0.1, np.max(points_array[:, 0]) + 0.1)  # Adjust limits appropriately
    ax.set_ylim(np.min(points_array[:, 1]) - 0.1, np.max(points_array[:, 1]) + 0.1)  # Adjust limits appropriately
    ax.set_aspect('equal', 'box')  # Keep the aspect ratio equal

    # Hide the axes
    ax.axis('off')

    # Show the plot
    plt.show()