import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# File to save points
points_file = 'portrait_data1.pkl'

# Load previously saved points

points = []

def onclick(event):
    # Check if the click was within the axes
    if event.xdata is not None and event.ydata is not None:
        # Store the (x, y) coordinates in the numpy array
        point = np.array([event.xdata, event.ydata])
        points.append(point)
        # Mark the point on the image
        ax.plot(event.xdata, event.ydata, 'ro')  # Red dot
        plt.draw()

# Load your image
image_path = ('image.jpeg')  # Replace with your image path
img = plt.imread(image_path)
img = np.rot90(img, k=3)
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title('Click to mark points')

# Connect the click event to the onclick function
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

points = np.array(points)
points[:,1]=-points[:,1]
points=points-np.min(points,axis=0)
points=points/np.max(points)
# Save points to a file when the program is closing
with open(points_file, 'wb') as f:
    pickle.dump(points, f)

print("Points saved:", points)