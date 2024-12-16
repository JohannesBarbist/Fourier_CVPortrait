import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def make_fourier_portrait(n_size=2 ** 10, diameter=200, pad_size=10, points_file = 'portrait_data.pkl', out_points_file = 'portrait_data_fourier.pkl', show_plots=False):


    # Load points from the file
    if os.path.exists(points_file):
        with open(points_file, 'rb') as f:
            points = pickle.load(f)

    radius = int(diameter / 2)
    orig_len = len(points)

    temp_points = np.zeros((points.shape[0]+2*pad_size,2))
    temp_points[:pad_size]= points[0]
    temp_points[-pad_size:]= points[-1]
    temp_points[pad_size:-pad_size]=points
    points = temp_points

    temp_points = np.zeros((n_size, 2))
    for i in range(2):
        temp_points[:,i] = np.interp(np.linspace(0, 1, num=n_size), np.linspace(0, 1, num=len(points)), points[:, i])
    points = temp_points

    if show_plots:
        # Prepare the figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot X coordinates on the top
        ax1.plot(range(len(points)), points[:, 0], marker='o', color='red')
        ax1.set_title('X Coordinates of Points')
        ax1.set_ylabel('X values')
        ax1.grid()

        # Plot Y coordinates on the bottom
        ax2.plot(range(len(points)), points[:, 1], marker='o', color='blue')
        ax2.set_title('Y Coordinates of Points')
        ax2.set_xlabel('Sample Number')
        ax2.set_ylabel('Y values')
        ax2.grid()
        # Plot layout adjustments
        plt.tight_layout()
        plt.show()

    #win = np.bartlett(len(points))[:,np.newaxis]
    #points = points*win
    fft_points = np.fft.fft2(points)

    fft_points = np.fft.fftshift(fft_points)  # Shift zero frequency component to center
    temp_points = np.zeros_like(fft_points)
    mid = int(n_size/2)
    temp_points[mid-radius:mid+radius] = fft_points[mid-radius:mid+radius]
    fft_points = temp_points
    fft_points = np.fft.ifftshift(fft_points)  # Inverse shift

    rec_points = np.fft.ifft2(fft_points)
    rec_points = np.abs(rec_points).astype(np.float32)
    remove_pad = int(pad_size/orig_len*len(rec_points))
    rec_points = rec_points[remove_pad:-remove_pad]
    rec_points = rec_points-np.min(rec_points, axis=0)
    rec_points = rec_points/np.max(rec_points, axis=0)*np.max(points,axis=0)


    if show_plots:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # Plot X coordinates on the top
        ax1.plot(range(len(rec_points)), rec_points[:, 0], marker='o', color='red')
        ax1.set_title('X Coordinates of Points')
        ax1.set_ylabel('X values')
        ax1.grid()

        # Plot Y coordinates on the bottom
        ax2.plot(range(len(rec_points)), rec_points[:, 1], marker='o', color='blue')
        ax2.set_title('Y Coordinates of Points')
        ax2.set_xlabel('Sample Number')
        ax2.set_ylabel('Y values')
        ax2.grid()
        plt.tight_layout()
        plt.show()


    with open(out_points_file, 'wb') as f:
        pickle.dump(rec_points, f)

if __name__=="__main__":
    make_fourier_portrait(show_plots=True)
