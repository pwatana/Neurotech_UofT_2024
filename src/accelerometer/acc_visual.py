import mne
import pickle
import numpy as np

from spectrum import aryule, arma2psd
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal.windows import hamming
from scipy.signal import find_peaks, firwin
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import windows
from scipy.signal.windows import hamming


def vis_data(accelerometer_data):
    """
    Visualize accelerometer data for all axes. 
    
    Parameters:
        accelerometer_data (np.ndarray): Accelerometer data with shape (3, time_samples).
    """
    if accelerometer_data is None:
        print("No data to visualize.")
        return
    if len(accelerometer_data.shape) != 2 or accelerometer_data.shape[0] != 3:
        print("Invalid data shape. Expected (3, time_samples).")
        return
    
    x = np.arange(accelerometer_data.shape[1])
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']
    plt.figure(figsize=(12, 6))

    for i in range(3):
        plt.plot(x, accelerometer_data[i], label=axis_labels[i])

    plt.title('Accelerometer Data (All Axes)')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def vis_x_axis(accelerometer_data):
    """
    Visualize accelerometer data for the X-axis. 
    
    Parameters:
        accelerometer_data (np.ndarray): Accelerometer data with shape (3, time_samples).
    """
    if accelerometer_data is None:
        print("No data to visualize.")
        return
    if len(accelerometer_data.shape) != 2 or accelerometer_data.shape[0] != 3:
        print("Invalid data shape. Expected (3, time_samples).")
        return
    
    x = np.arange(accelerometer_data.shape[1])
    plt.figure(figsize=(12, 6))
    plt.plot(x, accelerometer_data[0], label='X-axis')
    plt.title('Accelerometer Data (X-axis)')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def vis_ma_filter(drift_removed_data):
    """
    Visualize accelerometer data after applying moving average filter.
    """
    # Plot all axes after drift removal
    # Plot all axes after drift removal
    # Labels for the axes
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    # Loop through each axis and plot them separately
    for i, label in enumerate(axis_labels):  # Iterate through axes and their labels
        plt.figure(figsize=(8, 4))  # Adjust figure size
        plt.plot(drift_removed_data[i], label=f'Drift-Removed {label}', color='blue')
        plt.title(f"Drift Removal - {label}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)  # Add grid for better visualization
        plt.tight_layout()  # Prevent overlapping of elements
        plt.show()


def vis_bandpass_filter(filtered_data):
    """
    Visualize accelerometer data after applying bandpass filter.
    """
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    # Loop through each axis and plot separately
    for i, label in enumerate(axis_labels):  # Loop over indices and labels
        plt.figure(figsize=(10, 6))  # Create a new figure for each axis
        plt.plot(filtered_data[i], label=f"Bandpass-Filtered {label} (1–30 Hz)")
        plt.title(f"Bandpass Filtering - {label}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)  # Add a grid
        plt.tight_layout()  # Adjust layout
        plt.show()