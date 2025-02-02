# imports
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


def load_data(acc_file_path):
    """
    Load accelerometer data from a pickle file.

    Parameters:
        acc_file_path (str): Path to the acclerometer pickel file.

    Returns:
        np.ndarray: Loaded accelerometer data.
    """
    try:
        with open(acc_file_path, 'rb') as f:
            accelerometer_data = pickle.load(f)
        
        if isinstance(accelerometer_data, list):
            accelerometer_data = np.array(accelerometer_data)
        
        return accelerometer_data
    except Exception as e:
            print(f"Error loading file {acc_file_path}: {e}")
            return None


def visualize_data_(accelerometer_data):
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
    
    x = np.arrange(accelerometer_data.shape[1])
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


