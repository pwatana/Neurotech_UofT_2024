# Import dependencies
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
    


def moving_average_filter(data, window_size=50):
    """
    Removes drift by subtracting a moving average from the signal.
    Args:
        data: 1D array, accelerometer data for one axis.
        window_size: Number of samples for the moving average window.
    Returns:
        Data with drift removed.
    """
    b = np.ones(window_size) / window_size  # Moving average coefficients
    a = [1]  # No feedback
    smoothed_data = lfilter(b, a, data)
    drift_removed = data - smoothed_data  # Subtract moving average from original data
    return drift_removed



def bandpass_filter(data, fs, lowcut, highcut, numtaps=101):
    """
    Apply a bandpass filter to the data.
    
    Parameters:
        data (np.ndarray): Input data.
        fs (float): Sampling frequency in Hz.
        lowcut (float): Low cut-off frequency in Hz.
        highcut (float): High cut-off frequency in Hz.
        numtaps (int): Number of taps for the FIR filter (high value = sharper frequency cutoff).
        
    Returns:
        Filtered data with only frequenceis in the [lowcut, highcut] range.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b = firwin(numtaps, [low, high], pass_zero=False)
    filtered_data = filtfilt(b, [1.0], data)
    return filtered_data



def segment_with_hamming(signal, window_size, overlap, fs):
    """
    Segments the signal into overlapping windows and applies a Hamming window.

    Args:
        signal: 1D array, input signal (e.g., bandpass-filtered data).
        window_size: Duration of each window in seconds.
        overlap: Fraction of overlap between windows (e.g., 0.5 for 50% overlap).
        fs: Sampling frequency in Hz.

    Returns:
        A 2D array of segmented and windowed data, where each row is a window.
    """
    samples_per_window = int(window_size * fs)  # Convert window size to samples
    step_size = int(samples_per_window * (1 - overlap))  # Step size for overlapping
    hamming_window = windows.hamming(samples_per_window)  # Hamming window

    # Create windows
    num_windows = (len(signal) - samples_per_window) // step_size + 1
    segments = np.array([
        signal[i:i + samples_per_window] * hamming_window
        for i in range(0, len(signal) - samples_per_window + 1, step_size)
    ])

    return segments