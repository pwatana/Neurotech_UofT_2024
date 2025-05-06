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


# In[1]: 
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
    


# In[2]:
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



# In[3]:
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



# In[4]:
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


# In[5]:
def detect_dominant_frequencies(segmented_windows, fs, low_freq=3, high_freq=8, ar_order=10, threshold_divisor=10):
    """
    Detects dominant frequencies for each window using an autoregressive model.
    Applies power spectral density (PSD) estimation and finds the dominant frequency within the 3–8 Hz tremor range.

    Args:
        segmented_windows: 2D array (num_windows x window_samples), segmented signal for one axis.
        fs: Sampling frequency in Hz.
        low_freq: Lower bound of the tremor frequency range (Hz).
        high_freq: Upper bound of the tremor frequency range (Hz).
        ar_order: Order of the autoregressive model.
        threshold_divisor: Divisor for power thresholding.

    Returns:
        dominant_frequencies: 1D array of detected dominant frequencies per segment.
        dominant_powers: 1D array of power values at detected frequencies.
    """
    dominant_frequencies = []
    dominant_powers = []

    for window in segmented_windows:
        try:
            # Step 1: Fit AR Model
            ar_coeffs, noise_variance, _ = arburg(window, ar_order)

            # Step 2: Compute PSD
            psd = arma2psd(ar_coeffs, rho=noise_variance)
            psd = psd[:len(psd) // 2]  # Keep only positive frequencies
            nfft = len(psd)
            freqs = np.linspace(0, fs / 2, nfft)

            # Step 3: Find Peaks in PSD
            peaks, _ = find_peaks(psd)

            if peaks.size > 0:
                max_peak_power = max(psd[peaks])  # Get the strongest peak power
            else:
                dominant_frequencies.append(1)  # Default to 1 Hz if no peaks
                dominant_powers.append(0)
                continue

            # Step 4: Filter Peaks in 3–8 Hz Range
            freq_peaks = freqs[peaks]
            freq_indices = (freq_peaks >= low_freq) & (freq_peaks <= high_freq)
            filtered_freqs = freq_peaks[freq_indices]

            if filtered_freqs.size > 0:
                peak_filtered_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))
            else:
                dominant_frequencies.append(1)
                dominant_powers.append(0)
                continue

            # Step 5: Select the Strongest Peak
            freqs_filtered = freqs[peak_filtered_indices]
            psd_filtered = psd[peak_filtered_indices]
            psd_max_freq_index = np.argmax(psd_filtered)

            # Apply Thresholding (Only Detect if Above Threshold)
            if len(filtered_freqs) > 0 and np.any(psd_filtered[psd_max_freq_index] > max_peak_power / threshold_divisor):
                dominant_frequencies.append(freqs_filtered[psd_max_freq_index])
                dominant_powers.append(psd_filtered[psd_max_freq_index])
            else:
                dominant_frequencies.append(1)
                dominant_powers.append(0)

        except Exception as e:
            print(f"Error in AR model fitting: {e}")
            dominant_frequencies.append(1)
            dominant_powers.append(0)

    return np.array(dominant_frequencies), np.array(dominant_powers)



# In[6]:
def map_windows_to_timesteps(dominant_frequencies_all_axes, window_size, overlap_ratio, n_timesteps):
    """
    Maps window-based dominant frequencies to the original time scale of n_timesteps.
    The dominant frequency for each window segment is upsampled to match the
    length of the original accelerometer data. This results in a time-frequency representation
    of the accelerometer data across the three axes.

    Args:
        dominant_frequencies_all_axes: 2D array (3 x num_windows), dominant frequencies per axis.
        window_size (int): Number of samples per window.
        overlap_ratio (float): Fraction of overlap between consecutive windows.
        n_timesteps (int): Total number of time samples in the original signal.

    Returns:
        time_freq_map: 2D array (3 x n_timesteps), mapped time-frequency representation.
    """
    step_size = int(window_size * (1 - overlap_ratio))  # Step size between windows
    n_windows = dominant_frequencies_all_axes.shape[1]  # Number of windows
    n_channels = dominant_frequencies_all_axes.shape[0]  # Number of axes (X, Y, Z)

    # Initialize output array
    time_freq_map = np.zeros((n_channels, n_timesteps))

    for ch in range(n_channels):  # Process each axis independently
        for i in range(n_windows):
            # Calculate the range of timesteps covered by the current window
            start_idx = i * step_size
            end_idx = min(start_idx + window_size, n_timesteps)
            
            # Assign the dominant frequency to all timesteps in the window range
            time_freq_map[ch, start_idx:end_idx] = dominant_frequencies_all_axes[ch, i]

    return time_freq_map



# In[7]:
def combine_tfr_axes_with_multiplication(time_freq_map, low_freq=3, high_freq=8):
    """
    Combines the time-frequency representations (TFR) of X, Y, and Z axes using element-wise multiplication.

    Args:
        time_freq_map: 2D array (3 x num_timesteps), mapped TFR signals for each axis.
        low_freq: Lower bound of the tremor frequency range (Hz).
        high_freq: Upper bound of the tremor frequency range (Hz).

    Returns:
        combined_tfr_signal: 1D array, combined TFR signal across all axes.
    """
    # Normalize the time-frequency representation to range [0, 1]
    def normalize_tfr_signal(tfr_signal, low_freq, high_freq):
        return np.clip((tfr_signal - low_freq) / (high_freq - low_freq), 0, 1)

    # Normalize all three axes
    normalized_signals = [normalize_tfr_signal(axis_signal, low_freq, high_freq) for axis_signal in time_freq_map]

    # Perform element-wise multiplication across axes
    combined_tfr_signal = np.prod(normalized_signals, axis=0)

    return combined_tfr_signal


# In[8]:
def denormalize_tfr_signal(normalized_signal, low_freq=3, high_freq=8):
    """
    Converts a normalized TFR signal (0-1 scale) back into the actual frequency range (3-8 Hz).

    Args:
        normalized_signal: 1D array, the combined normalized TFR signal.
        low_freq: Lower bound of the tremor frequency range (Hz).
        high_freq: Upper bound of the tremor frequency range (Hz).

    Returns:
        denormalized_signal: 1D array, converted back to real frequency values (3-8 Hz).
    """
    return normalized_signal * (high_freq - low_freq) + low_freq


# In[9]:
def apply_tremor_threshold(denormalized_signal, threshold=3.5):
    """
    Applies a threshold to classify tremor (T > 3.5 Hz) vs. non-tremor (T < 3.5 Hz).

    Args:
        denormalized_signal: 1D array, the denormalized TFR signal (in Hz).
        threshold: Frequency threshold for tremor detection (default: 3.5 Hz).

    Returns:
        tremor_labels: 1D binary array (1 = Tremor, 0 = Non-Tremor).
    """
    tremor_labels = (denormalized_signal > threshold).astype(int)
    return tremor_labels

# In[10]:

def generate_rectangular_pulse(tremor_labels):
    """
    Converts tremor detection labels into a rectangular pulse representation.

    Args:
        tremor_labels: 1D binary array (1 = Tremor, 0 = Non-Tremor).

    Returns:
        pulse_signal: 1D array, rectangular pulse representation.
    """
    pulse_signal = np.zeros_like(tremor_labels)  # Initialize signal

    # Identify start and end indices of tremor events
    is_tremor = False
    for i in range(len(tremor_labels)):
        if tremor_labels[i] == 1 and not is_tremor:
            is_tremor = True  # Tremor starts
        elif tremor_labels[i] == 0 and is_tremor:
            is_tremor = False  # Tremor ends
        
        if is_tremor:
            pulse_signal[i] = 1  # Maintain tremor region

    return pulse_signal

# In[11]:

def detect_tremor_edges_and_durations(rectangular_pulse, time_index, min_duration=3, fs=100):
    """
    Detects tremor onset and duration based on the rectangular pulse representation.
    
    Args:
        rectangular_pulse: 1D array of binary tremor presence (1 = tremor, 0 = no tremor).
        time_index: 1D array of corresponding time indices.
        min_duration: Minimum duration (in seconds) for a detected tremor episode.
        fs: Sampling frequency (Hz).
    
    Returns:
        tremor_onsets: List of detected tremor onset times.
        tremor_durations: List of corresponding tremor durations.
    """
    # Detect edges where the signal transitions from 0 → 1 (tremor start) and 1 → 0 (tremor end)
    tremor_starts = np.where((rectangular_pulse[:-1] == 0) & (rectangular_pulse[1:] == 1))[0] + 1
    tremor_ends = np.where((rectangular_pulse[:-1] == 1) & (rectangular_pulse[1:] == 0))[0] + 1

    # Ensure each start has an end
    if len(tremor_ends) > 0 and len(tremor_starts) > 0:
        if tremor_ends[0] < tremor_starts[0]:  # If there's an end before a start, remove it
            tremor_ends = tremor_ends[1:]
        if len(tremor_starts) > len(tremor_ends):  # If a tremor starts but doesn't end, remove last start
            tremor_starts = tremor_starts[:-1]

    # Compute durations and filter out short tremors
    tremor_durations = (tremor_ends - tremor_starts) / fs  # Convert from samples to seconds
    valid_indices = np.where(tremor_durations >= min_duration)[0]

    # Store only valid tremor onsets and durations
    tremor_onsets = time_index[tremor_starts[valid_indices]]
    tremor_durations = tremor_durations[valid_indices]

    return tremor_onsets, tremor_durations


# In[12]: debugging and troubleshoot purposes

import numpy as np

def detect_tremor_edges_and_durations(rectangular_pulse, time_index, min_duration=3, fs=100):
    """
    Detects tremor onset and duration based on the rectangular pulse representation.
    """
    print("\n==== INPUT CHECK ====")
    print("Rectangular Pulse:", rectangular_pulse)
    
    # Detect edges
    tremor_starts = np.where((rectangular_pulse[:-1] == 0) & (rectangular_pulse[1:] == 1))[0] + 1
    tremor_ends = np.where((rectangular_pulse[:-1] == 1) & (rectangular_pulse[1:] == 0))[0] + 1

    print("\n==== EDGE DETECTION ====")
    print("Raw Tremor Starts:", tremor_starts)
    print("Raw Tremor Ends:", tremor_ends)

    # Ensure each start has an end
    if len(tremor_ends) > 0 and len(tremor_starts) > 0:
        if tremor_ends[0] < tremor_starts[0]:
            tremor_ends = tremor_ends[1:]
        if len(tremor_starts) > len(tremor_ends):
            tremor_starts = tremor_starts[:-1]

    print("\n==== AFTER CORRECTION ====")
    print("Corrected Tremor Starts:", tremor_starts)
    print("Corrected Tremor Ends:", tremor_ends)

    # Compute durations
    tremor_durations = (tremor_ends - tremor_starts) / fs
    print("\n==== DURATIONS BEFORE FILTERING ====")
    print("Tremor Durations:", tremor_durations)

    # Filter out short tremors
    valid_indices = np.where(tremor_durations >= min_duration)[0]
    print("\n==== VALID INDICES ====")
    print("Valid Indices:", valid_indices)

    tremor_onsets = time_index[tremor_starts[valid_indices]]
    tremor_durations = tremor_durations[valid_indices]

    print("\n==== FINAL OUTPUT ====")
    print("Tremor Onsets:", tremor_onsets)
    print("Tremor Durations:", tremor_durations)

    return tremor_onsets, tremor_durations

# Example data simulation
fs = 100  # Sampling frequency in Hz
min_duration = 3  # Minimum tremor duration in seconds
num_samples = 100000  # Total number of samples

# Generate a synthetic rectangular pulse signal for demonstration
np.random.seed(42)
rectangular_pulse = np.zeros(num_samples)
tremor_segments = np.random.choice(range(500, num_samples-500, 5000), size=10, replace=False)

for start in tremor_segments:
    end = min(start + np.random.randint(300, 800), num_samples)
    rectangular_pulse[start:end] = 1

# Assuming `time_index` represents the time for each sample
time_index = np.arange(num_samples) / fs  # Convert to seconds

# Detect tremor onset times and durations
tremor_onsets, tremor_durations = detect_tremor_edges_and_durations(rectangular_pulse, time_index, min_duration, fs)

# Visualization of detected tremor events
plt.figure(figsize=(12, 6))
plt.plot(time_index, rectangular_pulse, label="Tremor Rectangular Pulse", color="blue")

# Mark tremor onset times
plt.scatter(tremor_onsets, np.ones_like(tremor_onsets), color='red', label="Tremor Onset", marker='o', s=50)

plt.title("Detected Tremor Onset and Duration")
plt.xlabel("Time (seconds)")
plt.ylabel("Tremor Presence (1 = Yes, 0 = No)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Display detected tremor onsets and durations
tremor_events_df = pd.DataFrame({"Tremor Onset (s)": tremor_onsets, "Duration (s)": tremor_durations})
