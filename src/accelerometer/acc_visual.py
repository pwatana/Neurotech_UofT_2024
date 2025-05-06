import matplotlib.pyplot as plt


# In[1]:
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



# In[2]:
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



# In[3]:
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



# In[4]:
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



# In[5]:
def vis_segment_hamming(axis_labels, segmented_signals):
    """
    Visualize segmented data using a Hamming window.
    """
    for i, label in enumerate(axis_labels):
        plt.figure(figsize=(8, 4))
        plt.plot(segmented_signals[i][0], label=f"First Window - {label}")
        plt.title(f"Hamming Window Applied to First Window ({label})")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()




# In[6]:
def dominant_freq_vis(dominant_frequencies_all_axes):
    """
    Visualize the dominant frequencies detected for each axis.
    
    Parameters:
        dominant_freqs (array): array of dominant frequencies for each axis.
    """
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

    for i, label in enumerate(axis_labels):
        plt.figure(figsize=(10, 6))
        plt.plot(dominant_frequencies_all_axes[i], label=f"Dominant Frequencies ({label})", color="blue")
        plt.axhline(y=1, color="red", linestyle="--", label="Non-Tremor Baseline (1 Hz)")
        plt.axhline(y=3, color="green", linestyle="--", label="Lower Bound (3 Hz)")
        plt.axhline(y=8, color="purple", linestyle="--", label="Upper Bound (8 Hz)")
        plt.title(f"Dominant Frequencies Detected (3–8 Hz) - {label}")
        plt.xlabel("Window Index")
        plt.ylabel("Frequency (Hz)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# In[7]:

def time_frequency_vis(mapped_time_frequency_representation):
    """
    Visualize the time-frequency representation of the accelerometer data.
    
    Parameters:
        mapped_time_frequency_representation (array): Time-frequency representation of the data.
    """
    axis_labels = ["X-axis", "Y-axis", "Z-axis"]
    for i, label in enumerate(axis_labels):
        plt.figure(figsize=(10, 6))
        plt.plot(mapped_time_frequency_representation[i], label=f"TFR - {label}", color="blue")
        plt.title(f"Time-Frequency Representation ({label})")
        plt.xlabel("Time Index")
        plt.ylabel("Frequency (Hz)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



# In[8]:

def combined_axes_vis(combined_tfr_signal):
    """
    Visualize the combined time-frequency representation of all axes.
    
    Parameters:
        combined_tfr_signal (array): Combined time-frequency representation of all axes.
    """
    # Visualization: Combined Time-Frequency Representation
    plt.figure(figsize=(10, 6))
    plt.plot(combined_tfr_signal, label="Combined TFR Signal", color="blue")
    plt.axhline(y=0.5, color="red", linestyle="--", label="Detection Threshold (Normalized)")
    plt.title("Combined Time-Frequency Representation Across Axes")
    plt.xlabel("Time Index")
    plt.ylabel("Normalized Combined Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[9]:
def denormalized_tfr_vis(denormalized_tfr_signal):
    """
    Visualize the denormalized time-frequency representation.
    
    Parameters:
        denormalized_tfr_signal (array): Denormalized time-frequency representation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(denormalized_tfr_signal, label="Denormalized TFR Signal (Hz)", color="blue")
    plt.axhline(y=3.5, color="red", linestyle="--", label="Tremor Threshold (3.5 Hz)")
    plt.title("Denormalized Time-Frequency Representation (TFR)")
    plt.xlabel("Time Index")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# In[10]:
def threshold_vis(tremor_labels):
    """
    Visualize the tremor labels.
    
    Parameters:
        tremor_labels (array): Array of tremor labels.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(tremor_labels, label="Tremor Detection", color="blue")
    plt.axhline(y=0.5, color="red", linestyle="--", label="Tremor Threshold")
    plt.title("Tremor Detection (Binary Labels)")
    plt.xlabel("Time Index")
    plt.ylabel("Tremor Presence (1 = Yes, 0 = No)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[11]:

def rectangular_pulse_vis(pulse_signal):
    """
    Visualize the rectangular pulse signal.
    
    Parameters:
        pulse_signal (array): Rectangular pulse signal.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pulse_signal, label="Tremor Rectangular Pulse", color="blue", drawstyle="steps-post")
    plt.title("Rectangular Pulse Representation of Tremor Events")
    plt.xlabel("Time Index")
    plt.ylabel("Tremor Presence (1 = Yes, 0 = No)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[12]:

def tremor_edge_vis(tremor_onsets, time_index, rectangular_pulse):
    """
    Visualize the tremor onsets and offsets.
    
    Parameters:
        tremor_onsets (array): Array of tremor onsets and offsets.
    """
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


# In[13]:
