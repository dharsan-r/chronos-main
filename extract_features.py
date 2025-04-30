# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:40:36 2024

@author: fakbarifar
modified by : Dharsan
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

def calculate_posture_speed(speed_data, original_length, window_size=5, std_threshold=0.005, plot=False):
    """
    Calculate the mean posture speed over the steady phase of the resampled movement data,
    where the steady phase is defined as the first 200 ms in the original sampling length.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values (e.g., 64 samples).
    - original_length: The original length of the signal before resampling.
    - window_size: Number of samples to consider for calculating moving standard deviation (default: 5).
    - std_threshold: Standard deviation threshold below which the hand is considered steady (default: 0.005).
    - plot: Boolean indicating whether to plot the detected steady phase on the speed data (default: False).

    Returns:
    - mean_posture_speed: The average speed during the entire detected steady phase.
    - steady_indices: The indices of all samples in the steady phase.
    """
    
    steady_phase_samples = int((200 / original_length) * len(speed_data))
    
    steady_phase_data = speed_data[:steady_phase_samples]
    
    mean_posture_speed = np.mean(steady_phase_data)
    steady_indices = list(range(steady_phase_samples))  # indices for the steady phase


    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(speed_data, label='Speed Data', color='blue')
        plt.scatter(steady_indices, steady_phase_data, color='red', label='Detected Steady Phase')
        plt.axhline(y=mean_posture_speed, color='green', linestyle='--', label='Mean Posture Speed')
        plt.xlabel("Sample Index")
        plt.ylabel("Speed")
        plt.title("Detected Steady Phase in Speed Data")
        plt.legend()
        plt.show()

    return mean_posture_speed, steady_indices

# Example usage - The ultimate "posture speed" is determined as the median of this feature across all trials.
# posture_speed, _ = calculate_posture_speed(trial, original_length, window_size=1, std_threshold=0.005, plot=False)



def calculate_target_on_index(original_length, signal_length, target_on_offset=200):
    """
    Calculate the sample index corresponding to target_on based on the original trial length.

    Parameters:
    - original_length: The total duration of the original trial in milliseconds.
    - signal_length: The length of the resampled signal in samples.
    - target_on_offset: The time offset in milliseconds after the start when target_on occurs (default: 200 ms).

    Returns:
    - start_index: The sample index of target_on within the resampled signal.
    """
    
    time_per_sample = original_length / signal_length
    start_index = int(target_on_offset / time_per_sample)
    return start_index

def calculate_reaction_time_slope(speed_data, slope_threshold=0.01, original_length=None, target_on_offset=200):
    """
    Detect movement onset based on the slope of speed data, starting only after target_on.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - slope_threshold: Threshold for the rate of change in speed (default: 0.01).
    - original_length: The total duration of the original trial in milliseconds (default: None).
    - target_on_offset: Time after the start when target_on occurs (default: 200 ms).

    Returns:
    - onset_index / len(speed_data): Normalized index of movement onset.
    """
    if original_length is None:
        raise ValueError("original_length must be provided for accurate target_on calculation.")

    
    start_index = calculate_target_on_index(original_length, len(speed_data), target_on_offset)
    
    
    slopes = np.diff(speed_data)

    # Find the first instance where slope exceeds the threshold, starting from start_index
    if np.argmax(slopes[start_index:] > slope_threshold)==0:
        onset_index = start_index + np.argmax(slopes[start_index:] > slope_threshold) +1
    else: 
        onset_index = start_index + np.argmax(slopes[start_index:] > slope_threshold)
    
    return onset_index / len(speed_data)

# Function to calculate reaction time with original lengths
def calculate_physical_reaction_time(trial, detection_method, original_length, plot=False):
    """
    Calculate reaction time in physical units (milliseconds) and optionally plot the speed data
    with target_on and movement onset markers.

    Parameters:
    - trial: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect reaction time (e.g., calculate_reaction_time_slope).
    - original_length: The original length of the trial in milliseconds.
    - plot: Boolean indicating whether to plot the detected movement onset (default: False).

    Returns:
    - reaction_time_physical: Reaction time in milliseconds from target_on to movement onset.
    """
    
    reaction_time_resampled = detection_method(trial, original_length=original_length)
    reaction_time_index = int(reaction_time_resampled * len(trial))  # Index in 64-sample data

    # Convert reaction time index back to original length in milliseconds
    reaction_time_physical = (reaction_time_index / len(trial)) * original_length - 200  # Subtract 200 ms for target_on

    
    if plot:
        time_per_sample = original_length / len(trial)  # Convert samples to milliseconds
        time_axis = np.arange(len(trial)) * time_per_sample  # Create time axis in milliseconds

        
        target_on_time = 200
        movement_onset_time = reaction_time_index * time_per_sample

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, trial, label='Speed Data', color='blue')
        plt.axvline(x=target_on_time, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=movement_onset_time, color='red', linestyle='--', label='Movement Onset')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title("Speed Data with Target On and Movement Onset")
        plt.legend()
        plt.show()
        
    return reaction_time_physical

# Example usage - The ultimate "Reaction Time" is determined as the median of this feature across all trials.
# reaction_time_slope = calculate_physical_reaction_time(trial, calculate_reaction_time_slope, original_length, plot=False)


def calculate_significant_speed_peaks_dynamic_prominence(speed_data, detection_method, original_length, slope_threshold=0.01, plot=False):
    """
    Detect significant speed peaks (prominent peaks) between the movement onset and end of the signal,
    with prominence set as a fraction of the maximum peak in the trial.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset (e.g., calculate_reaction_time_slope).
    - original_length: The total duration of the original trial in milliseconds.
    - slope_threshold: Threshold for the rate of change in speed (default: 0.01).
    - plot: Boolean indicating whether to plot the detected peaks and movement onset (default: False).

    Returns:
    - significant_peaks_count: The count of significant peaks in the movement phase.
    - onset_index: The index of movement onset within the speed data.
    - offset_index: The index of movement offset (end of signal).
    - peaks: The indices of the detected significant peaks within the movement phase.
    """
    # Calculate dynamic prominence as a fraction of the maximum peak in the trial
    dynamic_prominence = np.max(speed_data) / 50  # Adjust the denominator as needed
    
    onset_index = int(detection_method(speed_data, slope_threshold=slope_threshold, original_length=original_length) * len(speed_data))
    
    offset_index = len(speed_data) - 1

    movement_phase_data = speed_data[onset_index:offset_index]

    peaks, _ = find_peaks(movement_phase_data, prominence=dynamic_prominence)
    significant_peaks_count = len(peaks)


    if plot:
        time_per_sample = original_length / len(speed_data)  
        time_axis = np.arange(len(speed_data)) * time_per_sample  
        movement_phase_time = time_axis[onset_index:offset_index]  

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, speed_data, label='Speed Data', color='blue')
        plt.axvline(x=200, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=onset_index * time_per_sample, color='green', linestyle='--', label='Movement Onset')
        plt.axvline(x=offset_index * time_per_sample, color='red', linestyle='--', label='End of Signal (Offset)')
        plt.plot(movement_phase_time[peaks], movement_phase_data[peaks], 'ro', label='Significant Peaks')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title(f"Speed Data with Movement Onset, Offset, and Significant Peaks (Count: {significant_peaks_count})")
        plt.legend()
        plt.show()


    return significant_peaks_count, onset_index, offset_index, peaks + onset_index

# Example usage - The ultimate "Speed Maxima Count" is determined as the mean of this feature across all trials.
# maxima_count_slope, _, _, _ = calculate_significant_speed_peaks_dynamic_prominence(
#     trial, calculate_reaction_time_slope, original_length, slope_threshold=0.01, plot=False
# )

def calculate_min_max_speed_difference_dynamic_extrema(speed_data, detection_method, original_length, slope_threshold=0.01, prominence_fraction=(1/14), plot=False):
    """
    Calculate the mean difference between adjacent prominent local minima and maxima within the movement phase,
    with dynamic prominence and optional plotting.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset (e.g., calculate_reaction_time_slope).
    - original_length: The total duration of the original trial in milliseconds.
    - slope_threshold: Threshold for the rate of change in speed for movement onset detection (default: 0.01).
    - prominence_fraction: Fraction of the maximum peak value to use for prominence in extrema detection (default: 0.1).
    - plot: Boolean indicating whether to plot the detected movement onset, offset, and extrema (default: False).

    Returns:
    - mean_min_max_diff: The mean difference between adjacent prominent local minima and maxima.
    - onset_index: The index of movement onset.
    - offset_index: The index of movement offset (end of signal).
    - extrema_indices: Indices of the detected local minima and maxima.
    """
    # Calculate dynamic prominence based on the signal’s maximum value
    dynamic_prominence = np.max(speed_data) * prominence_fraction

    onset_index = int(detection_method(speed_data, slope_threshold=slope_threshold, original_length=original_length) * len(speed_data))
    
    offset_index = len(speed_data) - 1

    movement_phase_data = speed_data[onset_index:offset_index]

    maxima, _ = find_peaks(movement_phase_data, prominence=dynamic_prominence)
    minima, _ = find_peaks(-movement_phase_data, prominence=dynamic_prominence)  # Invert to find minima

    extrema_indices = np.sort(np.concatenate([maxima, minima]))

    min_max_diffs = np.abs(np.diff(movement_phase_data[extrema_indices]))

    mean_min_max_diff = np.mean(min_max_diffs) if len(min_max_diffs) > 0 else 0.0

    if plot:
        time_per_sample = original_length / len(speed_data)  
        time_axis = np.arange(len(speed_data)) * time_per_sample  
        movement_phase_time = time_axis[onset_index:offset_index]  

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, speed_data, label='Speed Data', color='blue')
        plt.axvline(x=200, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=onset_index * time_per_sample, color='green', linestyle='--', label='Movement Onset')
        plt.axvline(x=offset_index * time_per_sample, color='red', linestyle='--', label='End of Signal (Offset)')
        plt.plot(movement_phase_time[extrema_indices], movement_phase_data[extrema_indices], 'ro', label='Prominent Extrema')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title(f"Speed Data with Movement Onset, Offset, and Prominent Extrema (Mean Difference: {mean_min_max_diff:.3f})")
        plt.legend()
        plt.show()

    return mean_min_max_diff, onset_index, offset_index, extrema_indices

# Example usage - The ultimate "Min-Max Speed Difference" is determined as the mean of this feature across all trials.
# min_max_diff_slope, _, _, _ = calculate_min_max_speed_difference_dynamic_extrema(
#     trial, calculate_reaction_time_slope, original_length, slope_threshold=0.01, prominence_fraction=(1/5), plot=False
# )

def calculate_movement_time(speed_data, detection_method, original_length, slope_threshold=0.01, plot=False):
    """
    Calculate movement time as the duration from movement onset to movement offset,
    where offset is defined as the end of the signal.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset (e.g., calculate_reaction_time_slope).
    - original_length: The total duration of the original trial in milliseconds.
    - slope_threshold: Threshold for the rate of change in speed for movement onset detection (default: 0.01).
    - plot: Boolean indicating whether to plot the detected movement onset, offset, and movement time (default: False).

    Returns:
    - movement_time: The time in milliseconds from movement onset to movement offset.
    - onset_index: The index of movement onset.
    - offset_index: The index of movement offset (end of signal).
    """
    
    onset_index = int(detection_method(speed_data, slope_threshold=slope_threshold, original_length=original_length) * len(speed_data))
    
    offset_index = len(speed_data) - 1

    time_per_sample = original_length / len(speed_data)  # Time per sample in ms
    movement_time = (offset_index - onset_index) * time_per_sample


    if plot:
        time_axis = np.arange(len(speed_data)) * time_per_sample  

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, speed_data, label='Speed Data', color='blue')
        plt.axvline(x=200, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=onset_index * time_per_sample, color='green', linestyle='--', label='Movement Onset')
        plt.axvline(x=offset_index * time_per_sample, color='red', linestyle='--', label='End of Signal (Offset)')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title(f"Speed Data with Movement Onset, Offset, and Movement Time: {movement_time:.2f} ms")
        plt.legend()
        plt.show()

    return movement_time, onset_index, offset_index

# Example usage - The ultimate "Movement Time" is the median of this feature across all trials.
# movement_time_slope, _, _ = calculate_movement_time(
#     trial, calculate_reaction_time_slope, original_length, slope_threshold=0.01, plot=False
# )

def calculate_max_speed_between_onset_offset(speed_data, detection_method, original_length, slope_threshold=0.01, plot=False):
    """
    Calculate the maximum speed between movement onset and offset based on advanced detection methods,
    with optional plotting for visualization.

    Parameters:
    - speed_data: 1D numpy array containing the resampled speed values.
    - detection_method: Function to detect movement onset (e.g., calculate_reaction_time_slope).
    - original_length: The total duration of the original trial in milliseconds.
    - slope_threshold: Threshold for the rate of change in speed for movement onset detection (default: 0.01).
    - plot: Boolean indicating whether to plot the detected movement onset, offset, and maximum speed (default: False).

    Returns:
    - max_speed: The maximum speed within the detected movement phase.
    - onset_index: The index of movement onset.
    - offset_index: The index of movement offset (end of signal).
    """

    onset_index = int(detection_method(speed_data, slope_threshold=slope_threshold, original_length=original_length) * len(speed_data))
    
    offset_index = len(speed_data) - 1

    movement_phase_data = speed_data[onset_index:offset_index]


    max_speed = np.max(movement_phase_data) if len(movement_phase_data) > 0 else 0.0


    if plot:
        time_per_sample = original_length / len(speed_data)  
        time_axis = np.arange(len(speed_data)) * time_per_sample  
        movement_phase_time = time_axis[onset_index:offset_index]  

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, speed_data, label='Speed Data', color='blue')
        plt.axvline(x=200, color='purple', linestyle='--', label='Target On (200 ms)')
        plt.axvline(x=onset_index * time_per_sample, color='green', linestyle='--', label='Movement Onset')
        plt.axvline(x=offset_index * time_per_sample, color='red', linestyle='--', label='End of Signal (Offset)')
        plt.scatter(movement_phase_time[np.argmax(movement_phase_data)], max_speed, color='red', label='Max Speed')
        plt.xlabel("Time (ms)")
        plt.ylabel("Speed")
        plt.title(f"Speed Data with Movement Onset, Offset, and Maximum Speed: {max_speed:.3f}")
        plt.legend()
        plt.show()

    return max_speed, onset_index, offset_index

# Example usage - The ultimate "Max Speed" is the median of this feature across all trials
# max_speed_slope, _, _ = calculate_max_speed_between_onset_offset(
#     trial, calculate_reaction_time_slope, original_length, slope_threshold=0.01, plot=False
# )


df = pd.DataFrame(np.load("test_stroke_Vabs02.npy"))

context_arrays = []
groundtruth_arrays = []

# Loop through each column and split into the gt and the context array
for col_idx in range(df.shape[1]):
    col_array = df.iloc[:, col_idx].to_numpy()
    
    # Split the array into first 512 elements and last 64 elements
    first_512 = col_array[:512]
    last_64 = col_array[-64:]
    
    # Append to respective lists with the new names
    context_arrays.append(first_512)
    groundtruth_arrays.append(last_64)


calculate_posture_speed(cur_trial_context)

# for sub_idx in range(len(context_arrays)):
#     cur_trial_context = context_arrays[sub_idx]
#     cur_trial_gt = groundtruth_arrays[sub_idx]
    
#     calculate_posture_speed(cur_trial_context)