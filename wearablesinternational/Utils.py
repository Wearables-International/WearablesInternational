import pytz
import math
import numpy as np
from datetime import datetime
from scipy.signal import butter, filtfilt

# Helper functions

# convert ISO/Z time to a local timezone
def convert_to_local_timezone(timestamps, local_timezone):
    """Convert ISO/Z time to a local timezone

    Args:
        timestamps (_type_): UNIX timestamp
        local_timezone (_type_): A local date/time

    Returns:
        _type_: _description_
    """
    local_times = []
    for ts in timestamps:
        # Parse the ISO timestamp and assign UTC timezone
        utc_time = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        # Convert to local timezone
        local_time = utc_time.astimezone(pytz.timezone(local_timezone))
        local_times.append(local_time.isoformat())
    return local_times


# Calculate g from X,Y,Z acc 
def acc_to_g(x_data, y_data, z_data):
    """Converts X,Y,Z from accelerometer to G

    Args:
        x_data (_type_): X signal
        y_data (_type_): Y signal
        z_data (_type_): X signal

    Returns:
        _type_: _description_
    """
    g_magnitude = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(x_data, y_data, z_data)]
    return g_magnitude


def scl_to_eda(scl_signal, sampling_rate):
    """
    Converts a Skin Conductance Level (SCL) signal to Electrodermal Activity (EDA).
    
    Parameters:
        scl_signal (array-like): The raw SCL signal (in microsiemens).
        sampling_rate (float): Sampling rate of the SCL signal in Hz.
    
    Returns:
        eda_signal (array-like): The phasic component (EDA) of the SCL signal.
    """
    scl_signal = np.array(scl_signal, dtype=float)
    non_zero_mean = np.mean(scl_signal[scl_signal != 0])
    scl_signal[scl_signal == 0] = non_zero_mean
    low_cutoff = 0.05
    high_cutoff = 0.5
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')
    eda_signal = filtfilt(b, a, scl_signal)
    return eda_signal


def rri_to_rr(rri):
    """
    Converts Respiratory Rate Interval (RRI) to Respiratory Rate (RR).

    Parameters:
        rri (float or array-like): Respiratory Rate Interval in seconds (time between breaths).

    Returns:
        float or array-like: Respiratory Rate in breaths per minute (bpm).
    """
    rri = np.array(rri, dtype=float)
    non_zero_mean = np.mean(rri[rri != 0])
    rri[rri == 0] = non_zero_mean
    rr = 60 / rri
    return rr

# TODO: below not working for embrace plus
def sampling_frequency_df(df):
    df = df.sort_values(by='timestamp_unix')
    time_diffs = df['timestamp_unix'].diff().dropna()
    average_time_diff = time_diffs.mean()
    if average_time_diff > 1000:
        sampling_frequency = average_time_diff / 1000
        return int(np.round(sampling_frequency))
    else:
        return int(np.round(average_time_diff))   