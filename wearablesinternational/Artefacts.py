import numpy as np


def flat_check(segment_data, col_name, flat_variance_threshold=1e-4, zcr_threshold=0.05, mad_threshold=0.01, window_size=32, flat_percentage_threshold=20):
    magnitude = segment_data[col_name]
    rolling_variance = magnitude.rolling(window=window_size).var()
    rolling_mad = magnitude.rolling(window=window_size).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    zero_crossings = magnitude.diff().apply(np.sign).diff().fillna(0) != 0
    rolling_zcr = zero_crossings.rolling(window=window_size).mean()

    # Flat region criterion: low variance, low ZCR, and low MAD
    is_flat_region = (rolling_variance < flat_variance_threshold) & (rolling_mad < mad_threshold) & (rolling_zcr < zcr_threshold)

    # Calculate flat region percentage and check against threshold
    flat_region_percentage = is_flat_region.mean() * 100
    is_flat = flat_region_percentage > flat_percentage_threshold

    if is_flat:
        print(f"Warning: Flat region percentage is {flat_region_percentage:.2f}% - exceeds threshold of {flat_percentage_threshold}%")
    return flat_region_percentage, is_flat


def remove_outliers(df, column_name):
    """
    Removes outliers from the specified column of a DataFrame using the IQR method.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column from which to remove outliers.

    Returns:
        pd.DataFrame: A DataFrame with outliers removed from the specified column.
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return filtered_df


def filter_outliers_iqr(dataframe, temp_col, multiplier=1.5):
    """
    Filter out temperature values based on IQR outlier detection.
    
    Args:
    dataframe (pd.DataFrame): The DataFrame containing the temperature data.
    temp_col (str): The name of the temperature column in the DataFrame.
    multiplier (float): The IQR multiplier to define the range of outliers (default is 1.5).
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = dataframe[temp_col].quantile(0.25)
    Q3 = dataframe[temp_col].quantile(0.75)
    
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for detecting outliers
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Filter out data points outside the lower and upper bounds
    filtered_data = dataframe[(dataframe[temp_col] >= lower_bound) & (dataframe[temp_col] <= upper_bound)].copy()
    
    return filtered_data


def reclassify_artifacts(artifact_peaks, normal_peaks, peaks, peak_amplitudes):
    # Convert lists to arrays for easier indexing
    peaks = np.array(peaks)
    peak_amplitudes = np.array(peak_amplitudes)
    
    # Initialize new lists for updated classifications
    updated_normal_peaks = list(normal_peaks)  # Start with all initially classified normal peaks
    updated_artifact_peaks = []
    
    # Process each artifact peak
    for artifact_idx in artifact_peaks:
        # Find the index of the artifact peak in the original 'peaks' array
        artifact_peak_amplitude = peak_amplitudes[np.where(peaks == artifact_idx)[0][0]]
        
        # Collect amplitudes of the last four normal peaks before this artifact
        preceding_normal_indices = [i for i in normal_peaks if i < artifact_idx]
        
        if len(preceding_normal_indices) >= 4:
            # Only consider the last four
            last_four_normals = preceding_normal_indices[-4:]
            last_four_amplitudes = peak_amplitudes[[np.where(peaks == idx)[0][0] for idx in last_four_normals]]
            
            # Calculate the average amplitude of these four normal peaks
            average_amplitude = np.mean(last_four_amplitudes)
            
            # Check if the artifact peak is within 20% of this average
            #lower_bound = average_amplitude * 0.8
            upper_bound = average_amplitude * 1.5
            
            if artifact_peak_amplitude <= upper_bound:
                # Reclassify as normal
                updated_normal_peaks.append(artifact_idx)
            else:
                # Keep as artifact
                updated_artifact_peaks.append(artifact_idx)
        else:
            # Not enough normal peaks before this artifact, keep as artifact
            updated_artifact_peaks.append(artifact_idx)

    return updated_normal_peaks, updated_artifact_peaks