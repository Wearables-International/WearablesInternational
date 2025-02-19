import numpy as np
import pandas as pd
from math import floor
from functools import reduce
from datetime import datetime, timezone
from scipy.stats import zscore
from scipy.signal import butter, filtfilt
from wearablesinternational.Devices.Readers import Dataset
from wearablesinternational.Exceptions import PreprocessingException


def smooth_column(dataframe, column_name, window_size=5):
    """
    Smooth the numbers in the specified column of a DataFrame using a moving average.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the column to smooth.
        column_name (str): The name of the column to smooth.
        window_size (int): The size of the moving average window. Default is 5.

    Returns:
        pd.Series: A smoothed version of the column.
    """
    # Ensure the window size is odd for symmetry
    if window_size % 2 == 0:
        window_size += 1
    smoothed = dataframe[column_name].rolling(window=window_size, center=True).mean()
    smoothed = smoothed.bfill().ffill()
    smoothed = smoothed.round()
    result_df = dataframe.copy()
    result_df[f"{column_name}_smoothed"] = smoothed
    return result_df


def calculate_daily_pdfs(data, T=60):
    if not np.issubdtype(data['timestamp_iso'].dtype, np.datetime64):
        data['timestamp_iso'] = pd.to_datetime(data['timestamp_iso'])
    data['Date'] = data['timestamp_iso'].dt.date
    data['Hour'] = data['timestamp_iso'].dt.hour + data['timestamp_iso'].dt.minute / 60.0
    daily_data = data.groupby('Date').agg(list).reset_index()
    pdf_results = []
    for _, row in daily_data.iterrows():
        day = row['Date']
        hours = np.array(row['Hour'])
        g_values = np.array(row['ACC'])
        
        # Aggregate by T-minute intervals
        n_boxes = floor(len(g_values) / T)
        averaged_values = [
            g_values[j * T:(j + 1) * T].mean() for j in range(n_boxes)
        ]
        averaged_hours = [
            hours[j * T:(j + 1) * T].mean() for j in range(n_boxes)
        ]
        
        day_df = pd.DataFrame({
            'Hour': averaged_hours,
            'ACC': averaged_values
        })
        day_df['Date'] = day
        pdf_results.append(day_df)
    
    combined_df = pd.concat(pdf_results, ignore_index=True)
    return combined_df


def impute_with_median(df, column_name):
    """
    Imputes 0 and NaN values in the specified column of a DataFrame with the column's mean.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to impute.
        column_name (str): The name of the column to impute.

    Returns:
        pd.DataFrame: The DataFrame with the imputed column.
    """
    valid_values = df[column_name][(df[column_name] != 0) & (~df[column_name].isna())]
    median_value = valid_values.median()
    df[column_name] = df[column_name].replace(0, np.nan)
    df[column_name] = df[column_name].fillna(median_value)
    return df


def impute_with_mean(df, column_name):
    """
    Imputes 0 and NaN values in the specified column of a DataFrame with the column's mean.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to impute.
        column_name (str): The name of the column to impute.

    Returns:
        pd.DataFrame: The DataFrame with the imputed column.
    """
    valid_values = df[column_name][(df[column_name] != 0) & (~df[column_name].isna())]
    mean_value = valid_values.mean()
    df[column_name] = df[column_name].replace(0, np.nan)
    df[column_name] = df[column_name].fillna(mean_value)
    return df

# warning: imputes missing values
def scale_to_range(numbers, min_val=0, max_val=100):
    numbers = np.array(numbers, dtype=float)
    nan_mask = np.isnan(numbers)
    if np.any(nan_mask):
        mean_value = np.nanmean(numbers)
        numbers[nan_mask] = mean_value
    min_num = np.min(numbers)
    max_num = np.max(numbers)
    if min_num == max_num:
        return np.full_like(numbers, min_val, dtype=float)
    scaled_numbers = min_val + (numbers - min_num) * (max_val - min_val) / (max_num - min_num)
    return scaled_numbers


def lowpass_filter(array, highcut, fs, order):
    nyquist = 0.5 * fs
    high = highcut / nyquist
    b, a = butter(order, high, btype='low')
    y = filtfilt(b, a, array)
    return y


def bandpass_filter(array, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, array)
    return y


def moving_average(array, window_size):
    return np.convolve(array, np.ones(window_size) / window_size, mode='valid')


# Apply Z-score normalization
def zscore_data(array):
    return zscore(array)


def normalize_data(array):
    # Normalize signal to [0, 1]
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def compute_hjorth_parameters(array):
    first_derivative = np.diff(array)
    second_derivative = np.diff(first_derivative)
    var_zero = np.var(array)
    var_d1 = np.var(first_derivative)
    var_d2 = np.var(second_derivative)
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return mobility, complexity


def upsample_list_to(data: list, target_length: int):
    """Upsamples a List of values to match target lengtj

    Args:
        data (list): List of numeric values to upsample
        target_length (int): Target length required

    Returns:
        _type_: List of upsampled values of length target_length
    """
    if not data:
        raise PreprocessingException(203, "Preprocessing.upsample_to: The input list is empty.")
    original_length = len(data)
    if original_length >= target_length:
        raise PreprocessingException(202, "Preprocessing.upsample_to: Target length must exceed the original data length for upsampling.")
    repeat_factor = target_length // original_length
    remainder = target_length % original_length
    upsampled_list = data * repeat_factor + data[:remainder]
    return upsampled_list


def upsample_to(ds: Dataset, target_length: int):
    """Upsamples a Dataset of values to match target lengtj

    Args:
        data (list): Dataset of numeric values to upsample
        target_length (int): Target length required

    Returns:
        _type_: Dataset upsampled to length target_length
    """
    if ds.dataframe.empty:
        raise PreprocessingException(203, "Preprocessing.upsample_to: The input DataFrame is empty.")
    original_length = len(ds.dataframe)
    if original_length >= target_length:
        raise PreprocessingException(202, "Preprocessing.upsample_to: Target length must exceed the original data length for upsampling.")
    repeat_factor = target_length // original_length
    remainder = target_length % original_length
    upsampled_df = pd.concat([ds.dataframe] * repeat_factor, ignore_index=True)
    if remainder > 0:
        extra_rows = ds.dataframe.iloc[:remainder]
        upsampled_df = pd.concat([upsampled_df, extra_rows], ignore_index=True)
    upsampled_df.reset_index(drop=True)
    ds.dataframe = upsampled_df
    return ds


def downsample_column(df, column_name, target_rows):
    # Calculate the number of rows per bin
    original_rows = len(df)
    rows_per_bin = original_rows / target_rows

    # Create bins and calculate the mean for each bin
    df['bin'] = np.floor(np.arange(original_rows) / rows_per_bin).astype(int)
    downsampled = df.groupby('bin')[column_name].mean().reset_index(drop=True)

    # Return the downsampled column as a new dataframe
    return downsampled


def downsample_to(ds: Dataset, target_length: int):
    """Downsample a Dataset to match target length

    Args:
        data (list): Dataset of numeric values to downsample
        target_length (int): Target length required

    Returns:
        _type_: Dataset downsampled to length target_length
    """
    if ds.dataframe.empty:
        raise PreprocessingException(203, "Preprocessing.downsample_to: The input DataFrame is empty.")
    original_length = len(ds.dataframe)
    downsampling_factor = original_length / target_length
    if downsampling_factor < 1:
        raise PreprocessingException(202, "Preprocessing.downsample_to: Target length exceeds the original data length. Upsampling is not supported.")
    indices = (ds.dataframe.index / downsampling_factor).round().astype(int)
    downsampled_indices = pd.unique(indices[:target_length])  # Ensure unique indices
    if len(downsampled_indices) != target_length:
        raise PreprocessingException(202, f"Preprocessing.downsample_to: Could not achieve exactly {target_length} rows due to rounding.")
    downsampled_df = ds.dataframe.iloc[downsampled_indices]
    downsampled_df.reset_index(drop=True)
    ds.dataframe = downsampled_df
    return ds


def downsample_freq(ds: Dataset, factor: int):
    """Downsample a Dataset tby a factor

    Args:
        ds (Dataset): Dataset of numeric values to downsample
        factor (int): Factor to downsample by

    Returns:
        _type_: Downsampled Dataset
    """
    if ds.dataframe.empty:
        raise PreprocessingException(203, "Preprocessing.downsample_freq: The input DataFrame is empty.")
    if factor <= 0:
        raise PreprocessingException(203, "Preprocessing.downsample_freq: The factor value must be a positive integer.")
    downsampled_df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in ds.dataframe.dtypes.items()})
    for start in range(0, len(ds.dataframe), factor):
        subset = ds.dataframe.iloc[start:start + factor]
        mean_values = subset.mean()
        downsampled_df = pd.concat([downsampled_df, pd.DataFrame([mean_values])], ignore_index=True)
    ds.dataframe = downsampled_df
    return ds


def merge_datasets_time(datasets: list):
    """Merge multiple dataframes on timestamp

    Args:
        datasets (list): A List of Dataset objects

    Returns:
        _type_: A merged DataFrame
    """
    shortest = None
    dfs = []
    for ds in datasets:
        if not isinstance(ds, Dataset):
            raise PreprocessingException(201, "Preprocessing.merge_datasets: Input must be of type Dataset.")
        else:
            if ds.dataframe.empty:
                raise PreprocessingException(203, "Preprocessing.merge_datasets: Input DataFrame is empty.")
            else:
                if shortest is None:
                    shortest = ds.dataframe.shape[0]
                else:
                    if ds.dataframe.shape[0] <- shortest:
                        shortest = ds.dataframe.shape[0]
                temp_df = ds.dataframe
                temp_df.drop("timestamp_iso", axis=1, inplace=True)
                dfs.append(temp_df)
    dfs_trim = []
    for df in dfs:
        dfs_trim.append(df.iloc[:shortest])

    all_columns = set(col for df in dfs_trim for col in df.columns)
    common_columns = set.intersection(*(set(df.columns) for df in dfs_trim)) - {"timestamp_unix"}
    non_common_columns = all_columns - common_columns - {"timestamp_unix"}
    result_df = reduce(lambda left, right: pd.merge(left, right, on="timestamp_unix", how="outer"), dfs_trim)
    result_df = result_df[["timestamp_unix"] + [col for col in non_common_columns if col in result_df.columns]]
    result_df["timestamp_iso"] = result_df["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    result_df.to_csv("merged.csv")
    return result_df


def merge_datasets(datasets: list):
    """Merge multiple dataframes after downsampling

    Args:
        datasets (list): A List of Dataset objects

    Returns:
        _type_: A merged DataFrame
    """
    shortest = None
    dfs = []
    for ds in datasets:
        if not isinstance(ds, Dataset):
            raise PreprocessingException(201, "Preprocessing.merge_datasets: Input must be of type Dataset.")
        else:
            if ds.dataframe.empty:
                raise PreprocessingException(203, "Preprocessing.merge_datasets: Input DataFrame is empty.")
            else:
                if shortest is None:
                    shortest = ds.dataframe.shape[0]
                else:
                    if ds.dataframe.shape[0] <- shortest:
                        shortest = ds.dataframe.shape[0]
                temp_df = ds.dataframe
                temp_df.drop("timestamp_unix", axis=1, inplace=True)
                dfs.append(temp_df)
    dfs_trim = []
    for df in dfs:
        dfs_trim.append(df.iloc[:shortest])
    return pd.concat(dfs_trim, axis=1)

