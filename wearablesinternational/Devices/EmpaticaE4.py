import os
import pandas as pd
from datetime import datetime, timezone
from wearablesinternational.Exceptions import ReaderException
from wearablesinternational.Devices.Readers import Reader, Dataset
from wearablesinternational.Utils import acc_to_g

class ReaderE4(Reader):
    def read(self, source):
        if isinstance(source, str):
            if os.path.exists(source):
                files = os.listdir(source)
                datasets = {}
                for file in files:
                    if "ACC" in file.upper():
                        temp = pd.read_csv(os.path.join(source, file), header=None, engine="pyarrow")
                        start_time = temp.values[0][0]
                        sampling_rate = temp.values[1][0]
                        temp["timestamp_unix"] = [start_time + i * (1 / sampling_rate) for i in range(temp.shape[0])]
                        temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                        temp.columns=["X","Y","Z", "timestamp_unix","timestamp_iso"]
                        temp["ACC"] = acc_to_g(temp["X"].tolist(), temp["Y"].tolist(), temp["Z"].tolist())
                        temp = temp[["timestamp_unix", "timestamp_iso", "ACC"]]
                        datasets["ACC"] = Dataset("ACC", start_time, sampling_rate, temp.iloc[2:])
                    if "HR" in file.upper():
                        temp = pd.read_csv(os.path.join(source, file), header=None, engine="pyarrow")
                        start_time = temp.values[0][0]
                        sampling_rate = temp.values[1][0]
                        temp["timestamp_unix"] = [start_time + i * (1 / sampling_rate) for i in range(temp.shape[0])]
                        temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                        temp.columns=["HR", "timestamp_unix","timestamp_iso"]
                        temp = temp[["timestamp_unix", "timestamp_iso", "HR"]]
                        datasets["HR"] = Dataset("HR", start_time, sampling_rate, temp.iloc[2:])
                    if "IBI" in file.upper():
                        temp = pd.read_csv(os.path.join(source, file), header=None, engine="pyarrow")
                        start_time = temp.values[0][0]
                        temp["timestamp_unix"] = start_time
                        for i in range(1, len(temp)):
                            temp.at[i, "timestamp_unix"] = start_time + temp.at[i, 0]
                        temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                        temp.drop(temp.columns[0], axis=1, inplace=True)
                        temp.columns=["IBI", "timestamp_unix","timestamp_iso"]
                        temp = temp[["timestamp_unix", "timestamp_iso", "IBI"]]
                        temp['IBI'] = pd.to_numeric(temp['IBI'], errors='coerce')
                        datasets["IBI"] = Dataset("IBI", start_time, 0.0, temp.iloc[2:])
                    if "TEMP" in file.upper():
                        temp = pd.read_csv(os.path.join(source, file), header=None, engine="pyarrow")
                        start_time = temp.values[0][0]
                        sampling_rate = temp.values[1][0]
                        temp["timestamp_unix"] = [start_time + i * (1 / sampling_rate) for i in range(temp.shape[0])]
                        temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                        temp.columns=["TEMP", "timestamp_unix","timestamp_iso"]
                        temp = temp[["timestamp_unix", "timestamp_iso", "TEMP"]]
                        datasets["TEMP"] = Dataset("TEMP", start_time, sampling_rate, temp.iloc[2:])
                    if "BVP" in file.upper():
                        temp = pd.read_csv(os.path.join(source, file), header=None, engine="pyarrow")
                        start_time = temp.values[0][0]
                        sampling_rate = temp.values[1][0]
                        temp["timestamp_unix"] = [start_time + i * (1 / sampling_rate) for i in range(temp.shape[0])]
                        temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                        temp.columns=["BVP", "timestamp_unix","timestamp_iso"]
                        temp = temp[["timestamp_unix", "timestamp_iso", "BVP"]]
                        datasets["BVP"] = Dataset("BVP", start_time, sampling_rate, temp.iloc[2:])
                    if "EDA" in file.upper():
                        temp = pd.read_csv(os.path.join(source, file), header=None, engine="pyarrow")
                        start_time = temp.values[0][0]
                        sampling_rate = temp.values[1][0]
                        temp["timestamp_unix"] = [start_time + i * (1 / sampling_rate) for i in range(temp.shape[0])]
                        temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                        temp.columns=["EDA", "timestamp_unix", "timestamp_iso"]
                        temp = temp[["timestamp_unix", "timestamp_iso", "EDA"]]
                        datasets["EDA"] = Dataset("EDA", start_time, sampling_rate, temp.iloc[2:])
                return datasets
            else:
                raise ReaderException(101, "ReaderE4.Read: source folder does not exists.")
        else:
            raise ReaderException(105, "ReaderE4.Read: Invalid source type.")