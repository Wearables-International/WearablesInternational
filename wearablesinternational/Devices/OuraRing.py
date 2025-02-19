import os
import pandas as pd
from wearablesinternational.Exceptions import ReaderException
from wearablesinternational.Devices.Readers import Reader, Dataset


class ReaderOuraRing(Reader):
    def read(self, source):
        if isinstance(source, str):
            if os.path.exists(source):
                files = os.listdir(source)
                datasets = {}
                for file in files:
                    oura_data = pd.read_csv(os.path.join(source, file))
                    oura_data["date"] = pd.to_datetime(oura_data["date"])
                    oura_data["timestamp_unix"] = oura_data["date"].apply(lambda x: int(x.timestamp()))
                    oura_data["timestamp_iso"] = oura_data["date"].apply(lambda x: x.isoformat() + 'Z' if pd.notnull(x) else None)

                    # STEPS
                    temp = oura_data[["date", "timestamp_unix", "timestamp_iso", "Steps"]]
                    temp.columns=["date", "timestamp_unix","timestamp_iso", "STEPS"]
                    temp = temp[["timestamp_unix", "timestamp_iso", "STEPS"]]
                    datasets["STEPS"] = Dataset("STEPS", temp.iloc[0,0], 0.0, temp)
                    # HR
                    temp = oura_data[["date", "timestamp_unix", "timestamp_iso", "Average Resting Heart Rate"]]
                    temp.columns=["date", "timestamp_unix","timestamp_iso", "HR"]
                    temp = temp[["timestamp_unix", "timestamp_iso", "HR"]]
                    datasets["HR"] = Dataset("HR", temp.iloc[0,0], 0.0, temp)
                    # HRV
                    temp = oura_data[["date", "timestamp_unix", "timestamp_iso", "Average HRV"]]
                    temp.columns=["date", "timestamp_unix","timestamp_iso", "HRV"]
                    temp = temp[["timestamp_unix", "timestamp_iso", "HRV"]]
                    datasets["HRV"] = Dataset("HRV", temp.iloc[0,0], 0.0, temp)
                    # RR
                    temp = oura_data[["date", "timestamp_unix", "timestamp_iso", "Respiratory Rate"]]
                    temp.columns=["date", "timestamp_unix","timestamp_iso", "RR"]
                    temp = temp[["timestamp_unix", "timestamp_iso", "RR"]]
                    datasets["RR"] = Dataset("RR", temp.iloc[0,0], 0.0, temp)
                    # ACTIVITY
                    temp = oura_data[["date", "timestamp_unix", "timestamp_iso", "Activity Score"]]
                    temp.columns=["date", "timestamp_unix","timestamp_iso", "ACTIVITY"]
                    temp = temp[["timestamp_unix", "timestamp_iso", "ACTIVITY"]]
                    datasets["ACTIVITY"] = Dataset("ACTIVITY", temp.iloc[0,0], 0.0, temp)
                    # TOTALSLEEP
                    temp = oura_data[["date", "timestamp_unix", "timestamp_iso", "Total Sleep Score"]]
                    temp.columns=["date", "timestamp_unix","timestamp_iso", "TOTALSLEEP"]
                    temp = temp[["timestamp_unix", "timestamp_iso", "TOTALSLEEP"]]
                    datasets["TOTALSLEEP"] = Dataset("TOTALSLEEP", temp.iloc[0,0], 0.0, temp)
                    # REMSLEEP
                    temp = oura_data[["date", "timestamp_unix", "timestamp_iso", "REM Sleep Score"]]
                    temp.columns=["date", "timestamp_unix","timestamp_iso", "REMSLEEP"]
                    temp = temp[["timestamp_unix", "timestamp_iso", "REMSLEEP"]]
                    datasets["REMSLEEP"] = Dataset("REMSLEEP", temp.iloc[0,0], 0.0, temp)
                    # DEEPSLEEP
                    temp = oura_data[["date", "timestamp_unix", "timestamp_iso", "Deep Sleep Score"]]
                    temp.columns=["date", "timestamp_unix","timestamp_iso", "DEEPSLEEP"]
                    temp = temp[["timestamp_unix", "timestamp_iso", "DEEPSLEEP"]]
                    datasets["DEEPSLEEP"] = Dataset("DEEPSLEEP", temp.iloc[0,0], 0.0, temp)
                    # EFFICIENTSLEEP
                    temp = oura_data[["date", "timestamp_unix", "timestamp_iso", "Sleep Efficiency Score"]]
                    temp.columns=["date", "timestamp_unix","timestamp_iso", "EFFICIENTSLEEP"]
                    temp = temp[["timestamp_unix", "timestamp_iso", "EFFICIENTSLEEP"]]
                    datasets["EFFICIENTSLEEP"] = Dataset("EFFICIENTSLEEP", temp.iloc[0,0], 0.0, temp)
                    
                return datasets
            else:
                raise ReaderException(101, "ReaderOuraRing.Read: source folder does not exists.")
        else:
            raise ReaderException(105, "ReaderOuraRing.Read: Invalid source type.")