import os
import boto3
import pandas as pd
from io import StringIO
from fastavro import reader
from datetime import datetime, timezone
from wearablesinternational.Exceptions import ReaderException
from wearablesinternational.Devices.Readers import Reader, Dataset
from wearablesinternational.Sources import AWSEmbracePlus # not currently used
from wearablesinternational.Utils import acc_to_g, sampling_frequency_df


class ReaderEmbracePlus(Reader):
    def __make_timestamp_column(self, start_time,sampling_freq,len_list):
        start_time_ns = start_time * 1000
        start_timestamp = pd.to_datetime(start_time_ns, unit="ns")
        end_timestamp = start_timestamp + pd.to_timedelta(len_list / sampling_freq, unit="s")
        timestamp_column = pd.date_range(start=start_timestamp, end=end_timestamp, freq=pd.to_timedelta(1 / sampling_freq, unit="s"))
        timestamp_df = pd.DataFrame({"timestamp": timestamp_column})
        timestamp_df["unix_timestamp"] = timestamp_df["timestamp"].astype("int64") // 10**9
        return timestamp_df["unix_timestamp"].values[0:len_list]


    def read(self, source):
        if isinstance(source, str):
            if os.path.exists(source):
                files = os.listdir(source)
                emb_sysp_data = None
                emb_bvp_data = None
                emb_eda_data = None
                emb_acc_data = None
                emb_prv_data = None
                emb_temp_data = None
                emb_step_counts_data = None
                emb_respiratory_rate_data = None
                emb_activity_counts_data = None
                emb_pulse_rate_data = None
                datasets = {}
                for file in files:
                    temp = None
                    if os.path.splitext(file.upper())[1] == ".AVRO":
                        with open(os.path.join(source, file), "rb") as avro_file:
                            avro_reader = reader(avro_file)
                            for record in avro_reader:
                                # SYSP
                                systolic_peaks=record["rawData"]["systolicPeaks"]["peaksTimeNanos"]
                                if len(systolic_peaks) > 0:
                                    temp = pd.DataFrame({"systolic_peaks":systolic_peaks})
                                    temp["timestamp_unix"] = [ts / 1e9 for ts in systolic_peaks]
                                    temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                                    temp = temp[["timestamp_unix", "timestamp_iso", "systolic_peaks"]]
                                    temp['timestamp_iso'] = temp['timestamp_iso'].astype(str)
                                    if emb_sysp_data is None:
                                        emb_sysp_data = temp.copy()
                                    else:
                                        emb_sysp_data = pd.concat([emb_sysp_data, temp], axis=0)
                                # ACC
                                start_time=record["rawData"]["accelerometer"]["timestampStart"]
                                if start_time > 0:
                                    sampling_rate=record["rawData"]["accelerometer"]["samplingFrequency"]
                                    temp = pd.DataFrame({"X":record["rawData"]["accelerometer"]["x"], "Y":record["rawData"]["accelerometer"]["y"], "Z":record["rawData"]["accelerometer"]["z"]})
                                    temp["timestamp_unix"] = self.__make_timestamp_column(start_time, sampling_rate, temp.shape[0])
                                    temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                                    temp.columns=["X","Y","Z", "timestamp_unix","timestamp_iso"]
                                    temp["temperature_celsius"] = acc_to_g(temp["X"].tolist(), temp["Y"].tolist(), temp["Z"].tolist())
                                    temp = temp[["timestamp_unix", "timestamp_iso", "temperature_celsius"]]
                                    if emb_temp_data is None:
                                        emb_temp_data = temp.copy()
                                    else:
                                        emb_temp_data = pd.concat([emb_temp_data, temp], axis=0)
                                # EDA
                                start_time=record["rawData"]["eda"]["timestampStart"]
                                if start_time > 0:
                                    sampling_rate=record["rawData"]["eda"]["samplingFrequency"]
                                    temp = pd.DataFrame({"eda_scl_usiemens":record["rawData"]["eda"]["values"]})
                                    temp["timestamp_unix"] = self.__make_timestamp_column(start_time, sampling_rate, temp.shape[0])
                                    temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                                    temp.columns=["eda_scl_usiemens", "timestamp_unix","timestamp_iso"]
                                    temp = temp[["timestamp_unix", "timestamp_iso", "eda_scl_usiemens"]]
                                    if emb_eda_data is None:
                                        emb_eda_data = temp.copy()
                                    else:
                                        emb_eda_data = pd.concat([emb_eda_data, temp], axis=0)
                                # TEMP
                                start_time=record["rawData"]["temperature"]["timestampStart"]
                                if start_time > 0:
                                    sampling_rate=record["rawData"]["temperature"]["samplingFrequency"]
                                    temp = pd.DataFrame({"temperature_celsius":record["rawData"]["temperature"]["values"]})
                                    temp["timestamp_unix"] = self.__make_timestamp_column(start_time, sampling_rate, temp.shape[0])
                                    temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                                    temp.columns=["temperature_celsius", "timestamp_unix","timestamp_iso"]
                                    temp = temp[["timestamp_unix", "timestamp_iso", "temperature_celsius"]]
                                    if emb_temp_data is None:
                                        emb_temp_data = temp.copy()
                                    else:
                                        emb_temp_data = pd.concat([emb_temp_data, temp], axis=0)
                                # BVP
                                start_time=record["rawData"]["bvp"]["timestampStart"]
                                if start_time > 0:
                                    sampling_rate=record["rawData"]["bvp"]["samplingFrequency"]
                                    temp = pd.DataFrame({"BVP":record["rawData"]["bvp"]["values"]})
                                    temp["timestamp_unix"] = self.__make_timestamp_column(start_time, sampling_rate, temp.shape[0])
                                    temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                                    temp.columns=["BVP", "timestamp_unix","timestamp_iso"]
                                    temp = temp[["timestamp_unix", "timestamp_iso", "BVP"]]
                                    if emb_bvp_data is None:
                                        emb_bvp_data = temp.copy()
                                    else:
                                        emb_bvp_data = pd.concat([emb_bvp_data, temp], axis=0)
                                # STEPS
                                start_time=record["rawData"]["steps"]["timestampStart"]
                                if start_time > 0:
                                    sampling_rate=record["rawData"]["steps"]["samplingFrequency"]
                                    temp = pd.DataFrame({"step_counts":record["rawData"]["steps"]["values"]})
                                    temp["timestamp_unix"] = self.__make_timestamp_column(start_time, sampling_rate, temp.shape[0])
                                    temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                                    temp.columns=["step_counts", "timestamp_unix","timestamp_iso"]
                                    temp = temp[["timestamp_unix", "timestamp_iso", "step_counts"]]

                    if os.path.splitext(file.upper())[1] == ".CSV":
                        temp = pd.read_csv(os.path.join(source, file))
                        if "_EDA" in file.upper():
                            temp = temp[["timestamp_unix", "timestamp_iso", "eda_scl_usiemens"]]
                            if emb_eda_data is None:
                                emb_eda_data = temp.copy()
                            else:
                                emb_eda_data = pd.concat([emb_eda_data, temp], axis=0)
                        if "_RESPIRATORY-RATE"  in file.upper():
                            temp = temp[["timestamp_unix", "timestamp_iso", "respiratory_rate_brpm"]]
                            if emb_respiratory_rate_data is None:
                                emb_respiratory_rate_data = temp.copy()
                            else:
                                emb_respiratory_rate_data = pd.concat([emb_respiratory_rate_data, temp], axis=0)
                        if "_STEP-COUNTS"  in file.upper():
                            temp = temp[["timestamp_unix", "timestamp_iso", "step_counts"]]
                            if emb_step_counts_data is None:
                                emb_step_counts_data = temp.copy()
                            else:
                                emb_step_counts_data = pd.concat([emb_step_counts_data, temp], axis=0)
                        if "_TEMPERATURE"  in file.upper():
                            temp = temp[["timestamp_unix", "timestamp_iso", "temperature_celsius"]]
                            if emb_temp_data is None:
                                emb_temp_data = temp.copy()
                            else:
                                emb_temp_data = pd.concat([emb_temp_data, temp], axis=0)
                        if "_PRV"  in file.upper():
                            temp = temp[["timestamp_unix", "timestamp_iso", "prv_rmssd_ms"]]
                            if emb_prv_data is None:
                                emb_prv_data = temp.copy()
                            else:
                                emb_prv_data = pd.concat([emb_prv_data, temp], axis=0)
                        if "_ACCELEROMETERS-STD"  in file.upper():
                            temp = temp[["timestamp_unix", "timestamp_iso", "accelerometers_std_g"]]
                            if emb_acc_data is None:
                                emb_acc_data = temp.copy()
                            else:
                                emb_acc_data = pd.concat([emb_acc_data, temp], axis=0)
                        if "_ACTIVITY-COUNTS" in file.upper():
                            temp = temp[["timestamp_unix", "timestamp_iso", "activity_counts"]]
                            if emb_activity_counts_data is None:
                                emb_activity_counts_data = temp.copy()
                            else:
                                emb_activity_counts_data = pd.concat([emb_activity_counts_data, temp], axis=0)
                        if "_PULSE-RATE" in file.upper():
                            temp = temp[["timestamp_unix", "timestamp_iso", "pulse_rate_bpm"]]
                            if emb_pulse_rate_data is None:
                                emb_pulse_rate_data = temp.copy()
                            else:
                                emb_pulse_rate_data = pd.concat([emb_pulse_rate_data, temp], axis=0)
                if emb_bvp_data is not None:
                    emb_bvp_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_bvp_data = emb_bvp_data[["timestamp_unix", "timestamp_iso", "BVP"]]
                    start_time = emb_bvp_data.values[0][0] / 1000
                    datasets["BVP"] = Dataset("BVP", start_time, sampling_frequency_df(emb_bvp_data), emb_bvp_data)                
                if emb_sysp_data is not None:
                    emb_sysp_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_sysp_data["SYSP"] = emb_sysp_data["systolic_peaks"]
                    emb_sysp_data = emb_sysp_data[["timestamp_unix", "timestamp_iso", "SYSP"]]
                    start_time = emb_sysp_data.values[0][0] / 1000
                    datasets["SYSP"] = Dataset("SYSP", start_time, sampling_frequency_df(emb_sysp_data), emb_sysp_data)
                if emb_eda_data is not None:
                    emb_eda_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_eda_data["EDA"] = emb_eda_data["eda_scl_usiemens"]
                    emb_eda_data = emb_eda_data[["timestamp_unix", "timestamp_iso", "EDA"]]
                    start_time = emb_eda_data.values[0][0] / 1000
                    datasets["EDA"] = Dataset("EDA", start_time, sampling_frequency_df(emb_eda_data), emb_eda_data)
                if emb_respiratory_rate_data is not None:
                    emb_respiratory_rate_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_respiratory_rate_data["RR"] = emb_respiratory_rate_data["respiratory_rate_brpm"]
                    emb_respiratory_rate_data = emb_respiratory_rate_data[["timestamp_unix", "timestamp_iso", "RR"]]
                    start_time = emb_respiratory_rate_data.values[0][0] / 1000
                    datasets["RR"] = Dataset("RR", start_time, sampling_frequency_df(emb_respiratory_rate_data), emb_respiratory_rate_data)
                if emb_step_counts_data is not None:
                    emb_step_counts_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_step_counts_data["STEPS"] = emb_step_counts_data["step_counts"]
                    emb_step_counts_data = emb_step_counts_data[["timestamp_unix", "timestamp_iso", "STEPS"]]
                    start_time = emb_step_counts_data.values[0][0] / 1000
                    datasets["STEPS"] = Dataset("STEPS", start_time, sampling_frequency_df(emb_step_counts_data), emb_step_counts_data)
                if emb_temp_data is not None:
                    emb_temp_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_temp_data["TEMP"] = emb_temp_data["temperature_celsius"]
                    emb_temp_data = emb_temp_data[["timestamp_unix", "timestamp_iso", "TEMP"]]
                    start_time = emb_temp_data.values[0][0] / 1000
                    datasets["TEMP"] = Dataset("TEMP", start_time, sampling_frequency_df(emb_temp_data), emb_temp_data)
                if emb_prv_data is not None:
                    emb_prv_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_prv_data["PRV"] = emb_prv_data["prv_rmssd_ms"]
                    emb_prv_data = emb_prv_data[["timestamp_unix", "timestamp_iso", "PRV"]]
                    start_time = emb_prv_data.values[0][0] / 1000
                    datasets["PRV"] = Dataset("PRV", start_time, sampling_frequency_df(emb_prv_data), emb_prv_data)
                if emb_acc_data is not None:
                    emb_acc_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_acc_data["ACC"] = emb_acc_data["accelerometers_std_g"]
                    emb_acc_data = emb_acc_data[["timestamp_unix", "timestamp_iso", "ACC"]]
                    start_time = emb_acc_data.values[0][0] / 1000
                    datasets["ACC"] = Dataset("ACC", start_time, sampling_frequency_df(emb_acc_data), emb_acc_data)
                if emb_activity_counts_data is not None:
                    emb_activity_counts_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_activity_counts_data["ACTIVITY"] = emb_activity_counts_data["activity_counts"]
                    emb_activity_counts_data = emb_activity_counts_data[["timestamp_unix", "timestamp_iso", "ACTIVITY"]]
                    start_time = emb_activity_counts_data.values[0][0] / 1000
                    datasets["ACTIVITY"] = Dataset("ACTIVITY", start_time, sampling_frequency_df(emb_activity_counts_data), emb_activity_counts_data)
                if emb_pulse_rate_data is not None:
                    emb_pulse_rate_data.sort_values(by=["timestamp_iso"], ascending=True)
                    emb_pulse_rate_data["PR"] = emb_pulse_rate_data["pulse_rate_bpm"]
                    emb_pulse_rate_data = emb_pulse_rate_data[["timestamp_unix", "timestamp_iso", "PR"]]
                    start_time = emb_pulse_rate_data.values[0][0] / 1000
                    datasets["PR"] = Dataset("PR", start_time, sampling_frequency_df(emb_pulse_rate_data), emb_pulse_rate_data)
                return datasets
            else:
                raise ReaderException(101, "ReaderEmbracePlus.Read: source folder does not exists.")
        else:
            raise ReaderException(105, "ReaderEmbracePlus.Read: Invalid source type.")





