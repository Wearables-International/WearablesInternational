import os
import struct
import pandas as pd
from array import array
from enum import IntEnum
from csv import DictReader
from itertools import chain
from collections import namedtuple
from datetime import datetime, timezone
from wearablesinternational.Exceptions import ReaderException
from wearablesinternational.Devices.Readers import Reader, Dataset
from wearablesinternational.Preprocessing import upsample_list_to
from wearablesinternational.Utils import acc_to_g, scl_to_eda, rri_to_rr, sampling_frequency_df

HeaderItem = namedtuple("HeaderItem", ["name", "format", "type"])


class BodyPosition(IntEnum):
    LEFT = 1
    RIGHT = 2
    UNSPECIFIED = 3


class BaseData:
    header_def = [
        HeaderItem("sequence", "B", int),
        HeaderItem("quality", "B", int),
        HeaderItem("body_position", "B", BodyPosition),
        HeaderItem("sample_format", "B", int),
    ]

    def __init__(self, timestamp: str, payload: str):
        self.timestamp = datetime.fromtimestamp(int(timestamp))
        self.payload = bytes.fromhex(payload)

    @property
    def header_format(self):
        items = "".join([item.format for item in self.header_def])
        return f"<{items}"

    @property
    def header_size(self):
        return struct.calcsize(self.header_format)

    @property
    def meta(self):
        head = self.payload[:self.header_size]
        header_data = struct.unpack(self.header_format, head)
        return {item.name: item.type(value)
                for item, value in zip(self.header_def, header_data)}

    @property
    def body(self):
        return self.payload[self.header_size:]

    @staticmethod
    def as_24bit_ints(body: bytes):
        def convert(chunk: bytes):
            return struct.unpack("I", chunk + b"\0")[0]

        chunks = [body[i:i + 3] for i in range(0, len(body), 3)]
        return list(map(convert, chunks))


class TwentyFourBitData(BaseData):
    @property
    def data(self):
        return BaseData.as_24bit_ints(self.body)


class AccelerometerData(BaseData):
    identifier = "ACC"

    @property
    def data(self):
        samples = struct.iter_unpack("<hhh", self.body)
        return list(map(self.scale, samples))

    def scale(self, sample):
        return tuple(map(self.to_ms_2, sample))

    def to_ms_2(self, x):
        # convert to g and then to ms^-2
        return x / 512 * 9.80665


class SkinConductanceData(TwentyFourBitData):
    identifier = "SKIN_CONDUCTANCE"


class BiozADCData(TwentyFourBitData):
    identifier = "ADC"
    header_def = [
        HeaderItem("sequence", "B", int),
        HeaderItem("quality", "B", int),
        HeaderItem("offset", "H", int),
    ]


class Repeated4x(int):
    def __new__(cls, v):
        return super().__new__(cls, v & 0xff)


class BasePPGData(BaseData):
    header_def = [
        HeaderItem("sequence", "B", int),
        HeaderItem("quality", "B", int),
        HeaderItem("body_position", "B", BodyPosition),
        HeaderItem("sample_format", "B", int),  # always 0x60
        HeaderItem("stream_location", "B", int),
        HeaderItem("ppg_offset", "B", int),  # always 0
        HeaderItem("ppg_exponent", "B", int),  # always 0
        HeaderItem("led_power", "I", Repeated4x),
        HeaderItem("adc_gain", "I", Repeated4x),
    ]

    @property
    def data(self):
        return array("H", self.body)

    def __repr__(self):
        stream_location = f"stream_location={self.meta["stream_location"]}"
        return f"<{len(self.data)} PPG values @ {self.timestamp} {stream_location}>"


class PPGAmbientData(BasePPGData):
    identifier = "PPG_SINGLE_AMBIENT"


class PPGGreenData(BasePPGData):
    identifier = "PPG_SINGLE_GREEN"


class ReaderNoWatch(Reader):
    def __init__(self):
        super().__init__()
        self.__processors = {v.identifier: v for k, v in globals().items() if hasattr(v, "identifier")}

    
    def process_row(self, row):
        if Processor := self.__processors.get(row["identifier"]):
            return Processor(row["timestamp"], row["value"])


    def read(self, source):
        if isinstance(source, str):
            if os.path.exists(source):
                files = os.listdir(source)
                datasets = {}
                for file in files:
                    if ".CSV" in file.upper():
                        # todo: sampling rates are wrong
                        if "ACTIVITY_COUNT" in file.upper():
                            temp = pd.read_csv(os.path.join(source, file), engine="pyarrow")
                            start_time = temp.values[0][0]
                            time_two = temp.values[1][0]
                            sampling_rate = 1 / (time_two - start_time)
                            temp["timestamp_unix"] = temp["timestamp"]
                            temp.drop(["timestamp", "quality_indicator"], inplace=True, axis=1)
                            temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                            temp.columns=["ACTIVITY_COUNT","BODY_POSITION","timestamp_unix","timestamp_iso"]
                            temp["ACTIVITY"] = temp["ACTIVITY_COUNT"]
                            temp = temp[['timestamp_unix', 'timestamp_iso', 'ACTIVITY']]
                            datasets["ACTIVITY"] = Dataset("ACTIVITY", sampling_frequency_df(temp), 0, temp.iloc[2:])
                        if "HEART_RATE" in file.upper():
                            temp = pd.read_csv(os.path.join(source, file), engine="pyarrow")
                            start_time = temp.values[0][0]
                            time_two = temp.values[1][0]
                            sampling_rate = 1 / (time_two - start_time)
                            temp["timestamp_unix"] = temp["timestamp"]
                            temp.drop(["timestamp", "quality_indicator"], inplace=True, axis=1)
                            temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                            temp.columns=["HR","timestamp_unix","timestamp_iso"]
                            temp = temp[['timestamp_unix', 'timestamp_iso', 'HR']]
                            datasets["HR"] = Dataset("HR", start_time, sampling_frequency_df(temp), temp.iloc[2:])
                        if "NATIVE_STEP_COUNT" in file.upper():
                            temp = pd.read_csv(os.path.join(source, file), engine="pyarrow")
                            start_time = temp.values[0][0]
                            time_two = temp.values[1][0]
                            sampling_rate = 1 / (time_two - start_time)
                            temp["timestamp_unix"] = temp["timestamp"]
                            temp.drop(["timestamp"], inplace=True, axis=1)
                            temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                            temp.columns=["STEPS","timestamp_unix","timestamp_iso"]
                            temp = temp[['timestamp_unix', 'timestamp_iso', 'STEPS']]
                            datasets["STEPS"] = Dataset("STEPS", start_time, sampling_frequency_df(temp), temp.iloc[2:])
                        if "RESPIRATION_RATE" in file.upper():
                            temp = pd.read_csv(os.path.join(source, file), engine="pyarrow")
                            start_time = temp.values[0][0]
                            time_two = temp.values[1][0]
                            sampling_rate = 1 / (time_two - start_time)
                            temp["timestamp_unix"] = temp["timestamp"]
                            temp.drop(["timestamp", "quality_indicator"], inplace=True, axis=1)
                            temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                            temp.columns=["RRI","timestamp_unix","timestamp_iso"]
                            temp = temp[['timestamp_unix', 'timestamp_iso', 'RRI']]
                            temp["RR"] = rri_to_rr(temp["RRI"].values)
                            temp = temp[['timestamp_unix', 'timestamp_iso', 'RR']]
                            datasets["RR"] = Dataset("RR", start_time, sampling_frequency_df(temp), temp.iloc[2:])
                        if "SKIN_CONDUCTANCE" in file.upper():
                            temp = pd.read_csv(os.path.join(source, file), engine="pyarrow")
                            start_time = temp.values[0][0]
                            time_two = temp.values[1][0]
                            sampling_rate = 1 / (time_two - start_time)
                            temp["timestamp_unix"] = temp["timestamp"]
                            temp.drop(["timestamp", "quality_indicator"], inplace=True, axis=1)
                            temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                            temp.columns=["EDA","timestamp_unix","timestamp_iso"]
                            temp = temp[['timestamp_unix', 'timestamp_iso', 'EDA']]
                            datasets["EDA"] = Dataset("EDA", start_time, sampling_frequency_df(temp), temp.iloc[2:])
                        if "TEMPERATURE" in file.upper():
                            temp = pd.read_csv(os.path.join(source, file), engine="pyarrow")
                            start_time = temp.values[0][0]
                            time_two = temp.values[1][0]
                            sampling_rate = 1 / (time_two - start_time)
                            temp["timestamp_unix"] = temp["timestamp"]
                            temp.drop(["timestamp", "quality_indicator"], inplace=True, axis=1)
                            temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                            temp.columns=["TEMP","timestamp_unix","timestamp_iso"]
                            temp = temp[['timestamp_unix', 'timestamp_iso', 'TEMP']]
                            datasets["TEMP"] = Dataset("TEMP", start_time, sampling_frequency_df(temp), temp.iloc[2:])                           
                    if ".TXT" in file.upper():
                        with open(os.path.join(source, file)) as f:
                            rows = filter(None, map(self.process_row, DictReader(f)))
                            output = list(reversed(list(rows)))
                            # PPG
                            ppg = list(filter(lambda e: e.identifier == "PPG_SINGLE_GREEN", output))
                            if len(ppg) > 0:
                                ppg2 = list(chain(*[row.data for row in ppg]))
                                ppg2_timestamps = [item.timestamp.timestamp() for item in ppg]
                                sampling_rate = len(ppg2) / len(ppg2_timestamps)
                                ppg2_timestamps = upsample_list_to(ppg2_timestamps, len(ppg2))
                                start_time = ppg2_timestamps[0]
                                temp = pd.DataFrame(columns=["timestamp_unix","timestamp_iso", "PPG"])
                                temp["PPG"] = ppg2
                                temp["timestamp_unix"] = ppg2_timestamps
                                temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                                datasets["PPG"] = Dataset("PPG", start_time, sampling_rate, temp)
                            # ACC
                            acc = list(filter(lambda e: e.identifier == "ACC", output))
                            if len(acc) > 0:
                                acct = list(chain(*[row.data for row in acc]))
                                acc_timestamps = [item.timestamp.timestamp() for item in acc]
                                sampling_rate = len(acct) / len(acc_timestamps)
                                acc_timestamps = upsample_list_to(acc_timestamps, len(acct))
                                start_time = acc_timestamps[0]
                                temp = pd.DataFrame(columns=["X", "Y", "Z", "timestamp_unix","timestamp_iso"])
                                temp["X"] = list(map(lambda t: t[0], acct))
                                temp["Y"] = list(map(lambda t: t[1], acct))
                                temp["Z"] = list(map(lambda t: t[2], acct))
                                temp["ACC"] = acc_to_g(temp["X"].tolist(), temp["Y"].tolist(), temp["Z"].tolist())
                                temp["timestamp_unix"] = acc_timestamps
                                temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                                temp = temp[["timestamp_unix","timestamp_iso", "ACC"]]
                                datasets["ACC"] = Dataset("ACC", start_time, sampling_rate, temp)
                            # SKIN_CONDUCTANCE
                            skin = list(filter(lambda e: e.identifier == "SKIN_CONDUCTANCE", output))
                            if len(skin) > 0:
                                scl = list(chain(*[row.data for row in skin]))
                                scl_timestamps = [item.timestamp.timestamp() for item in skin]
                                sampling_rate = len(scl) / len(scl_timestamps)
                                scl_timestamps = upsample_list_to(scl_timestamps, len(scl))
                                start_time = scl_timestamps[0]
                                temp = pd.DataFrame(columns=["timestamp_unix","timestamp_iso", "EDA"])
                                temp["EDA"] = scl_to_eda(scl, sampling_rate)
                                temp["timestamp_unix"] = scl_timestamps
                                temp["timestamp_iso"] = temp["timestamp_unix"].apply(lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                                datasets["EDA"] = Dataset("EDA", start_time, sampling_rate, temp)
                return datasets
            else:
                raise ReaderException(101, "ReaderNoWatch.Read: source folder does not exists.")
        else:
            raise ReaderException(105, "ReaderNoWatch.Read: Invalid source type.")