import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from wearablesinternational.Exceptions import ReaderException


# Dataset class
# Returned by all Reader.read functions containing the data and metadata read
class Dataset():
    def __init__(self, biomarker_type: str, start_time: float, frequency_hz: float, dataframe: pd.DataFrame):
        super().__init__()
        if not isinstance(biomarker_type, str) or not biomarker_type.strip():
            raise ReaderException(104, "Dataset: biomarker_type must be a non-empty string.")
        if start_time < 0:
            raise ReaderException(104, "Dataset: start_time must be a non-negative float.")
        if frequency_hz < 0:
            raise ReaderException(104, "Dataset: frequency_hz must be a non-negative number.")
        if not isinstance(dataframe, pd.DataFrame):
            raise ReaderException(104, "Dataset: dataframe must be a pandas DataFrame.")
        if dataframe.empty:
            raise ReaderException(104, "Dataset: dataframe cannot be empty.")
        self.biomarker_type = biomarker_type
        self.start_time = start_time
        if (len(str(int(start_time)))) < 13: 
            # seconds format
            self.utc_time = datetime.fromtimestamp(start_time)
        else:
            # milliseconds format
            self.utc_time = datetime.fromtimestamp(start_time / 1000)
        self.frequency_hz = frequency_hz
        self.dataframe = dataframe
        self.record_count = dataframe.shape[0]


# Abstract Reader class
# This class defines all possible reader methods
class Reader(ABC):
    @abstractmethod
    def read(self, source):
        pass
