from wearablesinternational.Devices.Readers import Reader, ReaderException
from wearablesinternational.Devices.EmpaticaE4 import ReaderE4
from wearablesinternational.Devices.OuraRing import ReaderOuraRing
from wearablesinternational.Devices.EmbracePlus import ReaderEmbracePlus
from wearablesinternational.Devices.NoWatch import ReaderNoWatch

# A class factory for instantiating a device reader
class ReaderFactory:
    _readers = {
        "E4": ReaderE4,
        "EmbracePlus": ReaderEmbracePlus,
        "OuraRing": ReaderOuraRing,
        "NoWatch": ReaderNoWatch,
    }

    @staticmethod
    def GetReader(reader_type: str) -> Reader:
        reader_class = ReaderFactory._readers.get(reader_type)
        if not reader_class:
            raise ReaderException(100, f"Reader type '{reader_type}' is not implemented.")
        return reader_class()

    @staticmethod
    def ListReaders() -> list:
        return list(ReaderFactory._readers.keys())