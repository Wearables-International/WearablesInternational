# Exception classes

class ReaderException(Exception):
    def __init__(self, error_code: int, error_message: str):
        super().__init__(f"[Error {error_code}]: {error_message}")
        self.error_code = error_code
        self.error_message = error_message

class PreprocessingException(Exception):
    def __init__(self, error_code: int, error_message: str):
        super().__init__(f"[Error {error_code}]: {error_message}")
        self.error_code = error_code
        self.error_message = error_message

class MLModelException(Exception):
    def __init__(self, error_code: int, error_message: str):
        super().__init__(f"[Error {error_code}]: {error_message}")
        self.error_code = error_code
        self.error_message = error_message

# Readers: 100, 101,102,103, 104, 105, 106
# Preprocessing: 201,202,203
# MLModels: 301, 302, 303, 304