# Structures for storing function input parameters when not a simple string

# AWS download source for Embrace Plus
class AWSEmbracePlus:
    def __init__(self, subject: str, biomarker: str, path: str, access_key_id: str, secret_access_key: str):
        self.subject = subject
        self.biomarker = biomarker
        self.path = path
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key