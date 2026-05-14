from dataclasses import dataclass


@dataclass
class FileEntry:
    file_path: str
    category: str = "Default"
