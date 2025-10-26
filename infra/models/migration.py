from dataclasses import dataclass

@dataclass
class Library:
    """Represent a certain version of a library"""
    lib_name: str
    version: str

@dataclass
class MigrationInfo:
    """Migration from library A to library B"""
    lib_a: Library
    lib_b: Library
