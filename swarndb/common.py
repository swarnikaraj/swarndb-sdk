"""
Common utilities and data structures for SwarndbDB SDK
"""

from pathlib import Path
from typing import Union, Optional, Dict, List
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from enum import Enum


class NetworkAddress:
    """Represents a network address with IP and port."""

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    @classmethod
    def from_uri(cls, uri: str) -> "NetworkAddress":
        """Create NetworkAddress from URI string."""
        if uri.startswith("swarndb://"):
            uri = uri[9:]
        host, port = uri.split(":")
        return cls(host, int(port))

    def __str__(self) -> str:
        return f"swarndb://{self.ip}:{self.port}"

    def __repr__(self) -> str:
        return f"NetworkAddress(ip='{self.ip}', port={self.port})"


@dataclass
class SparseVector:
    """Represents a sparse vector with indices and optional values."""

    indices: List[int]
    values: Optional[List[Union[float, int]]] = None

    def __post_init__(self):
        if self.values is not None and len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Convert to dictionary format."""
        if self.values is None:
            raise ValueError("SparseVector.values is None")
        return {str(idx): val for idx, val in zip(self.indices, self.values)}

    @classmethod
    def from_dict(cls, data: Dict) -> "SparseVector":
        """Create SparseVector from dictionary."""
        return cls(data["indices"], data.get("values"))


@dataclass
class Array:
    """Dynamic array implementation."""

    elements: List = field(default_factory=list)

    def __init__(self, *args):
        self.elements = list(args)

    def append(self, element):
        self.elements.append(element)

    def __str__(self) -> str:
        return f"Array({', '.join(str(e) for e in self.elements)})"


# Type aliases
URI = Union[NetworkAddress, Path]
VEC = Union[List, NDArray]
INSERT_DATA = Dict[str, Union[str, int, float, List[Union[int, float]], SparseVector, Dict, Array]]

# Constants
LOCAL_HOST = NetworkAddress("127.0.0.1", 23817)
LOCAL_SWARNDB_PATH = "/var/swarndb"
DEFAULT_MATCH_VECTOR_TOPN = 10
DEFAULT_MATCH_SPARSE_TOPN = 10


class ConflictType(Enum):
    """Conflict handling types."""
    ERROR = 0
    IGNORE = 1
    REPLACE = 2


class SortType:
    """Enumeration for sorting directions."""
    ASC = 0
    DESC = 1

class SwarndbException(Exception):
    def __init__(self, error_code=0, error_message=None):
        self.error_code = error_code
        self.error_message = error_message