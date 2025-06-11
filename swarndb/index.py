"""
Index management for SwarndbDB
"""

from enum import Enum
from typing import Dict, List, Optional, Union

import swarndb.remote_thrift.swarndb_thrift_rpc.ttypes as ttypes
from swarndb.common import SwarndbException
from swarndb.errors import ErrorCode


class IndexType(Enum):
    """Supported index types in SwarndbDB."""
    IVF = 1
    Hnsw = 2
    FullText = 3
    Secondary = 4
    EMVB = 5
    BMP = 6
    DiskAnn = 7

    def to_ttype(self):
        """Convert to Thrift type."""
        # Using direct mapping for performance instead of match statement
        if self is IndexType.IVF:
            return ttypes.IndexType.IVF
        elif self is IndexType.Hnsw:
            return ttypes.IndexType.Hnsw
        elif self is IndexType.FullText:
            return ttypes.IndexType.FullText
        elif self is IndexType.Secondary:
            return ttypes.IndexType.Secondary
        elif self is IndexType.EMVB:
            return ttypes.IndexType.EMVB
        elif self is IndexType.BMP:
            return ttypes.IndexType.BMP
        elif self is IndexType.DiskAnn:
            return ttypes.IndexType.DiskAnn
        else:
            raise SwarndbException(ErrorCode.INVALID_INDEX_TYPE, "Unknown index type")


class InitParameter:
    """Parameter for index initialization."""
    __slots__ = ('param_name', 'param_value')  # Using __slots__ for memory efficiency

    def __init__(self, param_name: str, param_value: str):
        self.param_name = param_name
        self.param_value = param_value

    def __str__(self) -> str:
        return f"InitParameter({self.param_name}, {self.param_value})"

    def __repr__(self) -> str:
        return self.__str__()

    def to_ttype(self):
        """Convert to Thrift type."""
        return ttypes.InitParameter(self.param_name, self.param_value)


class IndexInfo:
    """Information about an index."""
    __slots__ = ('column_name', 'index_type', 'params')  # Using __slots__ for memory efficiency

    def __init__(self, column_name: str, index_type: IndexType, params: Optional[Dict[str, str]] = None):
        self.column_name = column_name
        self.index_type = index_type

        # Fast path for None
        if params is None:
            self.params = None
            return

        # Type check only when params is provided
        if not isinstance(params, dict):
            raise SwarndbException(
                ErrorCode.INVALID_INDEX_PARAM,
                f"Expected dictionary for params, got {type(params).__name__}"
            )
        self.params = params

    def __str__(self) -> str:
        return f"IndexInfo({self.column_name}, {self.index_type}, {self.params})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, IndexInfo):
            return False
        return (self.column_name == other.index_name and
                self.index_type == other.index_type and
                self.params == other.params)

    def __hash__(self) -> int:
        return hash((self.column_name, self.index_type,
                    tuple(sorted(self.params.items())) if self.params else None))

    def to_ttype(self):
        """Convert to Thrift type."""
        # Pre-allocate list with known size for better performance
        init_params_list = []

        if self.params:
            for key, value in self.params.items():
                if isinstance(value, str):
                    init_params_list.append(ttypes.InitParameter(key, value))
                else:
                    raise SwarndbException(
                        ErrorCode.INVALID_INDEX_PARAM,
                        f"Parameter value must be string, got {type(value).__name__}"
                    )

        # Strip column name only once
        return ttypes.IndexInfo(
            self.column_name.strip(),
            self.index_type.to_ttype(),
            init_params_list
        )