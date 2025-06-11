"""
Table management for SwarndbDB
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, Any, Dict, List

import swarndb.remote_thrift.swarndb_thrift_rpc.ttypes as ttypes
from .index import IndexInfo
from .common import SwarndbException, INSERT_DATA
from .errors import ErrorCode


class ExplainType(Enum):
    """Types of query explanations."""
    Analyze = 1
    Ast = 2
    UnOpt = 3
    Opt = 4
    Physical = 5
    Pipeline = 6
    Fragment = 7

    def to_ttype(self) -> ttypes.ExplainType:
        """Convert to Thrift type."""
        # Using direct mapping for performance
        _EXPLAIN_TYPE_MAP = {
            ExplainType.Ast: ttypes.ExplainType.Ast,
            ExplainType.Analyze: ttypes.ExplainType.Analyze,
            ExplainType.UnOpt: ttypes.ExplainType.UnOpt,
            ExplainType.Opt: ttypes.ExplainType.Opt,
            ExplainType.Physical: ttypes.ExplainType.Physical,
            ExplainType.Pipeline: ttypes.ExplainType.Pipeline,
            ExplainType.Fragment: ttypes.ExplainType.Fragment,
        }
        try:
            return _EXPLAIN_TYPE_MAP[self]
        except KeyError:
            raise SwarndbException(ErrorCode.INVALID_EXPLAIN_TYPE, "Unknown explain type")


class Table(ABC):
    """Abstract base class for table operations."""
    __slots__ = ('name', 'schema')

    def __init__(self, name: str, schema: Dict[str, Any]):
        self.name = name
        self.schema = schema

    @abstractmethod
    def insert(self, data: INSERT_DATA) -> None:
        """Insert data into table."""
        pass

    @abstractmethod
    def create_index(self, index_info: IndexInfo) -> None:
        """Create an index on the table."""
        pass

    @abstractmethod
    def drop_index(self, column_name: str) -> None:
        """Drop an index from the table."""
        pass

    # Add other abstract methods as needed