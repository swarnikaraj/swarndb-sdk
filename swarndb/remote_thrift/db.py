"""
Remote database implementation for SwarndbDB
"""

from abc import ABC
from typing import Dict, Any, Optional, Union, List

import numpy as np
import swarndb.remote_thrift.swarndb_thrift_rpc.ttypes as ttypes
from swarndb.db import Database
from swarndb.errors import ErrorCode
from swarndb.remote_thrift.table import RemoteTable
from swarndb.remote_thrift.utils import (
    check_valid_name,
    name_validity_check,
    select_res_to_polars,
    get_ordinary_info,
)
from swarndb.common import ConflictType, SwarndbException
from swarndb.index import IndexInfo
from swarndb.remote_thrift.query_builder import Query, SwarndbThriftQueryBuilder, ExplainQuery
from swarndb.remote_thrift.types import build_result
from swarndb.remote_thrift.utils import (
    parsed_expression_to_string,
    search_to_string,
)
from swarndb.table import ExplainType
from swarndb.utils import deprecated_api


class RemoteDatabase(Database, ABC):
    """Remote database implementation using Thrift protocol."""

    __slots__ = ('_conn', '_db_name')

    def __init__(self, conn: Any, name: str):
        """
        Initialize remote database.

        Args:
            conn: Database connection
            name: Database name
        """
        self._conn = conn
        self._db_name = name

    @name_validity_check("table_name", "Table")
    def create_table(
        self,
        table_name: str,
        columns_definition: Dict[str, Dict[str, Any]],
        conflict_type: ConflictType = ConflictType.ERROR
    ) -> RemoteTable:
        """
        Create a new table in the database.

        Args:
            table_name: Name of the table
            columns_definition: Column definitions dictionary
            conflict_type: How to handle naming conflicts

        Returns:
            RemoteTable: The created table object

        Raises:
            SwarndbException: If table creation fails
        """
        # Process column definitions
        column_defs = []
        [
            self._process_column(column_name, column_info, index, column_defs)
            for index, (column_name, column_info) in enumerate(columns_definition.items())
        ]

        # Map conflict type using uppercase enum members
        create_table_conflict = self._map_conflict_type(conflict_type)

        # Create table via connection
        res = self._conn.create_table(
            db_name=self._db_name,
            table_name=table_name,
            column_defs=column_defs,
            conflict_type=create_table_conflict
        )

        if res.error_code == ErrorCode.OK:
            return RemoteTable(self._conn, self._db_name, table_name)

        raise SwarndbException(res.error_code, res.error_msg)

    def _process_column(
        self,
        column_name: str,
        column_info: Dict[str, Any],
        index: int,
        column_defs: List
    ) -> None:
        """Process a single column definition."""
        check_valid_name(column_name, "Column")
        get_ordinary_info(column_info, column_defs, column_name, index)

    @staticmethod
    def _map_conflict_type(conflict_type: ConflictType) -> ttypes.CreateConflict:
        """Map ConflictType to Thrift CreateConflict."""
        conflict_map = {
            ConflictType.ERROR: ttypes.CreateConflict.Error,
            ConflictType.IGNORE: ttypes.CreateConflict.Ignore,
            ConflictType.REPLACE: ttypes.CreateConflict.Replace
        }

        if conflict_type not in conflict_map:
            raise SwarndbException(ErrorCode.INVALID_CONFLICT_TYPE, "Invalid conflict type")

        return conflict_map[conflict_type]

    @name_validity_check("table_name", "Table")
    def drop_table(
        self,
        table_name: str,
        conflict_type: ConflictType = ConflictType.ERROR
    ) -> Any:
        """
        Drop a table from the database.

        Args:
            table_name: Name of the table to drop
            conflict_type: How to handle conflicts

        Returns:
            Response from the server

        Raises:
            SwarndbException: If conflict_type is invalid
        """
        if conflict_type not in (ConflictType.ERROR, ConflictType.IGNORE):
            raise SwarndbException(ErrorCode.INVALID_CONFLICT_TYPE, "Invalid conflict type")

        drop_conflict = (
            ttypes.DropConflict.Error
            if conflict_type == ConflictType.ERROR
            else ttypes.DropConflict.Ignore
        )

        return self._conn.drop_table(
            db_name=self._db_name,
            table_name=table_name,
            conflict_type=drop_conflict
        )

    def list_tables(self) -> Any:
        """
        List all tables in the database.

        Returns:
            List of tables

        Raises:
            SwarndbException: If operation fails
        """
        res = self._conn.list_tables(self._db_name)
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    @name_validity_check("table_name", "Table")
    def show_table(self, table_name: str) -> Any:
        """
        Show table details.

        Args:
            table_name: Name of the table

        Returns:
            Table details

        Raises:
            SwarndbException: If operation fails
        """
        res = self._conn.show_table(
            db_name=self._db_name,
            table_name=table_name
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    @name_validity_check("table_name", "Table")
    def get_table(self, table_name: str) -> RemoteTable:
        """
        Get a table object.

        Args:
            table_name: Name of the table

        Returns:
            RemoteTable: Table object

        Raises:
            SwarndbException: If operation fails
        """
        res = self._conn.get_table(
            db_name=self._db_name,
            table_name=table_name
        )
        if res.error_code == ErrorCode.OK:
            return RemoteTable(self._conn, self._db_name, table_name)
        raise SwarndbException(res.error_code, res.error_msg)

    def show_tables(self) -> Any:
        """
        Show all tables with details.

        Returns:
            Polars DataFrame with table details

        Raises:
            SwarndbException: If operation fails
        """
        res = self._conn.show_tables(self._db_name)
        if res.error_code == ErrorCode.OK:
            return select_res_to_polars(res)
        raise SwarndbException(res.error_code, res.error_msg)
