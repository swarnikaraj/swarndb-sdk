"""
Remote Thrift connection implementation for SwarndbDB
"""

from abc import ABC
import logging
from typing import Optional, Dict, Any

import swarndb.remote_thrift.swarndb_thrift_rpc.ttypes as ttypes
from swarndb import SwarndbConnection
from swarndb.errors import ErrorCode
from swarndb.remote_thrift.client import ThriftSwarndbClient
from swarndb.remote_thrift.db import RemoteDatabase
from swarndb.remote_thrift.utils import name_validity_check
from swarndb.common import ConflictType, SwarndbException


class RemoteThriftSwarndbConnection(SwarndbConnection, ABC):
    """Remote Thrift connection implementation for SwarndbDB."""

    __slots__ = ('db_name', '_client', '_is_connected')

    # Mapping for conflict types to maintain server compatibility
    _CONFLICT_TYPE_MAP = {
        ConflictType.ERROR: ttypes.CreateConflict.Error,
        ConflictType.IGNORE: ttypes.CreateConflict.Ignore,
        ConflictType.REPLACE: ttypes.CreateConflict.Replace
    }

    _DROP_CONFLICT_TYPE_MAP = {
        ConflictType.ERROR: ttypes.DropConflict.Error,
        ConflictType.IGNORE: ttypes.DropConflict.Ignore
    }

    def __init__(self, uri: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize remote Thrift connection.

        Args:
            uri: Connection URI
            logger: Optional logger instance
        """
        super().__init__(uri)
        self.db_name = "default_db"
        self._client = ThriftSwarndbClient(uri, logger=logger)
        self._is_connected = True

    def __del__(self):
        """Cleanup on deletion."""
        if getattr(self, '_is_connected', False):
            try:
                self.disconnect()
            except Exception:
                pass  # Ignore errors during cleanup

    @name_validity_check("db_name", "DB")
    def create_database(
        self,
        db_name: str,
        conflict_type: ConflictType = ConflictType.ERROR,
        comment: Optional[str] = None
    ) -> RemoteDatabase:
        """
        Create a new database.

        Args:
            db_name: Name of the database
            conflict_type: How to handle naming conflicts
            comment: Optional database comment

        Returns:
            RemoteDatabase instance

        Raises:
            SwarndbException: If creation fails
        """
        try:
            create_database_conflict = self._CONFLICT_TYPE_MAP[conflict_type]
        except KeyError:
            raise SwarndbException(
                ErrorCode.INVALID_CONFLICT_TYPE,
                "Invalid conflict type"
            )

        res = self._client.create_database(
            db_name=db_name,
            conflict_type=create_database_conflict,
            comment=comment
        )

        if res.error_code == ErrorCode.OK:
            return RemoteDatabase(self._client, db_name)
        raise SwarndbException(res.error_code, res.error_msg)

    def list_databases(self) -> Any:
        """
        List all databases.

        Returns:
            List of databases

        Raises:
            SwarndbException: If operation fails
        """
        res = self._client.list_databases()
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    @name_validity_check("db_name", "DB")
    def show_database(self, db_name: str) -> Any:
        """
        Show database details.

        Args:
            db_name: Name of the database

        Returns:
            Database details

        Raises:
            SwarndbException: If operation fails
        """
        res = self._client.show_database(db_name=db_name)
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    @name_validity_check("db_name", "DB")
    def drop_database(
        self,
        db_name: str,
        conflict_type: ConflictType = ConflictType.ERROR
    ) -> Any:
        """
        Drop a database.

        Args:
            db_name: Name of the database
            conflict_type: How to handle conflicts

        Returns:
            Operation result

        Raises:
            SwarndbException: If operation fails
        """
        try:
            drop_database_conflict = self._DROP_CONFLICT_TYPE_MAP[conflict_type]
        except KeyError:
            raise SwarndbException(
                ErrorCode.INVALID_CONFLICT_TYPE,
                "Invalid conflict type"
            )

        res = self._client.drop_database(
            db_name=db_name,
            conflict_type=drop_database_conflict
        )

        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    @name_validity_check("db_name", "DB")
    def get_database(self, db_name: str) -> RemoteDatabase:
        """
        Get a database instance.

        Args:
            db_name: Name of the database

        Returns:
            RemoteDatabase instance

        Raises:
            SwarndbException: If operation fails
        """
        res = self._client.get_database(db_name)
        if res.error_code == ErrorCode.OK:
            return RemoteDatabase(self._client, db_name)
        raise SwarndbException(res.error_code, res.error_msg)

    def show_current_node(self) -> Any:
        """
        Show current node information.

        Returns:
            Node information

        Raises:
            SwarndbException: If operation fails
        """
        res = self._client.show_current_node()
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def cleanup(self) -> Any:
        """
        Clean up resources.

        Returns:
            Operation result

        Raises:
            SwarndbException: If operation fails
        """
        res = self._client.cleanup()
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def optimize(
        self,
        db_name: str,
        table_name: str,
        optimize_opt: ttypes.OptimizeOptions
    ) -> Any:
        """
        Optimize a table.

        Args:
            db_name: Database name
            table_name: Table name
            optimize_opt: Optimization options

        Returns:
            Operation result

        Raises:
            SwarndbException: If operation fails
        """
        res = self._client.optimize(db_name, table_name, optimize_opt)
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def test_command(self, command_content: str) -> None:
        """
        Execute a test command.

        Args:
            command_content: Command content
        """
        command = ttypes.CommandRequest(
            command_type="test_command",
            test_command_content=command_content
        )
        self._client.command(command)

    def flush_data(self) -> None:
        """Flush data to disk."""
        self._client.flush(
            ttypes.FlushRequest(flush_type="data")
        )

    def flush_delta(self) -> None:
        """Flush delta to disk."""
        self._client.flush(
            ttypes.FlushRequest(flush_type="delta")
        )

    def disconnect(self) -> Any:
        """
        Disconnect from the server.

        Returns:
            Operation result

        Raises:
            SwarndbException: If operation fails
        """
        res = self._client.disconnect()
        if res.error_code == ErrorCode.OK:
            self._is_connected = False
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    @property
    def client(self) -> ThriftSwarndbClient:
        """Get the Thrift client instance."""
        return self._client