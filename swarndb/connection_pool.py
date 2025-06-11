"""
Connection pool implementation for SwarndbDB
"""

from threading import Lock
from typing import List, Optional
import logging

from . import swarndb
from .common import NetworkAddress
from swarndb.swarndb import SwarndbConnection


class ConnectionPool:
    """
    Manages a pool of connections to SwarndbDB.

    Provides thread-safe connection management with automatic connection
    creation and cleanup.
    """

    def __init__(
        self,
        uri: NetworkAddress = NetworkAddress("127.0.0.1", 23817),
        max_size: int = 16,
        logger: Optional[logging.Logger] = None
    ):
        self.uri = uri
        self.max_size = max_size
        self.logger = logger or logging.getLogger(__name__)
        self.free_pool: List[SwarndbConnection] = []
        self.lock = Lock()

        # Initialize connection pool
        for _ in range(max_size):
            self._create_conn()

    def __del__(self):
        self.destroy()

    def _create_conn(self) -> None:
        """Create a new connection and add it to the pool."""
        conn = swarndb.connect(self.uri, logger=self.logger)
        self.free_pool.append(conn)

    def get_conn(self) -> SwarndbConnection:
        """
        Get a connection from the pool.

        Returns:
            SwarndbConnection: A database connection

        Raises:
            Exception: If unable to create new connection
        """
        with self.lock:
            if not self.free_pool:
                self._create_conn()
            conn = self.free_pool.pop()
            self.logger.debug("Connection acquired from pool")
            return conn

    def release_conn(self, conn: SwarndbConnection) -> None:
        """
        Return a connection to the pool.

        Args:
            conn: The connection to release

        Raises:
            Exception: If connection was already released
        """
        with self.lock:
            if conn in self.free_pool:
                raise Exception("Connection has already been released")
            if len(self.free_pool) < self.max_size:
                self.free_pool.append(conn)
            self.logger.debug("Connection released to pool")

    def destroy(self) -> None:
        """Clean up all connections in the pool."""
        for conn in self.free_pool:
            try:
                conn.disconnect()
            except Exception as e:
                self.logger.warndbing(f"Error disconnecting: {e}")
        self.free_pool.clear()