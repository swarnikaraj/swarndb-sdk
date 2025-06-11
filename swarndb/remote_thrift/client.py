"""
Thrift client implementation for SwarndbDB
"""

import logging
from functools import wraps
from typing import Optional, List, Dict, Any, Union, Callable
from readerwriterlock import rwlock

from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket
from thrift.transport.TTransport import TTransportException
from thrift.transport import TTransport


from swarndb import URI
from swarndb.remote_thrift.swarndb_thrift_rpc import swarndbService
from swarndb.remote_thrift.swarndb_thrift_rpc.ttypes import (
    ConnectRequest, CommonRequest, CommonResponse, CreateDatabaseRequest,
    DropDatabaseRequest, ListDatabaseRequest, ShowDatabaseRequest,
    GetDatabaseRequest, CreateTableRequest, DropTableRequest, ListTableRequest,
    ShowTableRequest, ShowColumnsRequest, GetTableRequest, CreateIndexRequest,
    DropIndexRequest, ShowIndexRequest, ListIndexRequest, InsertRequest,
    ImportRequest, ExportRequest, SelectRequest, ExplainRequest, DeleteRequest,
    UpdateRequest, ShowTablesRequest, ShowSegmentsRequest, ShowSegmentRequest,
    ShowBlocksRequest, ShowBlockRequest, ShowBlockColumnRequest,
    ShowCurrentNodeRequest, OptimizeRequest, AddColumnsRequest,
    DropColumnsRequest, CommandRequest, FlushRequest, CompactRequest,
    CreateConflict, DropConflict, CreateOption, DropOption, Field,
    OptimizeOptions, IndexInfo
)
from swarndb.errors import ErrorCode
from swarndb.common import SwarndbException

# Constants
TRY_TIMES = 10
CLIENT_VERSION = 29  # 0.6.0.dev3


class ThriftSwarndbClient:
    """
    Thrift client for SwarndbDB.

    Handles communication with the SwarndbDB server using the Thrift protocol.
    """
    __slots__ = (
        'lock', 'session_id', 'uri', 'transport', 'protocol', 'client',
        '_is_connected', 'session_i', 'try_times', 'logger'
    )

    def __init__(
        self,
        uri: URI,
        *,
        try_times: int = TRY_TIMES,
        logger: Optional[logging.Logger] = None,
        timeout: float = 30.0
    ):
        """
        Initialize the Thrift client.

        Args:
            uri: The server URI
            try_times: Number of connection retry attempts
            logger: Optional logger instance
            timeout: Connection timeout in seconds
        """
        self.lock = rwlock.RWLockRead()
        self.session_id = -1
        self.uri = uri
        self.transport = None
        self.try_times = try_times
        self._is_connected = False
        self.session_i = 0

        # Initialize logger first so we can log connection issues
        self._init_logger(logger)

        # Connect to server
        self._reconnect(timeout)
        self._is_connected = True

    def _init_logger(self, logger: Optional[logging.Logger]) -> None:
        """Initialize the logger."""
        if logger is not None:
            self.logger = logger
            return

        self.logger = logging.getLogger("ThriftSwarndbClient")
        self.logger.setLevel(logging.DEBUG)

        # Only add handler if none exists
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def __del__(self) -> None:
        """Clean up resources on deletion."""
        if hasattr(self, '_is_connected') and self._is_connected:
            try:
                self.disconnect()
            except Exception:
                pass  # Ignore errors during cleanup

    def _reconnect(self, timeout: float = 30.0) -> None:
        """
        Reconnect to the server.

        Args:
            timeout: Connection timeout in seconds

        Raises:
            SwarndbException: If connection fails
        """
        # Close existing transport if any
        if self.transport is not None:
            self.transport.close()
            self.transport = None

        # Create socket with timeout
        socket = TSocket.TSocket(self.uri.ip, self.uri.port)
        socket.setTimeout(int(timeout * 1000))  # Convert to milliseconds

        # Use buffered transport for synchronous communication
        self.transport = TTransport.TBufferedTransport(socket)
        self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        self.client = swarndbService.Client(self.protocol)

        try:
            self.transport.open()

            # Connect with client version
            res = self.client.Connect(ConnectRequest(client_version=CLIENT_VERSION))
            if res.error_code != 0:
                raise SwarndbException(res.error_code, res.error_msg)

            self.session_id = res.session_id
            self.logger.debug(f"Connected to server, session_id: {self.session_id}")

        except Exception as e:
            if self.transport:
                self.transport.close()
            raise SwarndbException(
                ErrorCode.CONNECTION_FAILED,
                f"Failed to connect to server: {str(e)}"
            ) from e

    @staticmethod
    def retry_wrapper(func: Callable) -> Callable:
        """
        Decorator to retry operations on connection failure.

        Args:
            func: The function to wrap

        Returns:
            Wrapped function with retry logic
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for i in range(self.try_times):
                try:
                    with self.lock.gen_rlock():
                        old_session_i = self.session_i
                        ret = func(self, *args, **kwargs)
                        break
                except TTransportException as e:
                    with self.lock.gen_wlock():
                        if old_session_i == self.session_i:
                            self._reconnect()
                            self.session_i += 1
                            self.logger.debug(
                                f"Retry {i+1}/{self.try_times}, session_id: {self.session_id}, "
                                f"session_i: {self.session_i}, exception: {str(e)}"
                            )
                except Exception as e:
                    self.logger.error(f"Error in {func.__name__}: {str(e)}")
                    raise
            else:
                return CommonResponse(
                    ErrorCode.TOO_MANY_CONNECTIONS,
                    f"Failed after {self.try_times} connection attempts"
                )
            return ret
        return wrapper

    # Database operations
    @retry_wrapper
    def create_database(
        self,
        db_name: str,
        conflict_type: CreateConflict = CreateConflict.Error,
        comment: Optional[str] = None
    ):
        """Create a database."""
        db_comment = "" if comment is None else comment
        return self.client.CreateDatabase(
            CreateDatabaseRequest(
                session_id=self.session_id,
                db_name=db_name,
                db_comment=db_comment,
                create_option=CreateOption(conflict_type=conflict_type)
            )
        )

    @retry_wrapper
    def drop_database(
        self,
        db_name: str,
        conflict_type: DropConflict = DropConflict.Error
    ):
        """Drop a database."""
        return self.client.DropDatabase(
            DropDatabaseRequest(
                session_id=self.session_id,
                db_name=db_name,
                drop_option=DropOption(conflict_type=conflict_type)
            )
        )

    @retry_wrapper
    def list_databases(self):
        """List all databases."""
        return self.client.ListDatabase(
            ListDatabaseRequest(session_id=self.session_id)
        )

    @retry_wrapper
    def show_database(self, db_name: str):
        """Show database details."""
        return self.client.ShowDatabase(
            ShowDatabaseRequest(
                session_id=self.session_id,
                db_name=db_name
            )
        )

    @retry_wrapper
    def get_database(self, db_name: str):
        """Get a database."""
        return self.client.GetDatabase(
            GetDatabaseRequest(
                session_id=self.session_id,
                db_name=db_name
            )
        )

    # Table operations
    @retry_wrapper
    def create_table(
        self,
        db_name: str,
        table_name: str,
        column_defs,
        conflict_type: CreateConflict = CreateConflict.Error,
        properties: Optional[List] = None
    ):
        """Create a table."""
        return self.client.CreateTable(
            CreateTableRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                column_defs=column_defs,
                create_option=CreateOption(
                    conflict_type=conflict_type,
                    properties=properties
                )
            )
        )

    @retry_wrapper
    def drop_table(
        self,
        db_name: str,
        table_name: str,
        conflict_type: DropConflict = DropConflict.Error
    ):
        """Drop a table."""
        return self.client.DropTable(
            DropTableRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                drop_option=DropOption(conflict_type=conflict_type)
            )
        )

    @retry_wrapper
    def list_tables(self, db_name: str):
        """List all tables in a database."""
        return self.client.ListTable(
            ListTableRequest(
                session_id=self.session_id,
                db_name=db_name
            )
        )

    @retry_wrapper
    def show_table(self, db_name: str, table_name: str):
        """Show table details."""
        return self.client.ShowTable(
            ShowTableRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name
            )
        )

    @retry_wrapper
    def show_columns(self, db_name: str, table_name: str):
        """Show columns in a table."""
        return self.client.ShowColumns(
            ShowColumnsRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name
            )
        )

    @retry_wrapper
    def get_table(self, db_name: str, table_name: str):
        """Get a table."""
        return self.client.GetTable(
            GetTableRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name
            )
        )

    # Index operations
    @retry_wrapper
    def create_index(
        self,
        db_name: str,
        table_name: str,
        index_name: str,
        index_info: IndexInfo,
        conflict_type: CreateConflict = CreateConflict.Error,
        index_comment: str = ""
    ):
        """Create an index."""
        return self.client.CreateIndex(
            CreateIndexRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                index_name=index_name,
                index_comment=index_comment,
                index_info=index_info,
                create_option=CreateOption(conflict_type=conflict_type)
            )
        )

    @retry_wrapper
    def drop_index(
        self,
        db_name: str,
        table_name: str,
        index_name: str,
        conflict_type: DropConflict = DropConflict.Error
    ):
        """Drop an index."""
        return self.client.DropIndex(
            DropIndexRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                index_name=index_name,
                drop_option=DropOption(conflict_type=conflict_type)
            )
        )

    @retry_wrapper
    def show_index(self, db_name: str, table_name: str, index_name: str):
        """Show index details."""
        return self.client.ShowIndex(
            ShowIndexRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                index_name=index_name
            )
        )

    @retry_wrapper
    def list_indexes(self, db_name: str, table_name: str):
        """List all indexes on a table."""
        return self.client.ListIndex(
            ListIndexRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name
            )
        )

    # Data operations
    @retry_wrapper
    def insert(self, db_name: str, table_name: str, fields: List[Field]):
        """Insert data into a table."""
        return self.client.Insert(
            InsertRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                fields=fields
            )
        )

    @retry_wrapper
    def import_data(
        self,
        db_name: str,
        table_name: str,
        file_name: str,
        import_options
    ):
        """Import data from a file."""
        return self.client.Import(
            ImportRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                file_name=file_name,
                import_option=import_options
            )
        )

    @retry_wrapper
    def export_data(
        self,
        db_name: str,
        table_name: str,
        file_name: str,
        export_options: Dict,
        columns: List[str]
    ):
        """Export data to a file."""
        return self.client.Export(
            ExportRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                columns=columns,
                file_name=file_name,
                export_option=export_options
            )
        )

    @retry_wrapper
    def select(
        self,
        db_name: str,
        table_name: str,
        select_list,
        highlight_list,
        search_expr,
        where_expr,
        group_by_list,
        having_expr,
        limit_expr,
        offset_expr,
        order_by_list,
        total_hits_count
    ):
        """Execute a select query."""
        return self.client.Select(
            SelectRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                select_list=select_list,
                highlight_list=highlight_list,
                search_expr=search_expr,
                where_expr=where_expr,
                group_by_list=group_by_list,
                having_expr=having_expr,
                limit_expr=limit_expr,
                offset_expr=offset_expr,
                order_by_list=order_by_list,
                total_hits_count=total_hits_count
            )
        )

    @retry_wrapper
    def explain(
        self,
        db_name: str,
        table_name: str,
        select_list,
        highlight_list,
        search_expr,
        where_expr,
        group_by_list,
        limit_expr,
        offset_expr,
        explain_type
    ):
        """Explain a query execution plan."""
        return self.client.Explain(
            ExplainRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                select_list=select_list,
                highlight_list=highlight_list,
                search_expr=search_expr,
                where_expr=where_expr,
                group_by_list=group_by_list,
                limit_expr=limit_expr,
                offset_expr=offset_expr,
                explain_type=explain_type
            )
        )

    @retry_wrapper
    def delete(self, db_name: str, table_name: str, where_expr):
        """Delete data from a table."""
        return self.client.Delete(
            DeleteRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                where_expr=where_expr
            )
        )

    @retry_wrapper
    def update(
        self,
        db_name: str,
        table_name: str,
        where_expr,
        update_expr_array
    ):
        """Update data in a table."""
        return self.client.Update(
            UpdateRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                where_expr=where_expr,
                update_expr_array=update_expr_array
            )
        )

    @retry_wrapper
    def disconnect(self):
        """Disconnect from the server."""
        if not self._is_connected:
            return CommonResponse(ErrorCode.OK, "Already disconnected")

        try:
            res = self.client.Disconnect(
                CommonRequest(session_id=self.session_id)
            )
        except Exception as e:
            res = CommonResponse(ErrorCode.CLIENT_CLOSE, str(e))

        if self.transport:
            self.transport.close()

        self._is_connected = False
        return res

    # Additional operations
    @retry_wrapper
    def show_tables(self, db_name: str):
        """Show all tables in a database."""
        return self.client.ShowTables(
            ShowTablesRequest(
                session_id=self.session_id,
                db_name=db_name
            )
        )

    @retry_wrapper
    def show_segments(self, db_name: str, table_name: str):
        """Show segments in a table."""
        return self.client.ShowSegments(
            ShowSegmentsRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name
            )
        )

    @retry_wrapper
    def show_segment(self, db_name: str, table_name: str, segment_id: int):
        """Show a specific segment."""
        return self.client.ShowSegment(
            ShowSegmentRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                segment_id=segment_id
            )
        )

    @retry_wrapper
    def show_blocks(self, db_name: str, table_name: str, segment_id: int):
        """Show blocks in a segment."""
        return self.client.ShowBlocks(
            ShowBlocksRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                segment_id=segment_id
            )
        )

    @retry_wrapper
    def show_block(
        self,
        db_name: str,
        table_name: str,
        segment_id: int,
        block_id: int
    ):
        """Show a specific block."""
        return self.client.ShowBlock(
            ShowBlockRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                segment_id=segment_id,
                block_id=block_id
            )
        )

    @retry_wrapper
    def show_block_column(
        self,
        db_name: str,
        table_name: str,
        segment_id: int,
        block_id: int,
        column_id: int
    ):
        """Show a specific column in a block."""
        return self.client.ShowBlockColumn(
            ShowBlockColumnRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                segment_id=segment_id,
                block_id=block_id,
                column_id=column_id
            )
        )

    @retry_wrapper
    def show_current_node(self):
        """Show current node information."""
        return self.client.ShowCurrentNode(
            ShowCurrentNodeRequest(session_id=self.session_id)
        )

    @retry_wrapper
    def optimize(
        self,
        db_name: str,
        table_name: str,
        optimize_opt: OptimizeOptions
    ):
        """Optimize a table."""
        return self.client.Optimize(
            OptimizeRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                optimize_options=optimize_opt
            )
        )

    @retry_wrapper
    def add_columns(self, db_name: str, table_name: str, column_defs: List):
        """Add columns to a table."""
        return self.client.AddColumns(
            AddColumnsRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                column_defs=column_defs
            )
        )

    @retry_wrapper
    def drop_columns(self, db_name: str, table_name: str, column_names: List[str]):
        """Drop columns from a table."""
        return self.client.DropColumns(
            DropColumnsRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name,
                column_names=column_names
            )
        )

    @retry_wrapper
    def cleanup(self):
        """Clean up resources."""
        return self.client.Cleanup(
            CommonRequest(session_id=self.session_id)
        )

    @retry_wrapper
    def command(self, command: CommandRequest):
        """Execute a command."""
        command.session_id = self.session_id
        return self.client.Command(command)

    @retry_wrapper
    def flush(self, flush_request: FlushRequest):
        """Flush data to disk."""
        flush_request.session_id = self.session_id
        return self.client.Flush(flush_request)

    @retry_wrapper
    def compact(self, db_name: str, table_name: str):
        """Compact a table."""
        return self.client.Compact(
            CompactRequest(
                session_id=self.session_id,
                db_name=db_name,
                table_name=table_name
            )
        )