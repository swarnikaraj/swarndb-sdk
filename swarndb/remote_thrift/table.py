"""
Remote table implementation for SwarndbDB
"""

import json
import functools
import inspect
from typing import Optional, Union, List, Any, Dict, Tuple, Callable
from functools import lru_cache

from sqlglot import condition

import swarndb.remote_thrift.swarndb_thrift_rpc.ttypes as ttypes
from swarndb.common import (
    INSERT_DATA, VEC, SwarndbException, SparseVector,
    ConflictType, DEFAULT_MATCH_VECTOR_TOPN, SortType
)
from swarndb.errors import ErrorCode
from swarndb.index import IndexInfo
from swarndb.remote_thrift.query_builder import Query, SwarndbThriftQueryBuilder, ExplainQuery
from swarndb.remote_thrift.types import build_result
from swarndb.remote_thrift.utils import (
    traverse_conditions,
    name_validity_check,
    select_res_to_polars,
    check_valid_name,
    get_remote_constant_expr_from_python_value,
    get_ordinary_info,
    parsed_expression_to_string,
    search_to_string
)
from swarndb.table import ExplainType
from swarndb.utils import deprecated_api


class RemoteTable:
    """Remote table implementation using Thrift protocol."""

    __slots__ = ('_conn', '_db_name', '_table_name', 'query_builder')

    # Cache conflict type mappings
    _CREATE_CONFLICT_MAP = {
        ConflictType.ERROR: ttypes.CreateConflict.Error,
        ConflictType.IGNORE: ttypes.CreateConflict.Ignore,
        ConflictType.REPLACE: ttypes.CreateConflict.Replace
    }

    _DROP_CONFLICT_MAP = {
        ConflictType.ERROR: ttypes.DropConflict.Error,
        ConflictType.IGNORE: ttypes.DropConflict.Ignore
    }

    def __init__(self, conn: Any, db_name: str, table_name: str):
        """Initialize remote table."""
        self._conn = conn
        self._db_name = db_name
        self._table_name = table_name
        self.query_builder = SwarndbThriftQueryBuilder(table=self)

    @staticmethod
    def params_type_check(func: Callable) -> Callable:
        """Decorator for parameter type checking."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            params = sig.parameters

            # Check positional arguments
            for arg, param in zip(args, params.values()):
                if param.annotation is not param.empty and not isinstance(arg, param.annotation):
                    raise TypeError(f"Argument {param.name} must be {param.annotation}")

            # Check keyword arguments
            for kwarg, value in kwargs.items():
                if params[kwarg].annotation is not params[kwarg].empty and not isinstance(value, params[kwarg].annotation):
                    raise TypeError(f"Argument {kwarg} must be {params[kwarg].annotation}")

            return func(*args, **kwargs)
        return wrapper

    @name_validity_check("index_name", "Index")
    def create_index(
        self,
        index_name: str,
        index_info: IndexInfo,
        conflict_type: ConflictType = ConflictType.ERROR,
        index_comment: str = ""
    ) -> Any:
        """Create an index."""
        try:
            create_index_conflict = self._CREATE_CONFLICT_MAP[conflict_type]
        except KeyError:
            raise SwarndbException(
                ErrorCode.INVALID_CONFLICT_TYPE,
                "Invalid conflict type"
            )

        res = self._conn.create_index(
            db_name=self._db_name,
            table_name=self._table_name,
            index_name=index_name.strip(),
            index_info=index_info.to_ttype(),
            conflict_type=create_index_conflict,
            index_comment=index_comment
        )

        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    @name_validity_check("index_name", "Index")
    def drop_index(
        self,
        index_name: str,
        conflict_type: ConflictType = ConflictType.ERROR
    ) -> Any:
        """Drop an index."""
        try:
            drop_index_conflict = self._DROP_CONFLICT_MAP[conflict_type]
        except KeyError:
            raise SwarndbException(
                ErrorCode.INVALID_CONFLICT_TYPE,
                "Invalid conflict type"
            )

        res = self._conn.drop_index(
            db_name=self._db_name,
            table_name=self._table_name,
            index_name=index_name,
            conflict_type=drop_index_conflict
        )

        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    @name_validity_check("index_name", "Index")
    def show_index(self, index_name: str) -> Any:
        """Show index details."""
        res = self._conn.show_index(
            db_name=self._db_name,
            table_name=self._table_name,
            index_name=index_name
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def list_indexes(self) -> Any:
        """List all indexes."""
        res = self._conn.list_indexes(
            db_name=self._db_name,
            table_name=self._table_name
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def show_columns(self) -> Any:
        """Show table columns."""
        res = self._conn.show_columns(
            db_name=self._db_name,
            table_name=self._table_name
        )
        if res.error_code == ErrorCode.OK:
            return select_res_to_polars(res)
        raise SwarndbException(res.error_code, res.error_msg)

    def show_segments(self) -> Any:
        """Show table segments."""
        res = self._conn.show_segments(
            db_name=self._db_name,
            table_name=self._table_name
        )
        if res.error_code == ErrorCode.OK:
            return select_res_to_polars(res)
        raise SwarndbException(res.error_code, res.error_msg)

    def show_segment(self, segment_id: int) -> Any:
        """Show specific segment details."""
        res = self._conn.show_segment(
            db_name=self._db_name,
            table_name=self._table_name,
            segment_id=segment_id
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def show_blocks(self, segment_id: int) -> Any:
        """Show blocks in a segment."""
        res = self._conn.show_blocks(
            db_name=self._db_name,
            table_name=self._table_name,
            segment_id=segment_id
        )
        if res.error_code == ErrorCode.OK:
            return select_res_to_polars(res)
        raise SwarndbException(res.error_code, res.error_msg)

    def show_block(self, segment_id: int, block_id: int) -> Any:
        """Show specific block details."""
        res = self._conn.show_block(
            db_name=self._db_name,
            table_name=self._table_name,
            segment_id=segment_id,
            block_id=block_id
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def show_block_column(self, segment_id: int, block_id: int, column_id: int) -> Any:
        """Show column details in a specific block."""
        res = self._conn.show_block_column(
            db_name=self._db_name,
            table_name=self._table_name,
            segment_id=segment_id,
            block_id=block_id,
            column_id=column_id
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def insert(self, data: Union[INSERT_DATA, List[INSERT_DATA]]) -> Any:
        """Insert data into table."""
        if isinstance(data, dict):
            data = [data]

        fields = []
        for row in data:
            column_names = []
            parse_exprs = []
            for column_name, value in row.items():
                column_names.append(column_name)
                constant_expression = get_remote_constant_expr_from_python_value(value)
                parse_exprs.append(ttypes.ParsedExpr(
                    type=ttypes.ParsedExprType(constant_expr=constant_expression)
                ))

            fields.append(ttypes.Field(
                column_names=column_names,
                parse_exprs=parse_exprs
            ))

        res = self._conn.insert(
            db_name=self._db_name,
            table_name=self._table_name,
            fields=fields
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    @staticmethod
    def _get_copy_file_type(file_type: str) -> ttypes.CopyFileType:
        """Get copy file type from string."""
        type_map = {
            'csv': ttypes.CopyFileType.CSV,
            'json': ttypes.CopyFileType.JSON,
            'jsonl': ttypes.CopyFileType.JSONL,
            'fvecs': ttypes.CopyFileType.FVECS,
            'csr': ttypes.CopyFileType.CSR,
            'bvecs': ttypes.CopyFileType.BVECS
        }
        try:
            return type_map[file_type.lower()]
        except KeyError:
            raise SwarndbException(
                ErrorCode.IMPORT_FILE_FORMAT_ERROR,
                f"Unrecognized file type: {file_type}"
            )

    def import_data(
        self,
        file_path: str,
        import_options: Optional[Dict] = None
    ) -> Any:
        """Import data from file."""
        options = ttypes.ImportOption(
            has_header=False,
            delimiter=',',
            copy_file_type=ttypes.CopyFileType.CSV
        )

        if import_options:
            for k, v in import_options.items():
                key = k.lower()
                if key == 'file_type':
                    options.copy_file_type = self._get_copy_file_type(v)
                elif key == 'delimiter':
                    if len(v) != 1:
                        raise SwarndbException(
                            ErrorCode.IMPORT_FILE_FORMAT_ERROR,
                            f"Invalid delimiter: {v}"
                        )
                    options.delimiter = v[0]
                elif key == 'header':
                    if not isinstance(v, bool):
                        raise SwarndbException(
                            ErrorCode.IMPORT_FILE_FORMAT_ERROR,
                            "Header must be boolean"
                        )
                    options.has_header = v
                else:
                    raise SwarndbException(
                        ErrorCode.IMPORT_FILE_FORMAT_ERROR,
                        f"Unknown parameter: {k}"
                    )

        res = self._conn.import_data(
            db_name=self._db_name,
            table_name=self._table_name,
            file_name=file_path,
            import_options=options
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def export_data(
        self,
        file_path: str,
        export_options: Optional[Dict] = None,
        columns: Optional[List[str]] = None
    ) -> Any:
        """Export data to file."""
        options = ttypes.ExportOption(
            has_header=False,
            delimiter=',',
            copy_file_type=ttypes.CopyFileType.CSV,
            offset=0,
            limit=0,
            row_limit=0
        )

        if export_options:
            for k, v in export_options.items():
                key = k.lower()
                if key == 'file_type':
                    options.copy_file_type = self._get_copy_file_type(v)
                elif key == 'delimiter':
                    if len(v) != 1:
                        raise SwarndbException(
                            ErrorCode.IMPORT_FILE_FORMAT_ERROR,
                            f"Invalid delimiter: {v}"
                        )
                    options.delimiter = v[0]
                elif key == 'header':
                    if not isinstance(v, bool):
                        raise SwarndbException(
                            ErrorCode.IMPORT_FILE_FORMAT_ERROR,
                            "Header must be boolean"
                        )
                    options.has_header = v
                elif key in ('offset', 'limit', 'row_limit'):
                    if not isinstance(v, int):
                        raise SwarndbException(
                            ErrorCode.IMPORT_FILE_FORMAT_ERROR,
                            f"{key} must be integer"
                        )
                    setattr(options, key, v)
                else:
                    raise SwarndbException(
                        ErrorCode.IMPORT_FILE_FORMAT_ERROR,
                        f"Unknown parameter: {k}"
                    )

        res = self._conn.export_data(
            db_name=self._db_name,
            table_name=self._table_name,
            file_name=file_path,
            export_options=options,
            columns=columns
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def delete(self, cond: Optional[str] = None) -> Any:
        """Delete records from table."""
        where_expr = None if cond is None else traverse_conditions(condition(cond))
        res = self._conn.delete(
            db_name=self._db_name,
            table_name=self._table_name,
            where_expr=where_expr
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    def update(self, cond: str, data: Dict[str, Any]) -> Any:
        """Update records in table."""
        where_expr = traverse_conditions(condition(cond))
        update_expr_array = [
            ttypes.UpdateExpr(
                column_name=column_name,
                value=ttypes.ParsedExpr(
                    type=ttypes.ParsedExprType(
                        constant_expr=get_remote_constant_expr_from_python_value(value)
                    )
                )
            )
            for column_name, value in data.items()
        ]

        res = self._conn.update(
            db_name=self._db_name,
            table_name=self._table_name,
            where_expr=where_expr,
            update_expr_array=update_expr_array
        )
        if res.error_code == ErrorCode.OK:
            return res
        raise SwarndbException(res.error_code, res.error_msg)

    # Vector matching methods
    def match_dense(
        self,
        vector_column_name: str,
        embedding_data: VEC,
        embedding_data_type: str,
        distance_type: str,
        topn: int = DEFAULT_MATCH_VECTOR_TOPN,
        knn_params: Optional[Dict] = None
    ) -> 'RemoteTable':
        """Perform dense vector matching."""
        self.query_builder.match_dense(
            vector_column_name,
            embedding_data,
            embedding_data_type,
            distance_type,
            topn,
            knn_params
        )
        return self

    @deprecated_api("knn is deprecated, please use match_dense instead")
    def knn(self, *args, **kwargs) -> 'RemoteTable':
        """Legacy method for dense vector matching."""
        return self.match_dense(*args, **kwargs)

    @params_type_check
    def match_text(
        self,
        fields: str,
        matching_text: str,
        topn: int,
        extra_options: Optional[Dict] = None
    ) -> 'RemoteTable':
        """Perform text matching."""
        self.query_builder.match_text(fields, matching_text, topn, extra_options)
        return self

    @deprecated_api("match is deprecated, please use match_text instead")
    def match(self, *args, **kwargs) -> 'RemoteTable':
        """Legacy method for text matching."""
        return self.match_text(*args, **kwargs)

    @params_type_check
    def match_tensor(
        self,
        column_name: str,
        query_data: VEC,
        query_data_type: str,
        topn: int,
        extra_option: Optional[Dict] = None
    ) -> 'RemoteTable':
        """Perform tensor matching."""
        self.query_builder.match_tensor(
            column_name,
            query_data,
            query_data_type,
            topn,
            extra_option
        )
        return self

    def match_sparse(
        self,
        vector_column_name: str,
        sparse_data: Union[SparseVector, Dict],
        distance_type: str,
        topn: int,
        opt_params: Optional[Dict] = None
    ) -> 'RemoteTable':
        """Perform sparse vector matching."""
        self.query_builder.match_sparse(
            vector_column_name,
            sparse_data,
            distance_type,
            topn,
            opt_params
        )
        return self

    @params_type_check
    def fusion(
        self,
        method: str,
        topn: int,
        fusion_params: Optional[Dict] = None
    ) -> 'RemoteTable':
        """Perform fusion of multiple search results."""
        self.query_builder.fusion(method, topn, fusion_params)
        return self

    # Query building methods
    def output(self, columns: Optional[List[str]]) -> 'RemoteTable':
        """Set output columns."""
        self.query_builder.output(columns)
        return self

    def highlight(self, columns: Optional[List[str]]) -> 'RemoteTable':
        """Set highlight columns."""
        self.query_builder.highlight(columns)
        return self

    def filter(self, filter_expr: Optional[str]) -> 'RemoteTable':
        """Set filter condition."""
        self.query_builder.filter(filter_expr)
        return self

    def limit(self, limit: Optional[int]) -> 'RemoteTable':
        """Set result limit."""
        self.query_builder.limit(limit)
        return self

    def offset(self, offset: Optional[int]) -> 'RemoteTable':
        """Set result offset."""
        self.query_builder.offset(offset)
        return self

    def group_by(
        self,
        group_by_expr_list: Optional[Union[List[str], str]]
    ) -> 'RemoteTable':
        """Set grouping columns."""
        if group_by_expr_list is not None:
            self.query_builder.group_by(group_by_expr_list)
        return self

    def having(self, having_expr: Optional[str]) -> 'RemoteTable':
        """Set having clause."""
        self.query_builder.having(having_expr)
        return self

    def sort(self, order_by_expr_list: Optional[List[Tuple[str, SortType]]]) -> 'RemoteTable':
        """Set sort order."""
        if order_by_expr_list:
            for order_by_expr in order_by_expr_list:
                if len(order_by_expr) != 2:
                    raise SwarndbException(
                        ErrorCode.INVALID_PARAMETER_VALUE,
                        "Invalid order by expression format"
                    )
                if order_by_expr[1] not in (SortType.Asc, SortType.Desc):
                    raise SwarndbException(
                        ErrorCode.INVALID_PARAMETER_VALUE,
                        "Invalid sort type"
                    )
            self.query_builder.sort(order_by_expr_list)
        return self

    def option(self, option_kv: Dict) -> 'RemoteTable':
        """Set query options."""
        self.query_builder.option(option_kv)
        return self

    # Result methods
    def to_string(self) -> str:
        """Convert query to string representation."""
        return self.query_builder.to_string()

    def to_result(self) -> Any:
        """Execute query and return results."""
        return self.query_builder.to_result()

    def to_df(self) -> Any:
        """Convert results to pandas DataFrame."""
        return self.query_builder.to_df()

    def to_pl(self) -> Any:
        """Convert results to polars DataFrame."""
        return self.query_builder.to_pl()

    def to_arrow(self) -> Any:
        """Convert results to Arrow Table."""
        return self.query_builder.to_arrow()

    def explain(
        self,
        explain_type: ExplainType = ExplainType.Physical
    ) -> Any:
        """Explain query execution plan."""
        return self.query_builder.explain(explain_type)

    def optimize(
        self,
        index_name: str,
        opt_params: Dict[str, str]
    ) -> Any:
        """Optimize table index."""
        opt_options = ttypes.OptimizeOptions(
            index_name=index_name,
            opt_params=[
                ttypes.InitParameter(k, v)
                for k, v in opt_params.items()
            ]
        )
        return self._conn.optimize(
            db_name=self._db_name,
            table_name=self._table_name,
            optimize_opt=opt_options
        )

    def add_columns(self, column_defs: Dict) -> Any:
        """Add columns to table."""
        column_defs_list = []
        for index, (column_name, column_info) in enumerate(column_defs.items()):
            check_valid_name(column_name, "Column")
            get_ordinary_info(column_info, column_defs_list, column_name, index)

        return self._conn.add_columns(
            db_name=self._db_name,
            table_name=self._table_name,
            column_defs=column_defs_list
        )

    def drop_columns(
        self,
        column_names: Union[List[str], str]
    ) -> Any:
        """Drop columns from table."""
        if isinstance(column_names, str):
            column_names = [column_names]

        return self._conn.drop_columns(
            db_name=self._db_name,
            table_name=self._table_name,
            column_names=column_names
        )

    def compact(self) -> Any:
        """Compact table data."""
        return self._conn.compact(
            db_name=self._db_name,
            table_name=self._table_name
        )

    def _to_string(self, query: Query) -> str:
        """Convert query to string representation."""
        res = {
            "db": self._db_name,
            "table": self._table_name
        }

        if query.columns:
            res["columns"] = [
                parsed_expression_to_string(column)
                for column in query.columns
            ]

        if query.highlight:
            res["highlights"] = [
                parsed_expression_to_string(highlight)
                for highlight in query.highlight
            ]

        if query.search:
            res["search"] = search_to_string(query.search)

        if query.filter:
            res["filter"] = parsed_expression_to_string(query.filter)

        if query.limit:
            res["limit"] = parsed_expression_to_string(query.limit)

        if query.offset:
            res["offset"] = parsed_expression_to_string(query.offset)

        return json.dumps(res)

    def _execute_query(
        self,
        query: Query
    ) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
        """Execute query and return results."""
        res = self._conn.select(
            db_name=self._db_name,
            table_name=self._table_name,
            select_list=query.columns,
            highlight_list=query.highlight,
            search_expr=query.search,
            where_expr=query.filter,
            group_by_list=query.groupby,
            having_expr=query.having,
            limit_expr=query.limit,
            offset_expr=query.offset,
            order_by_list=query.sort,
            total_hits_count=query.total_hits_count
        )

        if res.error_code == ErrorCode.OK:
            return build_result(res)
        raise SwarndbException(res.error_code, res.error_msg)

    def _explain_query(self, query: ExplainQuery) -> Any:
        """Explain query execution plan."""
        res = self._conn.explain(
            db_name=self._db_name,
            table_name=self._table_name,
            select_list=query.columns,
            highlight_list=query.highlight,
            search_expr=query.search,
            where_expr=query.filter,
            group_by_list=None,
            limit_expr=query.limit,
            offset_expr=query.offset,
            explain_type=query.explain_type.to_ttype()
        )

        if res.error_code == ErrorCode.OK:
            return select_res_to_polars(res)
        raise SwarndbException(res.error_code, res.error_msg)