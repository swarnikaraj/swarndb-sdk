"""
Query builder for SwarndbDB
"""

from __future__ import annotations

from abc import ABC
from typing import List, Optional, Any, Dict, Tuple, Union
from functools import lru_cache
from enum import Enum

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from pyarrow import Table
from sqlglot import condition, maybe_parse

# Define a simple ExplainType for SwarndbDB
class ExplainType(Enum):
    """Query explain types."""
    Logical = 0
    Physical = 1
    Pipeline = 2

# A simple FunctionExpr class for our purposes (or import from your Thrift definitions)
class FunctionExpr:
    __slots__ = ('function_name', 'arguments')
    def __init__(self, function_name: str, arguments: List[Any]):
        self.function_name = function_name
        self.arguments = arguments
    def __repr__(self):
        return f"FunctionExpr({self.function_name}, {self.arguments})"

# Import Swarndb-specific types and exceptions
from swarndb.common import VEC, SparseVector, SwarndbException, SortType
from swarndb.errors import ErrorCode
from swarndb.remote_thrift.swarndb_thrift_rpc.ttypes import (
    ParsedExpr, SearchExpr, ParsedExprType, ColumnExpr, 
    ConstantExpr, LiteralType, OrderByExpr, KnnExpr, ElementType,
    KnnDistanceType, EmbeddingData, InitParameter, GenericMatchExpr,
    MatchExpr, FusionExpr
)
from swarndb.remote_thrift.types import (
    logic_type_to_dtype,
    make_match_tensor_expr,
    make_match_sparse_expr,
)
from swarndb.remote_thrift.utils import (
    traverse_conditions,
    parse_expr,
    get_search_optional_filter_from_opt_params
)

# Base Query classes
class Query(ABC):
    """Base query class."""
    __slots__ = (
        'columns', 'highlight', 'search', 'filter', 'groupby',
        'having', 'limit', 'offset', 'sort', 'total_hits_count'
    )

    def __init__(
        self,
        columns: Optional[List[ParsedExpr]] = None,
        highlight: Optional[List[ParsedExpr]] = None,
        search: Optional[SearchExpr] = None,
        filter: Optional[ParsedExpr] = None,
        groupby: Optional[List[ParsedExpr]] = None,
        having: Optional[ParsedExpr] = None,
        limit: Optional[ParsedExpr] = None,
        offset: Optional[ParsedExpr] = None,
        sort: Optional[List[OrderByExpr]] = None,
        total_hits_count: Optional[bool] = None
    ):
        self.columns = columns
        self.highlight = highlight
        self.search = search
        self.filter = filter
        self.groupby = groupby
        self.having = having
        self.limit = limit
        self.offset = offset
        self.sort = sort
        self.total_hits_count = total_hits_count

class ExplainQuery(Query):
    """Query with explain capabilities."""
    __slots__ = ('explain_type',)

    def __init__(
        self,
        columns: Optional[List[ParsedExpr]] = None,
        highlight: Optional[List[ParsedExpr]] = None,
        search: Optional[SearchExpr] = None,
        filter: Optional[ParsedExpr] = None,
        groupby: Optional[List[ParsedExpr]] = None,
        having: Optional[ParsedExpr] = None,
        limit: Optional[ParsedExpr] = None,
        offset: Optional[ParsedExpr] = None,
        sort: Optional[List[OrderByExpr]] = None,
        explain_type: Optional[ExplainType] = None
    ):
        super().__init__(columns, highlight, search, filter, groupby, having, limit, offset, sort, False)
        self.explain_type = explain_type

# SwarndbThriftQueryBuilder implementation
class SwarndbThriftQueryBuilder(ABC):
    """Query builder for SwarndbDB using Thrift protocol."""
    __slots__ = (
        '_table', '_columns', '_highlight', '_search', '_filter',
        '_groupby', '_having', '_limit', '_offset', '_sort',
        '_total_hits_count'
    )

    # Cache frequently used expressions
    _FUNCTION_EXPR_CACHE = {}
    _SPECIAL_COLUMNS = {
        '*': lambda: ColumnExpr(star=True, column_name=[]),
        '_row_id': lambda: FunctionExpr(function_name="row_id", arguments=[]),
        '_create_timestamp': lambda: FunctionExpr(function_name="create_timestamp", arguments=[]),
        '_delete_timestamp': lambda: FunctionExpr(function_name="delete_timestamp", arguments=[]),
        '_score': lambda: FunctionExpr(function_name="score", arguments=[]),
        '_similarity': lambda: FunctionExpr(function_name="similarity", arguments=[]),
        '_distance': lambda: FunctionExpr(function_name="distance", arguments=[]),
        '_score_factors': lambda: FunctionExpr(function_name="score_factors", arguments=[]),
        '_similarity_factors': lambda: FunctionExpr(function_name="similarity_factors", arguments=[]),
        '_distance_factors': lambda: FunctionExpr(function_name="distance_factors", arguments=[])
    }

    def __init__(self, table: Any):
        self._table = table
        self.reset()

    def reset(self) -> None:
        """Reset all query parameters."""
        self._columns = None
        self._highlight = None
        self._search = None
        self._filter = None
        self._groupby = None
        self._having = None
        self._limit = None
        self._offset = None
        self._sort = None
        self._total_hits_count = None

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_function_expr(name: str) -> FunctionExpr:
        """Get cached function expression."""
        return FunctionExpr(function_name=name, arguments=[])

    def _create_parsed_expr(self, expr_type: ParsedExprType) -> ParsedExpr:
        """Create a parsed expression."""
        return ParsedExpr(type=expr_type)

    def _process_embedding_data(
        self,
        embedding_data: VEC,
        embedding_data_type: str
    ) -> Tuple[List[Union[int, float]], ElementType]:
        """Process embedding data for vector search."""
        if isinstance(embedding_data, (list, tuple)):
            data = list(embedding_data)
        elif isinstance(embedding_data, np.ndarray):
            data = embedding_data.tolist()
        else:
            raise SwarndbException(
                ErrorCode.INVALID_DATA_TYPE,
                f"Invalid embedding data type: {type(embedding_data)}"
            )

        if embedding_data_type == "bit":
            if len(data) % 8 != 0:
                raise SwarndbException(
                    ErrorCode.INVALID_EMBEDDING_DATA_TYPE,
                    "Bit embeddings must have dimension multiple of 8"
                )
            data = [
                sum((data[i * 8 + j] > 0) << j for j in range(8))
                for i in range(len(data) // 8)
            ]

        if embedding_data_type in ("uint8", "int8", "int16", "int32", "int", "int64"):
            data = [int(x) for x in data]
        elif embedding_data_type in ("float", "float32", "double", "float64", "float16", "bfloat16"):
            data = [float(x) for x in data]

        elem_type = self._get_element_type(embedding_data_type)
        return data, elem_type

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_element_type(data_type: str) -> ElementType:
        type_map = {
            "bit": ElementType.ElementBit,
            "uint8": ElementType.ElementUInt8,
            "int8": ElementType.ElementInt8,
            "int16": ElementType.ElementInt16,
            "int": ElementType.ElementInt32,
            "int32": ElementType.ElementInt32,
            "int64": ElementType.ElementInt64,
            "float": ElementType.ElementFloat32,
            "float32": ElementType.ElementFloat32,
            "double": ElementType.ElementFloat64,
            "float64": ElementType.ElementFloat64,
            "float16": ElementType.ElementFloat16,
            "bfloat16": ElementType.ElementBFloat16
        }
        try:
            return type_map[data_type]
        except KeyError:
            raise SwarndbException(
                ErrorCode.INVALID_EMBEDDING_DATA_TYPE,
                f"Invalid embedding data type: {data_type}"
            )

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_distance_type(distance_type: str) -> KnnDistanceType:
        type_map = {
            "l2": KnnDistanceType.L2,
            "cosine": KnnDistanceType.Cosine,
            "cos": KnnDistanceType.Cosine,
            "ip": KnnDistanceType.InnerProduct,
            "hamming": KnnDistanceType.Hamming
        }
        try:
            return type_map[distance_type.lower()]
        except KeyError:
            raise SwarndbException(
                ErrorCode.INVALID_KNN_DISTANCE_TYPE,
                f"Invalid distance type: {distance_type}"
            )

    def match_dense(
        self,
        vector_column_name: str,
        embedding_data: VEC,
        embedding_data_type: str,
        distance_type: str,
        topn: int,
        knn_params: Optional[Dict[str, str]] = None
    ) -> SwarndbThriftQueryBuilder:
        if not isinstance(topn, int):
            raise SwarndbException(
                ErrorCode.INVALID_TOPK_TYPE,
                f"topn must be integer, got {type(topn)}"
            )
        if self._search is None:
            self._search = SearchExpr()
            self._search.match_exprs = []
        column_expr = ColumnExpr(column_name=[vector_column_name], star=False)
        data_list, elem_type = self._process_embedding_data(embedding_data, embedding_data_type)
        data = EmbeddingData()
        self._set_embedding_data(data, elem_type, data_list)
        dist_type = self._get_distance_type(distance_type)
        knn_opt_params = []
        optional_filter = None
        if knn_params:
            optional_filter = get_search_optional_filter_from_opt_params(knn_params)
            knn_opt_params = [InitParameter(k.lower(), str(v).lower()) for k, v in knn_params.items()]
        knn_expr = KnnExpr(
            column_expr=column_expr,
            embedding_data=data,
            embedding_data_type=elem_type,
            distance_type=dist_type,
            topn=topn,
            opt_params=knn_opt_params,
            filter_expr=optional_filter,
        )
        self._search.match_exprs.append(GenericMatchExpr(match_vector_expr=knn_expr))
        return self

    @staticmethod
    def _set_embedding_data(
        data: EmbeddingData,
        elem_type: ElementType,
        values: List[Union[int, float]]
    ) -> None:
        type_attr_map = {
            ElementType.ElementBit: 'u8_array_value',
            ElementType.ElementUInt8: 'u8_array_value',
            ElementType.ElementInt8: 'i8_array_value',
            ElementType.ElementInt16: 'i16_array_value',
            ElementType.ElementInt32: 'i32_array_value',
            ElementType.ElementInt64: 'i64_array_value',
            ElementType.ElementFloat32: 'f32_array_value',
            ElementType.ElementFloat64: 'f64_array_value',
            ElementType.ElementFloat16: 'f16_array_value',
            ElementType.ElementBFloat16: 'bf16_array_value'
        }
        setattr(data, type_attr_map[elem_type], values)

    def match_sparse(
        self,
        vector_column_name: str,
        sparse_data: Union[SparseVector, Dict],
        metric_type: str,
        topn: int,
        opt_params: Optional[Dict] = None,
    ) -> SwarndbThriftQueryBuilder:
        if self._search is None:
            self._search = SearchExpr()
            self._search.match_exprs = []
        optional_filter = None if opt_params is None else get_search_optional_filter_from_opt_params(opt_params)
        match_sparse_expr = make_match_sparse_expr(vector_column_name, sparse_data, metric_type, topn, opt_params, optional_filter)
        self._search.match_exprs.append(GenericMatchExpr(match_sparse_expr=match_sparse_expr))
        return self

    def match_text(
        self,
        fields: str,
        matching_text: str,
        topn: int,
        extra_options: Optional[Dict] = None
    ) -> SwarndbThriftQueryBuilder:
        if self._search is None:
            self._search = SearchExpr()
            self._search.match_exprs = []
        match_expr = MatchExpr(
            fields=fields,
            matching_text=matching_text,
            options_text=self._build_options_text(topn, extra_options),
            filter_expr=(get_search_optional_filter_from_opt_params(extra_options) if extra_options else None)
        )
        self._search.match_exprs.append(GenericMatchExpr(match_text_expr=match_expr))
        return self

    @staticmethod
    def _build_options_text(topn: int, options: Optional[Dict] = None) -> str:
        parts = [f"topn={topn}"]
        if options:
            parts.extend(f"{k}={v}" for k, v in options.items())
        return ";".join(parts)

    def match_tensor(
        self,
        column_name: str,
        query_data: VEC,
        query_data_type: str,
        topn: int,
        extra_option: Optional[Dict] = None,
    ) -> SwarndbThriftQueryBuilder:
        if self._search is None:
            self._search = SearchExpr()
            self._search.match_exprs = []
        option_str = self._build_options_text(topn, extra_option)
        optional_filter = (get_search_optional_filter_from_opt_params(extra_option) if extra_option else None)
        match_tensor_expr = make_match_tensor_expr(
            vector_column_name=column_name,
            embedding_data=query_data,
            embedding_data_type=query_data_type,
            method_type="maxsim",
            extra_option=option_str,
            filter_expr=optional_filter,
        )
        self._search.match_exprs.append(GenericMatchExpr(match_tensor_expr=match_tensor_expr))
        return self

    def fusion(
        self,
        method: str,
        topn: int,
        fusion_params: Optional[Dict] = None
    ) -> SwarndbThriftQueryBuilder:
        if self._search is None:
            self._search = SearchExpr()
        if self._search.fusion_exprs is None:
            self._search.fusion_exprs = []
        fusion_expr = FusionExpr(method=method)
        final_option_text = f"topn={topn}"
        if method in ("rrf", "weighted_sum"):
            if isinstance(fusion_params, dict):
                for k, v in fusion_params.items():
                    if k == "topn":
                        raise SwarndbException(ErrorCode.INVALID_EXPRESSION, "topn is not allowed in fusion params")
                    final_option_text += f";{k}={v}"
        elif method == "match_tensor":
            fusion_expr.optional_match_tensor_expr = make_match_tensor_expr(
                vector_column_name=fusion_params["field"],
                embedding_data=fusion_params["query_tensor"],
                embedding_data_type=fusion_params["element_type"],
                method_type="maxsim",
                extra_option=None
            )
        else:
            raise SwarndbException(ErrorCode.INVALID_EXPRESSION, "Invalid fusion method")
        fusion_expr.options_text = final_option_text
        self._search.fusion_exprs.append(fusion_expr)
        return self

    def filter(self, where: Optional[str]) -> SwarndbThriftQueryBuilder:
        if where:
            self._filter = traverse_conditions(condition(where))
        return self

    def limit(self, limit: Optional[int]) -> SwarndbThriftQueryBuilder:
        if limit is not None:
            self._limit = self._create_constant_expr(limit)
        return self

    def offset(self, offset: Optional[int]) -> SwarndbThriftQueryBuilder:
        if offset is not None:
            self._offset = self._create_constant_expr(offset)
        return self

    def _create_constant_expr(self, value: int) -> ParsedExpr:
        return ParsedExpr(
            type=ParsedExprType(
                constant_expr=ConstantExpr(
                    literal_type=LiteralType.Int64,
                    i64_value=value
                )
            )
        )

    def group_by(self, columns: Union[List[str], str]) -> SwarndbThriftQueryBuilder:
        if isinstance(columns, str):
            columns = [columns]
        self._groupby = [parse_expr(maybe_parse(col.lower())) for col in columns]
        return self

    def having(self, having: Optional[str]) -> SwarndbThriftQueryBuilder:
        if having:
            self._having = traverse_conditions(condition(having))
        return self

    def output(self, columns: Optional[List[str]]) -> SwarndbThriftQueryBuilder:
        if not columns:
            return self
        select_list: List[ParsedExpr] = []
        for column in columns:
            if isinstance(column, str):
                column = column.lower()
                match column:
                    case "*":
                        column_expr = ColumnExpr(star=True, column_name=[])
                        expr_type = ParsedExprType(column_expr=column_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case "_row_id":
                        func_expr = FunctionExpr(function_name="row_id", arguments=[])
                        expr_type = ParsedExprType(function_expr=func_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case "_create_timestamp":
                        func_expr = FunctionExpr(function_name="create_timestamp", arguments=[])
                        expr_type = ParsedExprType(function_expr=func_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case "_delete_timestamp":
                        func_expr = FunctionExpr(function_name="delete_timestamp", arguments=[])
                        expr_type = ParsedExprType(function_expr=func_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case "_score":
                        func_expr = FunctionExpr(function_name="score", arguments=[])
                        expr_type = ParsedExprType(function_expr=func_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case "_similarity":
                        func_expr = FunctionExpr(function_name="similarity", arguments=[])
                        expr_type = ParsedExprType(function_expr=func_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case "_distance":
                        func_expr = FunctionExpr(function_name="distance", arguments=[])
                        expr_type = ParsedExprType(function_expr=func_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case "_score_factors":
                        func_expr = FunctionExpr(function_name="score_factors", arguments=[])
                        expr_type = ParsedExprType(function_expr=func_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case "_similarity_factors":
                        func_expr = FunctionExpr(function_name="similarity_factors", arguments=[])
                        expr_type = ParsedExprType(function_expr=func_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case "_distance_factors":
                        func_expr = FunctionExpr(function_name="distance_factors", arguments=[])
                        expr_type = ParsedExprType(function_expr=func_expr)
                        parsed_expr = ParsedExpr(type=expr_type)
                        select_list.append(parsed_expr)
                    case _:
                        select_list.append(parse_expr(maybe_parse(column)))
        self._columns = select_list
        return self

    def highlight(self, columns: Optional[List[str]]) -> SwarndbThriftQueryBuilder:
        if columns:
            self._highlight = [parse_expr(maybe_parse(col.lower())) for col in columns if isinstance(col, str)]
        return self

    def option(self, option_kv: Dict) -> SwarndbThriftQueryBuilder:
        if 'total_hits_count' in option_kv:
            self._total_hits_count = bool(option_kv['total_hits_count'])
        return self

    def sort(self, order_by_expr_list: Optional[List[List[Union[str, SortType]]]]) -> SwarndbThriftQueryBuilder:
        if not order_by_expr_list:
            return self
        sort_list: List[OrderByExpr] = []
        for order_by_expr in order_by_expr_list:
            order_by_expr_str = order_by_expr[0].lower() if isinstance(order_by_expr[0], str) else str(order_by_expr[0])
            match order_by_expr_str:
                case "*":
                    column_expr = ColumnExpr(star=True, column_name=[])
                    expr_type = ParsedExprType(column_expr=column_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case "_row_id":
                    func_expr = FunctionExpr(function_name="row_id", arguments=[])
                    expr_type = ParsedExprType(function_expr=func_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case "_create_timestamp":
                    func_expr = FunctionExpr(function_name="create_timestamp", arguments=[])
                    expr_type = ParsedExprType(function_expr=func_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case "_delete_timestamp":
                    func_expr = FunctionExpr(function_name="delete_timestamp", arguments=[])
                    expr_type = ParsedExprType(function_expr=func_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case "_score":
                    func_expr = FunctionExpr(function_name="score", arguments=[])
                    expr_type = ParsedExprType(function_expr=func_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case "_similarity":
                    func_expr = FunctionExpr(function_name="similarity", arguments=[])
                    expr_type = ParsedExprType(function_expr=func_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case "_distance":
                    func_expr = FunctionExpr(function_name="distance", arguments=[])
                    expr_type = ParsedExprType(function_expr=func_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case "_score_factors":
                    func_expr = FunctionExpr(function_name="score_factors", arguments=[])
                    expr_type = ParsedExprType(function_expr=func_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case "_similarity_factors":
                    func_expr = FunctionExpr(function_name="similarity_factors", arguments=[])
                    expr_type = ParsedExprType(function_expr=func_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case "_distance_factors":
                    func_expr = FunctionExpr(function_name="distance_factors", arguments=[])
                    expr_type = ParsedExprType(function_expr=func_expr)
                    parsed_expr = ParsedExpr(type=expr_type)
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
                case _:
                    parsed_expr = parse_expr(maybe_parse(order_by_expr_str))
                    order_by_flag: bool = order_by_expr[1] == SortType.Asc
                    sort_list.append(OrderByExpr(expr=parsed_expr, asc=order_by_flag))
        self._sort = sort_list
        return self

    def to_string(self) -> str:
        query = Query(
            columns=self._columns,
            highlight=self._highlight,
            search=self._search,
            filter=self._filter,
            groupby=self._groupby,
            limit=self._limit,
            offset=self._offset,
            sort=self._sort,
            total_hits_count=self._total_hits_count,
        )
        return self._table._to_string(query)

    def to_result(self) -> Tuple[Dict[str, List[Any]], Dict[str, Any], Dict]:
        query = Query(
            columns=self._columns,
            highlight=self._highlight,
            search=self._search,
            filter=self._filter,
            groupby=self._groupby,
            having=self._having,
            limit=self._limit,
            offset=self._offset,
            sort=self._sort,
            total_hits_count=self._total_hits_count,
        )
        self.reset()
        return self._table._execute_query(query)

    def to_df(self) -> Tuple[pd.DataFrame, Dict]:
        data_dict, data_type_dict, extra_result = self.to_result()
        df_dict = {
            k: pd.Series(v, dtype=logic_type_to_dtype(data_type_dict[k]))
            for k, v in data_dict.items()
        }
        return pd.DataFrame(df_dict), extra_result

    def to_pl(self) -> Tuple[pl.DataFrame, Dict]:
        dataframe, extra_result = self.to_df()
        return pl.from_pandas(dataframe), extra_result

    def to_arrow(self) -> Tuple[Table, Dict]:
        dataframe, extra_result = self.to_df()
        return pa.Table.from_pandas(dataframe), extra_result

    def explain(self, explain_type: ExplainType = ExplainType.Physical) -> Any:
        query = ExplainQuery(
            columns=self._columns,
            highlight=self._highlight,
            search=self._search,
            filter=self._filter,
            groupby=self._groupby,
            having=self._having,
            limit=self._limit,
            offset=self._offset,
            sort=self._sort,
            explain_type=explain_type,
        )
        return self._table._explain_query(query)
