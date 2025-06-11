
import struct
import json
import numpy as np
from swarndb.common import VEC, SparseVector, SwarndbException
from swarndb.remote_thrift.swarndb_thrift_rpc.ttypes import *
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import date, time, datetime, timedelta

import polars as pl
from numpy import dtype
from swarndb.errors import ErrorCode

import swarndb.remote_thrift.swarndb_thrift_rpc.ttypes as ttypes


def logic_type_to_dtype(ttype: ttypes.DataType):
    match ttype.logic_type:
        case ttypes.LogicType.Boolean:
            return dtype('bool')
        case ttypes.LogicType.TinyInt:
            return dtype('int8')
        case ttypes.LogicType.SmallInt:
            return dtype('int16')
        case ttypes.LogicType.Integer:
            return dtype('int32')
        case ttypes.LogicType.BigInt:
            return dtype('int64')
        case ttypes.LogicType.Float:
            return dtype('float32')
        case ttypes.LogicType.Double:
            return dtype('float64')
        case ttypes.LogicType.Float16:
            return dtype('float32')
        case ttypes.LogicType.BFloat16:
            return dtype('float32')
        case ttypes.LogicType.Varchar:
            return dtype('str')
        case ttypes.LogicType.Embedding:
            return object
        case ttypes.LogicType.MultiVector:
            return object
        case ttypes.LogicType.Tensor:
            return object
        case ttypes.LogicType.TensorArray:
            return object
        case ttypes.LogicType.Sparse:
            return object
        case ttypes.LogicType.Date:
            return object
        case ttypes.LogicType.Time:
            return object
        case ttypes.LogicType.DateTime:
            return object
        case ttypes.LogicType.Interval:
            return object
        case ttypes.LogicType.Timestamp:
            return object
        case ttypes.LogicType.Array:
            return object
        case _:
            raise NotImplementedError(f"Unsupported type {ttype}")


def bf16_bytes_to_float32_list(column_vector):
    tmp_u16 = np.frombuffer(column_vector, dtype='<i2')
    result_arr = np.zeros(2 * len(tmp_u16), dtype='<i2')
    result_arr[1::2] = tmp_u16
    view_float32 = result_arr.view('<f4')
    return list(view_float32)


def parse_bytes(bytes_data):
    results = []
    offset = 0
    while offset < len(bytes_data):
        string_data, offset = parse_single_str_bytes(bytes_data, offset)
        results.append(string_data)
    return results


def parse_single_str_bytes(bytes_data, offset):
    length = struct.unpack('I', bytes_data[offset:offset + 4])[0]
    offset += 4
    string_data = bytes_data[offset:offset + length].decode('utf-8')
    offset += length
    return string_data, offset


def parse_date_bytes(column_vector):
    parsed_list = list(struct.unpack('<{}i'.format(len(column_vector) // 4), column_vector))
    date_list = []
    epoch = date(1970, 1, 1)
    for value in parsed_list:
        date_list.append((epoch + timedelta(days=value)).strftime('%Y-%m-%d'))
    return date_list


def parse_time_bytes(column_vector):
    parsed_list = list(struct.unpack('<{}i'.format(len(column_vector) // 4), column_vector))
    time_list = []
    for value in parsed_list:
        hours = (value // 3600) % 24
        minutes = (value % 3600) // 60
        seconds = value % 60
        time_list.append(time(hour=hours, minute=minutes, second=seconds).strftime('%H:%M:%S'))
    return time_list


def parse_datetime_bytes(column_vector):
    parsed_list = list(struct.unpack('<{}i'.format(len(column_vector) // 4), column_vector))
    datetime_list = []
    epoch = datetime(1970, 1, 1)
    for i in range(0, len(parsed_list), 2):
        if i + 1 < len(parsed_list):
            datetime_list.append((epoch + timedelta(days=parsed_list[i], seconds=parsed_list[i + 1])).strftime('%Y-%m-%d %H:%M:%S'))
    return datetime_list


def parse_interval_bytes(column_vector):
    parsed_list = list(struct.unpack('<{}i'.format(len(column_vector) // 4), column_vector))
    interval_list = []
    for value in parsed_list:
        interval_list.append(str(timedelta(seconds=value).total_seconds()) + 's')
    return interval_list


def column_vector_to_list(column_type: ttypes.ColumnType, column_data_type: ttypes.DataType, column_vectors) -> list:
    column_vector = b''.join(column_vectors)
    match column_type:
        case ttypes.ColumnType.ColumnInt32:
            return list(struct.unpack('<{}i'.format(len(column_vector) // 4), column_vector))
        case ttypes.ColumnType.ColumnInt64:
            return list(struct.unpack('<{}q'.format(len(column_vector) // 8), column_vector))
        case ttypes.ColumnType.ColumnFloat32:
            return list(struct.unpack('<{}f'.format(len(column_vector) // 4), column_vector))
        case ttypes.ColumnType.ColumnFloat64:
            return list(struct.unpack('<{}d'.format(len(column_vector) // 8), column_vector))
        case ttypes.ColumnType.ColumnFloat16:
            return list(struct.unpack('<{}e'.format(len(column_vector) // 2), column_vector))
        case ttypes.ColumnType.ColumnBFloat16:
            return bf16_bytes_to_float32_list(column_vector)
        case ttypes.ColumnType.ColumnVarchar:
            return list(parse_bytes(column_vector))
        case ttypes.ColumnType.ColumnBool:
            return list(struct.unpack('<{}?'.format(len(column_vector)), column_vector))
        case ttypes.ColumnType.ColumnInt8:
            return list(struct.unpack('<{}b'.format(len(column_vector)), column_vector))
        case ttypes.ColumnType.ColumnInt16:
            return list(struct.unpack('<{}h'.format(len(column_vector) // 2), column_vector))
        case ttypes.ColumnType.ColumnRowID:
            return list(struct.unpack('<{}q'.format(len(column_vector) // 8), column_vector))
        case ttypes.ColumnType.ColumnEmbedding:
            dimension = column_data_type.physical_type.embedding_type.dimension
            if column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementUInt8:
                all_list = list(struct.unpack('<{}B'.format(len(column_vector)), column_vector))
                return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
            elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementInt8:
                all_list = list(struct.unpack('<{}b'.format(len(column_vector)), column_vector))
                return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
            elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementInt16:
                all_list = list(struct.unpack('<{}h'.format(len(column_vector) // 2), column_vector))
                return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
            elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementInt32:
                all_list = list(struct.unpack('<{}i'.format(len(column_vector) // 4), column_vector))
                return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
            elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementInt64:
                all_list = list(struct.unpack('<{}q'.format(len(column_vector) // 8), column_vector))
                return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
            elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementFloat32:
                all_list = list(struct.unpack('<{}f'.format(len(column_vector) // 4), column_vector))
                return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
            elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementFloat64:
                all_list = list(struct.unpack('<{}d'.format(len(column_vector) // 8), column_vector))
                return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
            elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementFloat16:
                all_list = list(struct.unpack('<{}e'.format(len(column_vector) // 2), column_vector))
                return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
            elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementBFloat16:
                all_list = bf16_bytes_to_float32_list(column_vector)
                return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
            elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementBit:
                all_list = list(struct.unpack('<{}B'.format(len(column_vector)), column_vector))
                result = []
                if dimension % 8 != 0:
                    raise ValueError(f"Unsupported dimension {dimension}")
                sub_dim: int = dimension // 8
                for i in range(0, len(all_list), sub_dim):
                    mid_res = all_list[i:i + sub_dim]
                    mid_res_int: int = 0
                    for j in reversed(mid_res):
                        mid_res_int = mid_res_int * 256 + j
                    # Format the integer as a binary string of fixed width, then reverse it.
                    result.append([f"{mid_res_int:0{dimension}b}"[::-1]])
                return result
            else:
                raise NotImplementedError(f"Unsupported type {column_data_type.physical_type.embedding_type.element_type}")
        case ttypes.ColumnType.ColumnMultiVector:
            return parse_tensor_bytes(column_data_type, column_vector)
        case ttypes.ColumnType.ColumnTensor:
            return parse_tensor_bytes(column_data_type, column_vector)
        case ttypes.ColumnType.ColumnTensorArray:
            return parse_tensorarray_bytes(column_data_type, column_vector)
        case ttypes.ColumnType.ColumnSparse:
            return parse_sparse_bytes(column_data_type, column_vector)
        case ttypes.ColumnType.ColumnDate:
            return parse_date_bytes(column_vector)
        case ttypes.ColumnType.ColumnTime:
            return parse_time_bytes(column_vector)
        case ttypes.ColumnType.ColumnDateTime:
            return parse_datetime_bytes(column_vector)
        case ttypes.ColumnType.ColumnTimestamp:
            return parse_datetime_bytes(column_vector)
        case ttypes.ColumnType.ColumnInterval:
            return parse_interval_bytes(column_vector)
        case ttypes.ColumnType.ColumnArray:
            return parse_array_bytes(column_data_type, column_vector)
        case _:
            raise NotImplementedError(f"Unsupported type {column_type}")

def parse_array_bytes(column_data_type: ttypes.DataType, bytes_data) -> list:
    results = []
    offset = 0
    while offset < len(bytes_data):
        array_data, offset = parse_single_array_bytes(column_data_type, bytes_data, offset)
        results.append(array_data)
    return results

def parse_tensor_bytes(column_data_type: ttypes.DataType, bytes_data) -> list:
    results = []
    offset = 0
    while offset < len(bytes_data):
        tensor_data, offset = parse_single_tensor_bytes(column_data_type, bytes_data, offset)
        results.append(tensor_data)
    return results

def parse_single_tensor_bytes(column_data_type: ttypes.DataType, bytes_data, offset) -> Tuple[Any, int]:
    length = struct.unpack('I', bytes_data[offset:offset + 4])[0]
    offset += 4
    tensor_data = tensor_to_list(column_data_type, bytes_data[offset:offset + length])
    offset += length
    return tensor_data, offset

def parse_tensorarray_bytes(column_data_type: ttypes.DataType, bytes_data) -> list:
    results = []
    offset = 0
    while offset < len(bytes_data):
        tensorarray_data, offset = parse_single_tensorarray_bytes(column_data_type, bytes_data, offset)
        results.append(tensorarray_data)
    return results

def parse_single_tensorarray_bytes(column_data_type: ttypes.DataType, bytes_data, offset) -> Tuple[Any, int]:
    tensorarray_data = []
    tensor_n = struct.unpack('I', bytes_data[offset:offset + 4])[0]
    offset += 4
    for _ in range(tensor_n):
        tensor_data, offset = parse_single_tensor_bytes(column_data_type, bytes_data, offset)
        tensorarray_data.append(tensor_data)
    return tensorarray_data, offset

def tensor_to_list(column_data_type: ttypes.DataType, binary_data) -> list:
    dimension = column_data_type.physical_type.embedding_type.dimension
    if column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementBit:
        all_list = list(struct.unpack('<{}B'.format(len(binary_data)), binary_data))
        result = []
        if dimension % 8 != 0:
            raise ValueError(f"Unsupported dimension {dimension}")
        sub_dim: int = dimension // 8
        for i in range(0, len(all_list), sub_dim):
            mid_res = all_list[i:i + sub_dim]
            mid_res_int: int = 0
            for j in reversed(mid_res):
                mid_res_int = mid_res_int * 256 + j
            result.append([f"{mid_res_int:0{dimension}b}"[::-1]])
        return result
    elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementUInt8:
        all_list = list(struct.unpack('<{}B'.format(len(binary_data)), binary_data))
        return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
    elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementInt8:
        all_list = list(struct.unpack('<{}b'.format(len(binary_data)), binary_data))
        return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
    elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementInt16:
        all_list = list(struct.unpack('<{}h'.format(len(binary_data) // 2), binary_data))
        return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
    elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementInt32:
        all_list = list(struct.unpack('<{}i'.format(len(binary_data) // 4), binary_data))
        return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
    elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementInt64:
        all_list = list(struct.unpack('<{}q'.format(len(binary_data) // 8), binary_data))
        return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
    elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementFloat32:
        all_list = list(struct.unpack('<{}f'.format(len(binary_data) // 4), binary_data))
        return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
    elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementFloat64:
        all_list = list(struct.unpack('<{}d'.format(len(binary_data) // 8), binary_data))
        return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
    elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementFloat16:
        all_list = list(struct.unpack('<{}e'.format(len(binary_data) // 2), binary_data))
        return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
    elif column_data_type.physical_type.embedding_type.element_type == ttypes.ElementType.ElementBFloat16:
        all_list = bf16_bytes_to_float32_list(binary_data)
        return [all_list[i:i + dimension] for i in range(0, len(all_list), dimension)]
    else:
        raise NotImplementedError(f"Unsupported type {column_data_type.physical_type.embedding_type.element_type}")

def parse_sparse_bytes(column_data_type: ttypes.DataType, column_vector) -> list:
    res = []
    offset = 0
    while offset < len(column_vector):
        sparse_vector, offset = parse_single_sparse_bytes(column_data_type, column_vector, offset)
        res.append(sparse_vector)
    return res

def parse_single_sparse_bytes(column_data_type: ttypes.DataType, column_vector, offset) -> Tuple[Any, int]:
    dimension = column_data_type.physical_type.sparse_type.dimension
    element_type = column_data_type.physical_type.sparse_type.element_type
    index_type = column_data_type.physical_type.sparse_type.index_type
    nnz = struct.unpack('I', column_vector[offset:offset + 4])[0]
    offset += 4
    match index_type:
        case ttypes.ElementType.ElementInt8:
            indices = struct.unpack('<{}b'.format(nnz), column_vector[offset:offset + nnz])
            offset += nnz
        case ttypes.ElementType.ElementInt16:
            indices = struct.unpack('<{}h'.format(nnz), column_vector[offset:offset + nnz * 2])
            offset += nnz * 2
        case ttypes.ElementType.ElementInt32:
            indices = struct.unpack('<{}i'.format(nnz), column_vector[offset:offset + nnz * 4])
            offset += nnz * 4
        case ttypes.ElementType.ElementInt64:
            indices = struct.unpack('<{}q'.format(nnz), column_vector[offset:offset + nnz * 8])
            offset += nnz * 8
        case _:
            raise NotImplementedError(f"Unsupported type {index_type}")
    match element_type:
        case ttypes.ElementType.ElementUInt8:
            values = struct.unpack('<{}B'.format(nnz), column_vector[offset:offset + nnz])
            offset += nnz
        case ttypes.ElementType.ElementInt8:
            values = struct.unpack('<{}b'.format(nnz), column_vector[offset:offset + nnz])
            offset += nnz
        case ttypes.ElementType.ElementInt16:
            values = struct.unpack('<{}h'.format(nnz), column_vector[offset:offset + nnz * 2])
            offset += nnz * 2
        case ttypes.ElementType.ElementInt32:
            values = struct.unpack('<{}i'.format(nnz), column_vector[offset:offset + nnz * 4])
            offset += nnz * 4
        case ttypes.ElementType.ElementInt64:
            values = struct.unpack('<{}q'.format(nnz), column_vector[offset:offset + nnz * 8])
            offset += nnz * 8
        case ttypes.ElementType.ElementFloat32:
            values = struct.unpack('<{}f'.format(nnz), column_vector[offset:offset + nnz * 4])
            offset += nnz * 4
        case ttypes.ElementType.ElementFloat64:
            values = struct.unpack('<{}d'.format(nnz), column_vector[offset:offset + nnz * 8])
            offset += nnz * 8
        case ttypes.ElementType.ElementFloat16:
            values = struct.unpack('<{}e'.format(nnz), column_vector[offset:offset + nnz * 2])
            offset += nnz * 2
        case ttypes.ElementType.ElementBFloat16:
            values = bf16_bytes_to_float32_list(column_vector[offset:offset + nnz * 2])
            offset += nnz * 2
        case ttypes.ElementType.ElementBit:
            raise NotImplementedError(f"Unsupported type {element_type}")
        case _:
            raise NotImplementedError(f"Unsupported type {element_type}")
    return SparseVector(list(indices), list(values)).to_dict(), offset

def find_data_type(column_name: str, column_defs: List[ttypes.ColumnDef]) -> ttypes.DataType:
    for column_def in column_defs:
        if column_def.name == column_name:
            return column_def.data_type
    raise KeyError(f"column name {column_name} not found in column defs")

def build_result(res: ttypes.SelectResponse) -> Tuple[Dict[str, List[Any]], Dict[str, Any], Dict]:
    data_dict = {}
    data_type_dict = {}
    column_counter = defaultdict(int)
    for column_def, column_field in zip(res.column_defs, res.column_fields):
        original_column_name = column_def.name
        column_counter[original_column_name] += 1
        column_name = (
            f"{original_column_name}_{column_counter[original_column_name]}"
            if column_counter[original_column_name] > 1
            else original_column_name
        )
        column_type = column_field.column_type
        column_data_type = column_def.data_type
        column_vectors = column_field.column_vectors
        data_list = column_vector_to_list(column_type, column_data_type, column_vectors)
        data_dict[column_name] = data_list
        data_type_dict[column_name] = column_data_type
    extra_result = None
    if res.extra_result is not None:
        try:
            extra_result = json.loads(res.extra_result)
        except json.JSONDecodeError:
            pass
    return data_dict, data_type_dict, extra_result

def make_match_tensor_expr(vector_column_name: str, embedding_data: VEC, embedding_data_type: str, method_type: str,
                           extra_option: str = None, filter_expr: Optional[ParsedExpr] = None) -> MatchTensorExpr:
    match_tensor_expr = MatchTensorExpr()
    match_tensor_expr.column_expr = ColumnExpr(column_name=[vector_column_name], star=False)
    match_tensor_expr.search_method = method_type
    match_tensor_expr.extra_options = extra_option
    match_tensor_expr.filter_expr = filter_expr
    data = EmbeddingData()
    if embedding_data_type == 'bit':
        raise SwarndbException(ErrorCode.INVALID_EMBEDDING_DATA_TYPE, f"Invalid embedding {embedding_data[0]} type")
    elif embedding_data_type in ['unsigned tinyint', 'uint8', 'u8']:
        elem_type = ElementType.ElementUInt8
        data.u8_array_value = np.asarray(embedding_data, dtype=np.uint8).flatten()
    elif embedding_data_type in ['tinyint', 'int8', 'i8']:
        elem_type = ElementType.ElementInt8
        data.i8_array_value = np.asarray(embedding_data, dtype=np.int8).flatten()
    elif embedding_data_type in ['smallint', 'int16', 'i16']:
        elem_type = ElementType.ElementInt16
        data.i16_array_value = np.asarray(embedding_data, dtype=np.int16).flatten()
    elif embedding_data_type in ['int', 'int32', 'i32']:
        elem_type = ElementType.ElementInt32
        data.i32_array_value = np.asarray(embedding_data, dtype=np.int32).flatten()
    elif embedding_data_type in ['bigint', 'int64', 'i64']:
        elem_type = ElementType.ElementInt64
        data.i64_array_value = np.asarray(embedding_data, dtype=np.int64).flatten()
    elif embedding_data_type in ['float', 'float32', 'f32']:
        elem_type = ElementType.ElementFloat32
        data.f32_array_value = np.asarray(embedding_data, dtype=np.float32).flatten()
    elif embedding_data_type in ['double', 'float64', 'f64']:
        elem_type = ElementType.ElementFloat64
        data.f64_array_value = np.asarray(embedding_data, dtype=np.float64).flatten()
    elif embedding_data_type in ['float16', 'fp16', 'f16']:
        elem_type = ElementType.ElementFloat16
        data.f16_array_value = np.asarray(embedding_data, dtype=np.float16).flatten()
    elif embedding_data_type in ['bfloat16', 'bf16']:
        elem_type = ElementType.ElementBFloat16
        data.bf16_array_value = np.asarray(embedding_data, dtype=np.float32).flatten()
    else:
        raise SwarndbException(ErrorCode.INVALID_EMBEDDING_DATA_TYPE, f"Invalid embedding {embedding_data[0]} type")
    match_tensor_expr.embedding_data_type = elem_type
    match_tensor_expr.embedding_data = data
    return match_tensor_expr

def make_match_sparse_expr(vector_column_name: str, sparse_data: Union[SparseVector, Dict], metric_type: str, topn: int,
                           opt_params: Optional[Dict] = None, filter_expr: Optional[ParsedExpr] = None) -> MatchSparseExpr:
    column_expr = ColumnExpr(column_name=[vector_column_name], star=False)
    query_sparse_expr = ConstantExpr()
    match sparse_data:
        case SparseVector([int(), *_] as indices, [int(), *_] as values):
            query_sparse_expr.literal_type = LiteralType.SparseIntegerArray
            query_sparse_expr.i64_array_idx = indices
            query_sparse_expr.i64_array_value = values
        case SparseVector([int(), *_] as indices, [float(), *_] as values):
            query_sparse_expr.literal_type = LiteralType.SparseDoubleArray
            query_sparse_expr.i64_array_idx = indices
            query_sparse_expr.f64_array_value = values
        case SparseVector([int(), *_], None):
            raise SwarndbException(ErrorCode.INVALID_CONSTANT_TYPE,
                                    f"No values! Sparse data does not support bool value type now")
        case dict():
            if len(sparse_data) == 0:
                raise SwarndbException(ErrorCode.INVALID_EXPRESSION, "Empty sparse vector")
            match next(iter(sparse_data.values())):
                case int():
                    query_sparse_expr.literal_type = LiteralType.SparseIntegerArray
                    query_sparse_expr.i64_array_idx = [int(kk) for kk in sparse_data.keys()]
                    query_sparse_expr.i64_array_value = [int(vv) for vv in sparse_data.values()]
                case float():
                    query_sparse_expr.literal_type = LiteralType.SparseDoubleArray
                    query_sparse_expr.i64_array_idx = [int(kk) for kk in sparse_data.keys()]
                    query_sparse_expr.f64_array_value = [float(vv) for vv in sparse_data.values()]
                case _:
                    raise SwarndbException(ErrorCode.INVALID_EXPRESSION,
                                            f"Invalid sparse vector value type: {type(next(iter(sparse_data.values())))}")
        case _:
            raise SwarndbException(ErrorCode.INVALID_CONSTANT_TYPE, f"Invalid sparse data type {type(sparse_data)}")
    match_sparse_options = []
    if opt_params is not None:
        for k, v in opt_params.items():
            match_sparse_options.append(InitParameter(param_name=k, param_value=v))
    match_sparse_expr = MatchSparseExpr(
        column_expr=column_expr, 
        query_sparse_expr=query_sparse_expr,
        metric_type=metric_type,
        topn=topn, 
        opt_params=match_sparse_options, 
        filter_expr=filter_expr
    )
    return match_sparse_expr
