"""
Utility functions for SwarndbDB
"""
__all__ = ["binary_exp_to_paser_exp", "deprecated_api"]

import warnings
from typing import Dict, Callable
# from functools import lru_cache

from .common import SwarndbException
from .errors import ErrorCode


# Performance optimization: Use a lookup dictionary instead of if-elif chains
BINARY_EXPR_MAP: Dict[str, str] = {
    "eq": "=",
    "gt": ">",
    "lt": "<",
    "gte": ">=",
    "lte": "<=",
    "neq": "!=",
    "and": "and",
    "or": "or",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
}

def binary_exp_to_paser_exp(binary_expr_key: str) -> str:
    """
    Convert binary expression key to parser expression.

    Args:
        binary_expr_key: The binary expression key

    Returns:
        str: The corresponding parser expression

    Raises:
        SwarndbException: If the expression is unknown
    """
    try:
        return BINARY_EXPR_MAP[binary_expr_key]
    except KeyError:
        raise SwarndbException(
            ErrorCode.INVALID_EXPRESSION,
            f"Unknown binary expression: {binary_expr_key}"
        )


def deprecated_api(message: str) -> Callable:
    """
    Decorator to mark APIs as deprecated.

    Args:
        message: The deprecation message

    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator