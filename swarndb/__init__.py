"""
SwarndbDB Python SDK initialization
"""

import logging
import os

from swarndb.common import (
    URI, 
    NetworkAddress, 
    LOCAL_HOST, 
    LOCAL_SWARNDB_PATH, 
    SwarndbException
)
from swarndb.swarndb import SwarndbConnection
from swarndb.remote_thrift.swarndb import RemoteThriftSwarndbConnection
from swarndb.errors import ErrorCode

def swarndb_client(uri=LOCAL_HOST, logger: logging.Logger = None) -> SwarndbConnection:
    """
    Create a connection to SwarndbDB.

    Args:
        uri: Connection URI or NetworkAddress object
        logger: Optional logger instance

    Returns:
        SwarndbConnection: A connection object

    Raises:
        SwarndbException: If connection fails
    """
    print(f"THIS IS uri: {uri}")
    print(f"THIS IS networkaddress: {NetworkAddress}")
    print(f"THIS iS INSTANCE: {isinstance(uri, NetworkAddress)}")

    if isinstance(uri, NetworkAddress):
        return RemoteThriftSwarndbConnection(uri, logger)
    else:
        raise SwarndbException(ErrorCode.INVALID_SERVER_ADDRESS, f"Unknown uri: {uri}")