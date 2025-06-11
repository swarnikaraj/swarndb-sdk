Below is an example README.md file for the SwarnaDB SDK. This file explains what the SDK is, its features, how the project is structured, how to install it, how to use it, and other relevant details. You can adjust it further as needed.

---

```markdown
# SwarnaDB SDK

The **SwarnaDB SDK** is a Python client library for interacting with SwarnaDB – a vector database optimized for handling high-dimensional data such as vectors, tensors, and sparse data. This SDK provides a clean, Thrift‑based API for database operations, table management, and complex vector search queries, with a focus on performance and ease-of-use.

## Overview

SwarnaDB SDK is a fork of the Infinity SDK with the following improvements:
- **Rebranded for SwarnaDB:** All references have been updated from "infinity" to "swarna".
- **Enhanced Code Quality:** Type hints, docstrings, and caching for better performance and clarity.
- **Cross-Platform Compatibility:** Designed to run on both macOS and Windows.
- **Modular Structure:** Clean separation between core functionality and Thrift communication.

## Features

- **Connect to SwarnaDB:** Establish connections to your SwarnaDB server using a simple API.
- **Database & Table Management:** Create, drop, and query databases and tables.
- **Advanced Query Builder:** Build complex queries (dense, sparse, text, tensor, and fusion searches) using a fluent, chainable interface.
- **Result Conversion:** Convert query results into pandas, polars, or Apache Arrow DataFrames.
- **Thrift-Based Communication:** Efficient and reliable communication with the server using Apache Thrift.
- **Customizable & Extensible:** Easily extend or modify the SDK for your specific use case.

## Project Structure

```
swarna_sdk/
├── pyproject.toml                # Build configuration and dependency information
├── README.md                     # This file
├── swarna/                       # Core package for SwarnaDB SDK
│   ├── __init__.py               # Package initialization and high-level API (e.g., connect())
│   ├── common.py                 # Common utilities, exceptions, data classes (e.g., ConflictType, SwarnaException)
│   ├── connection_pool.py        # Manages connection pooling
│   ├── db.py                     # Database-level operations
│   ├── index.py                  # Index management
│   ├── table.py                  # Table management (RemoteTable, etc.)
│   ├── utils.py                  # Utility functions (expression conversion, name validation, etc.)
│   └── ...                       # Other modules as needed
└── swarna/remote_thrift/         # Thrift communication layer
    ├── client.py                 # Thrift client for communication
    ├── db.py                     # Thrift‑based database operations
    ├── query_builder.py          # Query builder for constructing queries
    ├── table.py                  # Thrift table operations implementation
    └── swarna_thrift_rpc/        # Auto‑generated Thrift code (service definitions, constants, types)
        ├── swarnaService.py
        ├── constants.py
        └── ttypes.py
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/swarna_sdk.git
   cd swarna_sdk
   ```

2. **Create and Activate a Virtual Environment (Python 3.11 recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: myenv\Scripts\activate
   ```

3. **Install the SDK in Editable Mode:**

   ```bash
   pip install -e .
   ```

## Usage

Below is a simple example to demonstrate how to connect to SwarnaDB, list databases, and run a query.

```python
# test_swarna.py

from swarna import connect
from swarna.common import NetworkAddress

def main():
    # Define the server address (update with your actual server IP and port)
    server_ip = "127.0.0.1"
    server_port = 23817
    uri = NetworkAddress(server_ip, server_port)
    
    # Connect to SwarnaDB
    connection = connect(uri)
    print("Connected to SwarnaDB!")

    # List available databases
    db_list = connection.list_databases()
    print("Available databases:", db_list)
    
    # Get a database and table
    db = connection.get_database("default_db")
    table = db.get_table("test_table")
    
    # Build a query using the query builder
    query_builder = table.query_builder()  # Get query builder for the table
    query_builder.output(["*"])  # Select all columns
    query_string = query_builder.to_string()
    print("Generated Query:", query_string)
    
    # Execute the query and get results as a pandas DataFrame
    df, extra = query_builder.to_df()
    print("Query Results:")
    print(df)
    
    # Disconnect when finished
    connection.disconnect()
    print("Disconnected successfully.")

if __name__ == "__main__":
    main()
```

Run the example:

```bash
python test_swarna.py
```

## Testing

- **Local Testing:**  
  Use the provided test script (`test_swarna.py`) or write your own tests.
  
- **Automated Testing:**  
  Use pytest to run the test suite:
  ls
  
  ```bash
  pytest
  ```

## Cross-Platform Compatibility

SwarnaDB SDK is designed to work on both macOS and Windows. Ensure you have the proper Python version (3.11 recommended) and the required build tools installed (e.g., Xcode Command Line Tools on macOS).

