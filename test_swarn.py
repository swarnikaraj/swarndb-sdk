# test_swarn_sdk.py
"""
Sample Test for Swarn SDK

This script demonstrates:
1. Connecting to SwarnaDB using your server's IP and port.
2. Listing the available databases.
3. Building and executing a query on a table.
4. Converting the query results to a pandas DataFrame.
5. Disconnecting from the server.
"""

from swarn import connect       
from swarn.common import NetworkAddress


def main():
    # Step 1: Specify the server's address.
    # Think of this as dialing the robot car's phone number.
    server_ip = "127.0.0.1"
    server_port = 23817
    address = NetworkAddress(server_ip, server_port)
    
    try:
        # Step 2: Connect to the server.
        # This creates a connection using our remote control (the SDK).
        print(f"THIS IS address:{address}")
        connection = connect(uri=address)
        print("Connected successfully to the server!")
        
        # Step 3: Test an operation.
        # For example, list databases. This is like asking the robot car: "What databases do you have?"
        databases = connection.list_databases()
        print("Databases available:", databases)
        
        # You can add more tests here:
        # For example, create a new database (if your server supports it):
        # new_db = connection.create_database("TestDB")
        # print("Created new database:", new_db)
        
        # Step 4: Disconnect from the server when done.
        connection.disconnect()
        print("Disconnected successfully!")
        
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()