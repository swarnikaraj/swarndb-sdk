import sys
from pathlib import Path

# Add the parent directory to sys.path to access swarna_sdk
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from swarna import connect
from swarna.common import NetworkAddress
import numpy as np
from typing import List, Dict
import time

# Database configuration
SWARN_HOST = "127.0.0.1"
SWARN_PORT = 23817

DATABASE_NAME = "default_db"
COLLECTION_NAME = "benchmark_table"
VECTOR_DIM = 1536

class SwarnDB:
    def __init__(self):
        self.dimension = VECTOR_DIM
        self.client = self._create_client()
        self.db = self.client.get_database(DATABASE_NAME)
        self.table = self._get_or_create_table()
        self._init_connection()

    def _init_connection(self):
        """Initialize and verify connection"""
        try:
            # Perform a simple operation to verify connection
            self.table.output(["id"]).limit(1).to_pl()
        except Exception as e:
            print(f"Connection initialization error: {e}")
            raise

    def _create_client(self):
        """Create a new client connection to SwarnDB"""
        try:
            address = NetworkAddress(SWARN_HOST, SWARN_PORT)
            print(f"THIS IS ADDRESS: {address}")
            return connect(uri=address)
        except Exception as e:
            print(f"Error connecting to SwarnDB: {e}")
            raise

    def _get_or_create_table(self):
        """Get existing table or create a new one"""
        try:
            # Create new table with optimized schema
            table = self.db.create_table(
                COLLECTION_NAME,
                {
                    "id": {"type": "integer"},
                    "vec": {"type": f"vector,{self.dimension},float"},
                    "metadata": {"type": "string"}
                }
            )
            return table
        except Exception as e:
            print(f"Error creating table: {e}")
            raise

    async def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Perform optimized vector similarity search"""
        try:
            # Ensure query vector is normalized and in the correct format
            query_vector = query_vector.astype(np.float32)
            if np.linalg.norm(query_vector) != 0:
                query_vector = query_vector / np.linalg.norm(query_vector)

            # Perform search with minimal output fields
            results = self.table.output(["id", "metadata"]).match_dense(
                "vec",
                query_vector.tolist(),
                "float",
                "cosine",
                top_k
            ).to_pl()

            # Optimize result formatting
            return [
                {
                    "id": str(row['id']),
                    "similarity": float(1 - row.get('distance', 0)),
                    "metadata": row.get('metadata', '')
                }
                for row in results.to_dict('records')
            ]

        except Exception as e:
            print(f"Search error: {e}")
            return []

    async def get_table_stats(self) -> Dict:
        """Get basic table statistics"""
        try:
            return {
                "vector_dimension": self.dimension,
                "total_vectors": len(self.table.output(["id"]).to_pl())
            }
        except Exception as e:
            print(f"Error getting table stats: {e}")
            return {}