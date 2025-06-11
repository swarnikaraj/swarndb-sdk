
import numpy as np
import time
import argparse
from tqdm import tqdm
import concurrent.futures
import threading
from typing import List, Dict, Any
import logging
import os
from swarndb import swarndb_client , index      
from swarndb.common import NetworkAddress,LOCAL_HOST
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataGenerator:
    """Class to generate artificial vector data for benchmarking"""
    
    @staticmethod
    def generate_dataset(num_vectors=100000, dim=1536, seed=42):
        """Generate artificial dataset with specified dimensions"""
        logger.info(f"Generating artificial dataset with {num_vectors} vectors of dimension {dim}")
        np.random.seed(seed)
        # Generate random vectors
        vectors = np.random.randn(num_vectors, dim)
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        return vectors
    
    @staticmethod
    def generate_queries(num_queries=10000, dim=1536, seed=43):
        """Generate artificial query vectors"""
        logger.info(f"Generating {num_queries} query vectors of dimension {dim}")
        np.random.seed(seed)
        queries = np.random.randn(num_queries, dim)
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / norms
        return queries
    
    @staticmethod
    def generate_ground_truth(dataset, queries, top_k=100):
        """Generate ground truth by computing exact nearest neighbors"""
        logger.info(f"Computing ground truth for {len(queries)} queries (top-{top_k})")
        num_queries = queries.shape[0]
        ground_truth = np.zeros((num_queries, top_k), dtype=np.int32)
        
        # For each query, find top_k nearest neighbors
        for i in tqdm(range(num_queries), desc="Computing ground truth"):
            # Compute cosine similarity (dot product of normalized vectors)
            similarities = np.dot(dataset, queries[i])
            # Get indices of top_k most similar vectors
            indices = np.argsort(similarities)[-top_k:][::-1]
            ground_truth[i] = indices
        
        return ground_truth
    
    @staticmethod
    def save_vectors(vectors, filename):
        """Save vectors to binary file"""
        logger.info(f"Saving {len(vectors)} vectors to {filename}")
        vectors.astype(np.float32).tofile(filename)
        
    @staticmethod
    def save_ground_truth(ground_truth, filename):
        """Save ground truth to binary file"""
        logger.info(f"Saving ground truth to {filename}")
        ground_truth.astype(np.int32).tofile(filename)


class SwarndbBenchmark:
    def __init__(self, host: str, port: int, table_name: str, vector_dim: int):
        self.host = host
        self.port = port
        self.table_name = table_name
        self.vector_dim = vector_dim
        self._local = threading.local()
        self.connection_lock = threading.Lock()
        self.error_count = 0
        self.max_retries = 3

    def get_connection(self):
        """Get or create a thread-local connection with retry mechanism"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            for attempt in range(self.max_retries):
                try:
                    self._local.connection = swarndb_client(LOCAL_HOST)
                    self._local.db = self._local.connection.get_database("default_db")
                    self._local.table = self._local.db.get_table(self.table_name)
                    break
                except Exception as e:
                    logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(1)  # Wait before retrying
        return self._local.table

    def setup_table(self, create_new: bool = False):
        """Initialize the table with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                connection = swarndb_client(LOCAL_HOST)
                db = connection.get_database("default_db")

                if create_new:
                    try:
                        db.drop_table(self.table_name)
                        logger.info(f"Dropped existing table: {self.table_name}")
                    except Exception as e:
                        logger.warndbing(f"Drop table warndbing: {str(e)}")

                    logger.info(f"Creating table: {self.table_name}")
                    start_time = time.time()
                    db.create_table(
                        self.table_name,
                        {
                            "id": {"type": "integer"},
                            "vec": {"type": f"vector,{self.vector_dim},float"}
                        }
                    )
                    table_creation_time = time.time() - start_time
                    logger.info(f"Table creation time: {table_creation_time:.2f} seconds")

                return db.get_table(self.table_name)
            except Exception as e:
                logger.error(f"Setup attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)

    def create_index(self, ef_construction: int = 200, metric: str = "cosine"):
        """Create index with accurate timing measurement"""
        logger.info(f"Creating index with ef_construction={ef_construction}, metric={metric}")

        table = self.get_connection()

        # Prepare parameters based on the correct API format
        index_params = {
            "m": "32",
            "ef_construction": str(ef_construction),
            "metric": metric
        }

        logger.info(f"Creating index with parameters: {index_params}")

        # Measure only the actual index creation time
        start_time = time.time()

        # Use the correct API format
        table.create_index("hnsw_index",
                        index.IndexInfo("vec",
                                                index.IndexType.Hnsw,
                                                index_params))

        index_creation_time = time.time() - start_time

        logger.info(f"Index creation time: {index_creation_time:.2f} seconds")
        return index_creation_time

    def insert_batch(self, batch_vectors: np.ndarray, start_idx: int) -> float:
        """Insert a batch of vectors with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                table = self.get_connection()

                # IMPORTANT: Use sequential IDs starting from 0
                # This ensures IDs match the ground truth
                batch_ids = list(range(start_idx, start_idx + len(batch_vectors)))

                # Prepare records before timing to minimize overhead
                records = [{"id": idx, "vec": vec.tolist()} for idx, vec in zip(batch_ids, batch_vectors)]

                # Measure only the actual insertion time
                start_time = time.time()
                table.insert(records)
                insertion_time = time.time() - start_time

                return insertion_time
            except Exception as e:
                logger.error(f"Batch insertion attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
                self._local.connection = None  # Force new connection on retry

    def parallel_insert(self, vectors: np.ndarray, batch_size: int = 10000, num_threads: int = 4) -> Dict[str, float]:
        """Insert vectors using multiple threads with improved error handling"""
        logger.info(f"Starting parallel insertion of {len(vectors)} vectors with {num_threads} threads")
        total_start_time = time.time()

        # Split vectors into batches
        num_vectors = len(vectors)
        batches = [(i, vectors[i:i + batch_size])
                  for i in range(0, num_vectors, batch_size)]

        batch_times = []
        failed_batches = []

        # Create thread pool and submit tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_batch = {
                executor.submit(self.insert_batch, batch_vectors, start_idx): (start_idx, batch_vectors)
                for start_idx, batch_vectors in batches
            }

            # Process completed futures with progress bar
            with tqdm(total=len(batches), desc="Inserting batches") as pbar:
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        batch_time = future.result()
                        batch_times.append(batch_time)
                    except Exception as e:
                        start_idx, batch_vectors = future_to_batch[future]
                        logger.error(f"Batch starting at index {start_idx} failed: {str(e)}")
                        failed_batches.append((start_idx, batch_vectors))
                    pbar.update(1)

        # Report results
        total_time = time.time() - total_start_time
        successful_vectors = (len(batches) - len(failed_batches)) * batch_size
        qps = successful_vectors / total_time if successful_vectors > 0 else 0

        logger.info("\nInsertion Performance:")
        logger.info(f"Total insertion time: {total_time:.2f} seconds")
        logger.info(f"Successfully inserted: {successful_vectors}/{num_vectors} vectors")
        logger.info(f"Failed batches: {len(failed_batches)}")
        logger.info(f"Average batch time: {np.mean(batch_times):.2f} seconds")
        logger.info(f"Vectors per second (QPS): {qps:.2f}")

        return {
            "total_time": total_time,
            "qps": qps,
            "avg_batch_time": np.mean(batch_times) if batch_times else 0,
            "failed_batches": len(failed_batches)
        }

    def search_single(self, query_vector: np.ndarray, top_k: int = 100) -> float:
        """Perform single vector search with minimal overhead"""
        for attempt in range(self.max_retries):
            try:
                table = self.get_connection()

                # Prepare query parameters before timing to minimize overhead
                query_params = {
                    "vec": query_vector.tolist(),
                    "float": "float",
                    "cosine": "cosine",
                    "top_k": top_k
                }

                # Measure only the actual search time
                start_time = time.perf_counter()  # More precise timing
                _ = table.output(["id"]).match_dense(
                    "vec",
                    query_vector.tolist(),
                    "float",
                    "cosine",
                    top_k
                ).to_pl()
                end_time = time.perf_counter()

                return (end_time - start_time) * 1000  # Convert to ms
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
                self._local.connection = None

    def search_batch(self, query_vectors: np.ndarray, batch_size: int = 100, top_k: int = 100, ef_search: int = None) -> float:
        """Perform batch vector search with minimal overhead"""
        table = self.get_connection()

        # Prepare search parameters if ef_search is specified
        search_params = {}
        if ef_search is not None:
            search_params["ef"] = str(ef_search)

        # Measure only the actual search time
        start_time = time.perf_counter()  # More precise timing

        # Process in batches to reduce Python overhead
        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:i+batch_size]

            # Convert batch to list of lists once
            batch_lists = [vec.tolist() for vec in batch]

            # Process each query in the batch
            for query in batch_lists:
                if search_params:
                    _ = table.output(["id"]).match_dense(
                        "vec",
                        query,
                        "float",
                        "cosine",
                        top_k,
                        search_params
                    ).to_pl()
                else:
                    _ = table.output(["id"]).match_dense(
                        "vec",
                        query,
                        "float",
                        "cosine",
                        top_k
                    ).to_pl()

        end_time = time.perf_counter()

        # Return total time for the batch in ms
        return (end_time - start_time) * 1000

    def run_batch_search_benchmark(self, query_vectors: np.ndarray, top_k: int = 100, warmup: int = 100, batch_size: int = 100, ef_search: int = None) -> Dict[str, float]:
        """Run search benchmark with minimal Python overhead"""
        # Perform warmup queries
        logger.info(f"Running {warmup} warmup queries (not included in metrics)...")
        warmup_vectors = query_vectors[:min(warmup, len(query_vectors))]
        self.search_batch(warmup_vectors, batch_size, top_k, ef_search)

        logger.info(f"Running batch search benchmark with {len(query_vectors)} queries (batch size: {batch_size})")

        # Use C-based timing for maximum precision
        total_start_time = time.perf_counter()

        # Process all queries with minimal Python overhead
        total_latency = self.search_batch(query_vectors, batch_size, top_k, ef_search)

        total_time = time.perf_counter() - total_start_time

        # Calculate QPS directly
        qps = len(query_vectors) / total_time if total_time > 0 else 0
        avg_latency = total_latency / len(query_vectors)

        logger.info("\nBatch Search Performance:")
        logger.info(f"Total search time: {total_time:.2f} seconds")
        logger.info(f"Queries per second (QPS): {qps:.2f}")
        logger.info(f"Average Latency: {avg_latency:.2f} ms")
        logger.info(f"Failed queries: 0")  # We're not tracking individual failures in batch mode

        return {
            "avg_latency": avg_latency,
            "total_time": total_time,
            "qps": qps
        }

    def evaluate_accuracy(self, query_vectors: np.ndarray, ground_truth: np.ndarray, top_k: int = 100) -> Dict[str, float]:
        """
        Evaluate search accuracy using ground truth data

        Args:
            query_vectors: Query vectors to search with
            ground_truth: Ground truth results for each query
            top_k: Number of results to retrieve

        Returns:
            Dictionary with precision and recall metrics
        """
        logger.info(f"Evaluating search accuracy with {len(query_vectors)} queries")

        # Debug: Check ground truth structure
        if len(ground_truth) > 0:
            logger.info(f"Ground truth shape: {ground_truth.shape}")
            logger.info(f"First ground truth entry (first 10 elements): {ground_truth[0][:10]}")
            logger.info(f"Number of unique IDs in first ground truth: {len(set(ground_truth[0]))}")

        # Debug: Test with a single query first
        try:
            table = self.get_connection()
            test_result = table.output(["id"]).match_dense(
                "vec",
                query_vectors[0].tolist(),
                "float",
                "cosine",
                top_k
            ).to_result()

            # Print the type and structure of the result
            logger.info(f"Debug - Result type: {type(test_result)}")
            if isinstance(test_result, tuple) and len(test_result) > 0 and isinstance(test_result[0], dict) and 'id' in test_result[0]:
                ids = test_result[0]['id']
                logger.info(f"Debug - First search result (first 10 elements): {ids[:10]}")
                logger.info(f"Debug - Number of unique IDs in first result: {len(set(ids))}")

                # Check for any overlap with ground truth
                if len(ground_truth) > 0:
                    common = set(ids).intersection(set(ground_truth[0]))
                    logger.info(f"Debug - Common elements in first result: {len(common)}")
                    if len(common) > 0:
                        logger.info(f"Debug - Some common elements: {list(common)[:5]}")
        except Exception as e:
            logger.error(f"Debug query failed: {str(e)}")

        search_results = []
        failed_queries = 0

        for i, query_vector in enumerate(tqdm(query_vectors, desc="Running accuracy evaluation")):
            try:
                table = self.get_connection()
                result = table.output(["id"]).match_dense(
                    "vec",
                    query_vector.tolist(),
                    "float",
                    "cosine",
                    top_k
                ).to_result()

                # Based on the debug output, extract IDs correctly
                # The result is a tuple, and the first element is a dict with 'id' key
                if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], dict) and 'id' in result[0]:
                    ids = result[0]['id']
                    search_results.append(ids)
                else:
                    logger.warndbing(f"Unexpected result structure for query {i}: {type(result)}")
                    search_results.append([])
                    failed_queries += 1

            except Exception as e:
                logger.error(f"Query {i} failed during accuracy evaluation: {str(e)}")
                failed_queries += 1
                # Add empty result to maintain alignment with ground truth
                search_results.append([])

        # Calculate metrics
        recall = self._calculate_recall(search_results, ground_truth, top_k)
        precision = self._calculate_precision(search_results, ground_truth, top_k)

        logger.info("\nAccuracy Metrics:")
        logger.info(f"Recall@{top_k}: {recall:.4f}")
        logger.info(f"Precision@{top_k}: {precision:.4f}")
        logger.info(f"Failed queries: {failed_queries}")

        return {
            f"recall@{top_k}": recall,
            f"precision@{top_k}": precision,
            "failed_queries": failed_queries
        }

    def _calculate_recall(self, search_results: List[List[int]], ground_truth: np.ndarray, k: int = 100) -> float:
        """
        Calculate recall@k for search results compared to ground truth

        Args:
            search_results: List of lists, each inner list contains IDs returned by search
            ground_truth: numpy array of ground truth IDs for each query
            k: k value for recall@k

        Returns:
            Average recall@k across all queries
        """
        if len(search_results) != len(ground_truth):
            raise ValueError(f"Number of search results ({len(search_results)}) doesn't match ground truth ({len(ground_truth)})")

        recalls = []
        for i, (result, truth) in enumerate(zip(search_results, ground_truth)):
            # Convert result and truth to sets for intersection
            result_set = set(result[:k])
            truth_set = set(truth[:k])

            # Calculate recall
            if len(truth_set) > 0:
                recall = len(result_set.intersection(truth_set)) / len(truth_set)
                recalls.append(recall)

                # Debug first few results
                if i < 3:
                    logger.info(f"Query {i} - Result set size: {len(result_set)}, Truth set size: {len(truth_set)}")
                    logger.info(f"Query {i} - Intersection size: {len(result_set.intersection(truth_set))}")
                    logger.info(f"Query {i} - Recall: {recall:.4f}")

        return np.mean(recalls) if recalls else 0.0

    def _calculate_precision(self, search_results: List[List[int]], ground_truth: np.ndarray, k: int = 100) -> float:
        """
        Calculate precision@k for search results compared to ground truth

        Args:
            search_results: List of lists, each inner list contains IDs returned by search
            ground_truth: numpy array of ground truth IDs for each query
            k: k value for precision@k

        Returns:
            Average precision@k across all queries
        """
        if len(search_results) != len(ground_truth):
            raise ValueError(f"Number of search results ({len(search_results)}) doesn't match ground truth ({len(ground_truth)})")

        precisions = []
        for i, (result, truth) in enumerate(zip(search_results, ground_truth)):
            # Convert result and truth to sets for intersection
            result_set = set(result[:k])
            truth_set = set(truth[:k])

            # Calculate precision
            if len(result_set) > 0:
                precision = len(result_set.intersection(truth_set)) / len(result_set)
                precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0

    def run_search_benchmark(self, query_vectors: np.ndarray, top_k: int = 100, warmup: int = 100) -> Dict[str, float]:
        """Run search benchmark with warmup and error handling"""
        # Perform warmup queries
        logger.info(f"Running {warmup} warmup queries (not included in metrics)...")
        for i in range(min(warmup, len(query_vectors))):
            try:
                self.search_single(query_vectors[i], top_k)
            except Exception:
                pass

        logger.info(f"Running search benchmark with {len(query_vectors)} queries")
        latencies = []
        failed_queries = 0

        # Start timing after warmup
        total_start_time = time.time()

        for i, query_vector in enumerate(tqdm(query_vectors, desc="Running searches")):
            try:
                latency = self.search_single(query_vector, top_k)
                latencies.append(latency)
            except Exception as e:
                logger.error(f"Query {i} failed: {str(e)}")
                failed_queries += 1

        total_time = time.time() - total_start_time

        if not latencies:
            logger.error("No successful queries to report metrics on")
            return {
                "avg_latency": 0,
                "p95": 0,
                "p99": 0,
                "failed_queries": failed_queries,
                "total_time": total_time,
                "qps": 0
            }

        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        qps = len(latencies) / total_time if total_time > 0 else 0

        logger.info("\nSearch Performance:")
        logger.info(f"Total search time: {total_time:.2f} seconds")
        logger.info(f"Queries per second (QPS): {qps:.2f}")
        logger.info(f"Average Latency: {avg_latency:.2f} ms")
        logger.info(f"P95 Latency: {p95:.2f} ms")
        logger.info(f"P99 Latency: {p99:.2f} ms")
        logger.info(f"Failed queries: {failed_queries}")

        return {
            "avg_latency": avg_latency,
            "p95": p95,
            "p99": p99,
            "failed_queries": failed_queries,
            "total_time": total_time,
            "qps": qps
        }

def main():
    parser = argparse.ArgumentParser(description="swarndb Vector Search Benchmark with Artificial Data")
    parser.add_argument("--mode", choices=['insert', 'search'], required=True,
                      help="Benchmark mode: insert or search")
    parser.add_argument("--host", default="localhost", help="swarndb host")
    parser.add_argument("--port", type=int, default=23817, help="swarndb port")
    parser.add_argument("--table", default="artificial_benchmark", help="Table name")
    parser.add_argument("--batch-size", type=int, default=10000,
                      help="Batch size for insertion")
    parser.add_argument("--num-threads", type=int, default=4,
                      help="Number of threads for parallel insertion")
    parser.add_argument("--top-k", type=int, default=100,
                      help="Number of results to retrieve")
    parser.add_argument("--create-new", action="store_true",
                      help="Create new table")
    parser.add_argument("--ef-construction", type=int, default=200,
                      help="ef_construction parameter for HNSW index")
    parser.add_argument("--metric", default="cosine",
                      help="Distance metric for vector search")
    parser.add_argument("--warmup", type=int, default=100,
                      help="Number of warmup queries")
    parser.add_argument("--evaluate-accuracy", action="store_true",
                      help="Evaluate search accuracy using ground truth")
    parser.add_argument("--ef-search", type=int, default=100,
                      help="ef_search parameter for HNSW search")
    parser.add_argument("--batch-search", action="store_true",
                      help="Use batch search for minimal Python overhead")
    parser.add_argument("--search-batch-size", type=int, default=100,
                      help="Batch size for search operations")
    parser.add_argument("--num-vectors", type=int, default=100000,
                      help="Number of vectors in the dataset")
    parser.add_argument("--num-queries", type=int, default=10000,
                      help="Number of query vectors")
    parser.add_argument("--vector-dim", type=int, default=1536,
                      help="Vector dimension")
    parser.add_argument("--save-data", action="store_true",
                      help="Save generated data to files")
    parser.add_argument("--data-dir", default="artificial_data",
                      help="Directory to save generated data")

    args = parser.parse_args()

    # Create data directory if saving data
    if args.save_data and not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # Generate artificial data
    data_generator = DataGenerator()
    
    # Generate dataset (base vectors)
    dataset = data_generator.generate_dataset(args.num_vectors, args.vector_dim)
    
    # Generate query vectors
    query_vectors = data_generator.generate_queries(args.num_queries, args.vector_dim)
    
    # Save data if requested
    if args.save_data:
        data_generator.save_vectors(dataset, f"{args.data_dir}/base_vectors.bin")
        data_generator.save_vectors(query_vectors, f"{args.data_dir}/query_vectors.bin")
    
    # Generate ground truth if evaluating accuracy
    ground_truth = None
    if args.evaluate_accuracy:
        # Use a smaller subset for ground truth computation if dataset is large
        max_gt_vectors = min(args.num_vectors, 100000)  # Limit to 100k vectors for ground truth
        max_gt_queries = min(args.num_queries, 1000)    # Limit to 1000 queries for ground truth
        
        if args.num_vectors > max_gt_vectors or args.num_queries > max_gt_queries:
            logger.info(f"Using subset of data for ground truth: {max_gt_vectors} vectors, {max_gt_queries} queries")
            gt_dataset = dataset[:max_gt_vectors]
            gt_queries = query_vectors[:max_gt_queries]
        else:
            gt_dataset = dataset
            gt_queries = query_vectors
            
        ground_truth = data_generator.generate_ground_truth(gt_dataset, gt_queries, args.top_k)
        
        if args.save_data:
            data_generator.save_ground_truth(ground_truth, f"{args.data_dir}/ground_truth.bin")

    # Initialize benchmark
    benchmark = SwarndbBenchmark(args.host, args.port, args.table, args.vector_dim)

    if args.create_new or args.mode == 'insert':
        benchmark.setup_table(args.create_new)

    if args.mode == 'insert':
        # Insert vectors
        insert_results = benchmark.parallel_insert(dataset, args.batch_size, args.num_threads)

        # Create index after insertion with accurate timing
        index_time = benchmark.create_index(args.ef_construction, args.metric)
        logger.info(f"Index creation time: {index_time:.2f} seconds")
    else:
        # Search mode
        if args.batch_search:
            # Use optimized batch search
            benchmark.run_batch_search_benchmark(
                query_vectors,
                args.top_k,
                args.warmup,
                args.search_batch_size,
                args.ef_search
            )
        else:
            # Use original search method
            benchmark.run_search_benchmark(query_vectors, args.top_k, args.warmup)

        # Evaluate accuracy if requested
        if args.evaluate_accuracy and ground_truth is not None:
            # Use the same subset of queries as used for ground truth generation
            if args.num_queries > len(ground_truth):
                eval_queries = query_vectors[:len(ground_truth)]
            else:
                eval_queries = query_vectors
                
            benchmark.evaluate_accuracy(eval_queries, ground_truth, args.top_k)

if __name__ == "__main__":
    main()

# Create a simple example script to demonstrate usage
with open('run_benchmark.py', 'w') as f:
    f.write('''
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run swarndb Vector Search Benchmark")
    parser.add_argument("--mode", choices=['insert', 'search'], default='insert',
                      help="Benchmark mode: insert or search")
    parser.add_argument("--host", default="localhost", help="swarndb host")
    parser.add_argument("--port", type=int, default=8080, help="swarndb port")
    parser.add_argument("--num-vectors", type=int, default=10000,
                      help="Number of vectors in dataset (default: 10000)")
    parser.add_argument("--num-queries", type=int, default=1000,
                      help="Number of query vectors (default: 1000)")
    
    args = parser.parse_args()
    
    # Example command for insertion benchmark
    if args.mode == 'insert':
        cmd = f"""python swarndb_benchmark.py --mode insert \\
            --host {args.host} --port {args.port} \\
            --table artificial_benchmark \\
            --num-vectors {args.num_vectors} --vector-dim 1536 \\
            --batch-size 1000 --num-threads 4 \\
            --create-new"""
    
    # Example command for search benchmark
    else:
        cmd = f"""python swarndb_benchmark.py --mode search \\
            --host {args.host} --port {args.port} \\
            --table artificial_benchmark \\
            --num-vectors {args.num_vectors} --num-queries {args.num_queries} \\
            --vector-dim 1536 --top-k 100 \\
            --evaluate-accuracy --batch-search"""
    
    print("\\nRunning benchmark with command:")
    print(cmd)
    print("\\nExecute this command to run the benchmark.")
''')

print("Created swarndb_benchmark.py and run_benchmark.py")