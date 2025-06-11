
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
        cmd = f"""python swarndb_benchmark.py --mode insert \
            --host {args.host} --port {args.port} \
            --table artificial_benchmark \
            --num-vectors {args.num_vectors} --vector-dim 1536 \
            --batch-size 1000 --num-threads 4 \
            --create-new"""
    
    # Example command for search benchmark
    else:
        cmd = f"""python swarndb_benchmark.py --mode search \
            --host {args.host} --port {args.port} \
            --table artificial_benchmark \
            --num-vectors {args.num_vectors} --num-queries {args.num_queries} \
            --vector-dim 1536 --top-k 100 \
            --evaluate-accuracy --batch-search"""
    
    print("\nRunning benchmark with command:")
    print(cmd)
    print("\nExecute this command to run the benchmark.")
