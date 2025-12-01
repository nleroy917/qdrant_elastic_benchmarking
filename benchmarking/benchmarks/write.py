from typing import Dict, List

from benchmarking.backends import SearchBackend
from benchmarking.metrics import Timer, CPUMonitor, BenchmarkResult

def generate_elasticsearch_docs(backend: SearchBackend):
    """
    Generate documents in Elasticsearch format
    """
    for idx, row in enumerate(backend.df.iter_rows(named=True)):
        doc = {
            "_id": idx,
            "_source": {
                "main_category": row["main_category"],
                "title": row["title"],
                "average_rating": row["average_rating"],
                "rating_number": row["rating_number"],
                "features": row["features"],
                "description": row["description"],
                "price": row["price"],
                "store": row["store"],
                "categories": row["categories"],
                "brand": row["brand"],
                "manufacturer": row["manufacturer"],
                "brand_name": row["brand_name"],
                "embedding": row["embedding"]
            }
        }
        yield doc


def generate_qdrant_docs(backend: SearchBackend):
    """
    Generate documents in Qdrant format
    """
    for idx, row in enumerate(backend.df.iter_rows(named=True)):
        doc = {
            "_id": idx,
            "vector": {
                "embedding": row["embedding"],
                "bm25": {
                    'values': row['bm25_values'],
                    'indices': row['bm25_indices']
                }
            },
            "payload": {
                "main_category": row["main_category"],
                "title": row["title"],
                "average_rating": row["average_rating"],
                "rating_number": row["rating_number"],
                "features": row["features"],
                "description": row["description"],
                "price": row["price"],
                "store": row["store"],
                "categories": row["categories"],
                "brand": row["brand"],
                "manufacturer": row["manufacturer"],
                "brand_name": row["brand_name"]
            }
        }
        yield doc


def benchmark_write(
    backend: SearchBackend,
    backend_name: str,
    index_name: str,
    schema: Dict,
    batch_sizes: List[int] = None,
) -> Dict[int, BenchmarkResult]:
    """
    Benchmark write performance for a backend

    Args:
        backend: SearchBackend instance (Elasticsearch or Qdrant)
        backend_name: Name of the backend for reporting
        index_name: Name of index/collection to create
        schema: Schema/config for the index
        batch_sizes: List of batch sizes to test

    Returns:
        Dictionary mapping batch_size to BenchmarkResult
    """
    if batch_sizes is None:
        batch_sizes = [100, 500, 1000]

    results = {}

    for batch_size in batch_sizes:
        # reset index before each run
        backend.reset_index(index_name)
        backend.create_index(index_name, schema)

        # generate documents based on backend type
        if "Elasticsearch" in backend.__class__.__name__:
            docs = generate_elasticsearch_docs(backend)
        else:
            docs = generate_qdrant_docs(backend)

        # Run benchmark
        cpu_monitor = CPUMonitor()
        cpu_monitor.start()

        timer = Timer()
        with timer:
            success = backend.index_documents(index_name, docs, batch_size=batch_size)

        cpu_stats = cpu_monitor.stop()

        result = BenchmarkResult(
            name=f"{backend_name}_write_batch_{batch_size}",
            engine=backend_name,
            workload_type="write",
            duration_seconds=timer.elapsed_seconds,
            total_operations=success,
            latency_metrics=timer.get_latency_metrics(),
            throughput_ops_per_sec=success / timer.elapsed_seconds if timer.elapsed_seconds > 0 else 0,
            avg_cpu_usage_percent=cpu_stats["avg_cpu_percent"],
            peak_cpu_usage_percent=cpu_stats["peak_cpu_percent"],
            avg_memory_mb=cpu_stats["avg_memory_mb"],
            peak_memory_mb=cpu_stats["peak_memory_mb"],
        )

        results[batch_size] = result

        print(f"\n{backend_name} Write Benchmark (batch_size={batch_size}):")
        print(f"  Duration: {timer.elapsed_seconds:.2f}s")
        print(f"  Operations: {success}")
        print(f"  Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
        print(f"  Avg CPU: {cpu_stats['avg_cpu_percent']:.1f}%")
        print(f"  Peak Memory: {cpu_stats['peak_memory_mb']:.1f} MB")

    return results
