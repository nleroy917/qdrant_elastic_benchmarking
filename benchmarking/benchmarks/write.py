"""Write workload benchmarks for comparing search backends"""

import time
from typing import Dict, List
from benchmarking.backends.backends import SearchBackend
from benchmarking.metrics import Timer, CPUMonitor, BenchmarkResult

PARQUET_FILE = "ecommerce_products_with_embeddings.parquet"

INDEX_SCHEMA_ES = {
    "mappings": {
        "properties": {
            "review_text": {"type": "text"},
            "age": {"type": "integer"},
            "rating": {"type": "integer"},
            "positive_feedback_count": {"type": "integer"},
            "division_name": {"type": "keyword"},
            "department_name": {"type": "keyword"},
            "class_name": {"type": "keyword"},
            "recommended_ind": {"type": "integer"},
            "text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "umap1": {"type": "float"},
            "umap2": {"type": "float"}
        }
    }
}

INDEX_SCHEMA_QDRANT = {
    "vector_size": 384,
}


def generate_elasticsearch_docs(backend: SearchBackend):
    """Generate documents in Elasticsearch format"""
    for idx, row in enumerate(backend.df.iter_rows(named=True)):
        doc = {
            "_id": idx,
            "_source": {
                "review_text": row["review_text"],
                "age": row["age"],
                "rating": row["rating"],
                "positive_feedback_count": row["positive_feedback_count"],
                "division_name": row["division_name"],
                "department_name": row["department_name"],
                "class_name": row["class_name"],
                "recommended_ind": row["recommended_ind"],
                "text": row["text"],
                "embedding": row["embedding"],
                "umap1": row["umap1"],
                "umap2": row["umap2"]
            }
        }
        yield doc


def generate_qdrant_docs(backend: SearchBackend):
    """Generate documents in Qdrant format"""
    for idx, row in enumerate(backend.df.iter_rows(named=True)):
        doc = {
            "_id": idx,
            "vector": row["embedding"],
            "payload": {
                "review_text": row["review_text"],
                "age": row["age"],
                "rating": row["rating"],
                "positive_feedback_count": row["positive_feedback_count"],
                "division_name": row["division_name"],
                "department_name": row["department_name"],
                "class_name": row["class_name"],
                "recommended_ind": row["recommended_ind"],
                "text": row["text"],
                "umap1": row["umap1"],
                "umap2": row["umap2"]
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
        # Reset index before each run
        backend.reset_index(index_name)
        backend.create_index(index_name, schema)

        # Generate documents based on backend type
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


if __name__ == "__main__":
    from benchmarking.backends.backends import ElasticsearchBackend, QdrantBackend

    print("=" * 70)
    print("ELASTICSEARCH WRITE BENCHMARKS")
    print("=" * 70)

    es_backend = ElasticsearchBackend(PARQUET_FILE)
    es_backend.connect()

    if not es_backend.health_check():
        print("ERROR: Could not connect to Elasticsearch")
        exit(1)

    es_results = benchmark_write(
        es_backend,
        "elasticsearch",
        "bench_write",
        INDEX_SCHEMA_ES,
        batch_sizes=[100, 500, 1000]
    )

    es_backend.disconnect()

    print("\n" + "=" * 70)
    print("QDRANT WRITE BENCHMARKS")
    print("=" * 70)

    qdrant_backend = QdrantBackend(PARQUET_FILE)
    qdrant_backend.connect()

    if not qdrant_backend.health_check():
        print("ERROR: Could not connect to Qdrant")
        exit(1)

    qdrant_results = benchmark_write(
        qdrant_backend,
        "qdrant",
        "bench_write",
        INDEX_SCHEMA_QDRANT,
        batch_sizes=[100, 500, 1000]
    )

    qdrant_backend.disconnect()

    print("\n" + "=" * 70)
    print("WRITE WORKLOAD SUMMARY")
    print("=" * 70)

    for batch_size in [100, 500, 1000]:
        es_result = es_results[batch_size]
        qdrant_result = qdrant_results[batch_size]

        print(f"\nBatch Size: {batch_size}")
        print(f"  Elasticsearch Throughput: {es_result.throughput_ops_per_sec:.2f} ops/sec")
        print(f"  Qdrant Throughput: {qdrant_result.throughput_ops_per_sec:.2f} ops/sec")

        if qdrant_result.throughput_ops_per_sec > 0:
            speedup = es_result.throughput_ops_per_sec / qdrant_result.throughput_ops_per_sec
            print(f"  Speedup: {speedup:.2f}x")