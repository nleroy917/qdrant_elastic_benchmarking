"""Query workload benchmarks for comparing search backends"""

import random
from typing import Dict, List, Tuple
from benchmarking.backends.backends import SearchBackend
from benchmarking.metrics import Timer, CPUMonitor, BenchmarkResult

PARQUET_FILE = "data/ecommerce_products_with_embeddings.parquet"


def sample_queries(backend: SearchBackend, num_queries: int = 100) -> Tuple[List[str], List[List[float]]]:
    """
    Sample random queries and vectors from the dataset

    Args:
        backend: SearchBackend instance
        num_queries: Number of queries to sample

    Returns:
        Tuple of (lexical_queries, vector_queries)
    """
    total_rows = len(backend.df)
    sample_indices = random.sample(range(total_rows), min(num_queries, total_rows))

    lexical_queries = []
    vector_queries = []

    for idx in sample_indices:
        row = backend.df[idx]
        # sample from text field for lexical queries
        text = row["text"]
        if isinstance(text, list):
            text = text[0] if text else None
        else:
            # Convert Polars Series to Python value
            text = text.item() if hasattr(text, 'item') else text

        if text:
            # Take first 3-5 words as query
            words = str(text).split()[:5]
            if words:
                lexical_queries.append(" ".join(words))

        # sample embeddings for vector queries
        embedding = row["embedding"]
        if isinstance(embedding, list):
            embedding = embedding[0] if embedding else None
        else:
            # Convert Polars Series to list
            if hasattr(embedding, 'to_list'):
                # Polars Series
                embedding = embedding.to_list()
            elif hasattr(embedding, 'tolist'):
                # NumPy array
                embedding = embedding.tolist()
            else:
                embedding = list(embedding) if embedding is not None else None

        if embedding is not None:
            vector_queries.append(embedding)

    return lexical_queries, vector_queries


def benchmark_lexical_search(
    backend: SearchBackend,
    backend_name: str,
    index_name: str,
    queries: List[str],
    result_limit: int = 10,
) -> BenchmarkResult:
    """
    Benchmark lexical (full-text) search performance

    Args:
        backend: SearchBackend instance
        backend_name: Name of the backend
        index_name: Name of index/collection
        queries: List of lexical queries to execute
        result_limit: Number of results per query

    Returns:
        BenchmarkResult with latency and throughput metrics
    """
    cpu_monitor = CPUMonitor()
    cpu_monitor.start()

    timer = Timer()

    for query in queries:
        with timer:
            backend.lexical_search(index_name, query, limit=result_limit)

    cpu_stats = cpu_monitor.stop()

    result = BenchmarkResult(
        name=f"{backend_name}_lexical_search",
        engine=backend_name,
        workload_type="lexical_query",
        duration_seconds=timer.elapsed_seconds,
        total_operations=len(queries),
        latency_metrics=timer.get_latency_metrics(),
        throughput_ops_per_sec=len(queries) / timer.elapsed_seconds if timer.elapsed_seconds > 0 else 0,
        avg_cpu_usage_percent=cpu_stats["avg_cpu_percent"],
        peak_cpu_usage_percent=cpu_stats["peak_cpu_percent"],
        avg_memory_mb=cpu_stats["avg_memory_mb"],
        peak_memory_mb=cpu_stats["peak_memory_mb"],
    )

    return result


def benchmark_vector_search(
    backend: SearchBackend,
    backend_name: str,
    index_name: str,
    vectors: List[List[float]],
    result_limit: int = 10,
) -> BenchmarkResult:
    """
    Benchmark vector (ANN) search performance

    Args:
        backend: SearchBackend instance
        backend_name: Name of the backend
        index_name: Name of index/collection
        vectors: List of query vectors
        result_limit: Number of results per query

    Returns:
        BenchmarkResult with latency and throughput metrics
    """
    cpu_monitor = CPUMonitor()
    cpu_monitor.start()

    timer = Timer()

    for vector in vectors:
        with timer:
            backend.vector_search(index_name, vector, limit=result_limit)

    cpu_stats = cpu_monitor.stop()

    result = BenchmarkResult(
        name=f"{backend_name}_vector_search",
        engine=backend_name,
        workload_type="vector_query",
        duration_seconds=timer.elapsed_seconds,
        total_operations=len(vectors),
        latency_metrics=timer.get_latency_metrics(),
        throughput_ops_per_sec=len(vectors) / timer.elapsed_seconds if timer.elapsed_seconds > 0 else 0,
        avg_cpu_usage_percent=cpu_stats["avg_cpu_percent"],
        peak_cpu_usage_percent=cpu_stats["peak_cpu_percent"],
        avg_memory_mb=cpu_stats["avg_memory_mb"],
        peak_memory_mb=cpu_stats["peak_memory_mb"],
    )

    return result


def benchmark_hybrid_search(
    backend: SearchBackend,
    backend_name: str,
    index_name: str,
    lexical_queries: List[str],
    vector_queries: List[List[float]],
    result_limit: int = 10,
) -> BenchmarkResult:
    """
    Benchmark hybrid (lexical + vector) search performance

    Args:
        backend: SearchBackend instance
        backend_name: Name of the backend
        index_name: Name of index/collection
        lexical_queries: List of lexical queries
        vector_queries: List of query vectors
        result_limit: Number of results per query

    Returns:
        BenchmarkResult with latency and throughput metrics
    """
    cpu_monitor = CPUMonitor()
    cpu_monitor.start()

    timer = Timer()

    # Pair up queries and vectors
    num_queries = min(len(lexical_queries), len(vector_queries))
    for i in range(num_queries):
        with timer:
            backend.hybrid_search(
                index_name,
                lexical_queries[i],
                vector_queries[i],
                limit=result_limit
            )

    cpu_stats = cpu_monitor.stop()

    result = BenchmarkResult(
        name=f"{backend_name}_hybrid_search",
        engine=backend_name,
        workload_type="hybrid_query",
        duration_seconds=timer.elapsed_seconds,
        total_operations=num_queries,
        latency_metrics=timer.get_latency_metrics(),
        throughput_ops_per_sec=num_queries / timer.elapsed_seconds if timer.elapsed_seconds > 0 else 0,
        avg_cpu_usage_percent=cpu_stats["avg_cpu_percent"],
        peak_cpu_usage_percent=cpu_stats["peak_cpu_percent"],
        avg_memory_mb=cpu_stats["avg_memory_mb"],
        peak_memory_mb=cpu_stats["peak_memory_mb"],
    )

    return result


def run_query_benchmarks(
    backend: SearchBackend,
    backend_name: str,
    index_name: str,
    num_queries: int = 100,
) -> Dict[str, BenchmarkResult]:
    """
    Run all query benchmarks for a backend

    Args:
        backend: SearchBackend instance
        backend_name: Name of the backend
        index_name: Name of index/collection
        num_queries: Number of queries to run

    Returns:
        Dictionary of benchmark results by query type
    """
    print(f"\nSampling {num_queries} queries from dataset...")
    lexical_queries, vector_queries = sample_queries(backend, num_queries)

    results = {}

    print(f"\n{backend_name.upper()} - LEXICAL SEARCH")
    print("-" * 50)
    lexical_result = benchmark_lexical_search(
        backend, backend_name, index_name, lexical_queries
    )
    results["lexical"] = lexical_result
    print(f"  Queries: {lexical_result.total_operations}")
    print(f"  Duration: {lexical_result.duration_seconds:.2f}s")
    print(f"  Throughput: {lexical_result.throughput_ops_per_sec:.2f} queries/sec")
    print(f"  Mean Latency: {lexical_result.latency_metrics.get('mean_ms', 0):.2f}ms")
    print(f"  P99 Latency: {lexical_result.latency_metrics.get('p99_ms', 0):.2f}ms")

    print(f"\n{backend_name.upper()} - VECTOR SEARCH")
    print("-" * 50)
    vector_result = benchmark_vector_search(
        backend, backend_name, index_name, vector_queries
    )
    results["vector"] = vector_result
    print(f"  Queries: {vector_result.total_operations}")
    print(f"  Duration: {vector_result.duration_seconds:.2f}s")
    print(f"  Throughput: {vector_result.throughput_ops_per_sec:.2f} queries/sec")
    print(f"  Mean Latency: {vector_result.latency_metrics.get('mean_ms', 0):.2f}ms")
    print(f"  P99 Latency: {vector_result.latency_metrics.get('p99_ms', 0):.2f}ms")

    print(f"\n{backend_name.upper()} - HYBRID SEARCH")
    print("-" * 50)
    hybrid_result = benchmark_hybrid_search(
        backend, backend_name, index_name, lexical_queries, vector_queries
    )
    results["hybrid"] = hybrid_result
    print(f"  Queries: {hybrid_result.total_operations}")
    print(f"  Duration: {hybrid_result.duration_seconds:.2f}s")
    print(f"  Throughput: {hybrid_result.throughput_ops_per_sec:.2f} queries/sec")
    print(f"  Mean Latency: {hybrid_result.latency_metrics.get('mean_ms', 0):.2f}ms")
    print(f"  P99 Latency: {hybrid_result.latency_metrics.get('p99_ms', 0):.2f}ms")

    return results


if __name__ == "__main__":
    from benchmarking.backends.backends import ElasticsearchBackend, QdrantBackend

    print("=" * 70)
    print("ELASTICSEARCH QUERY BENCHMARKS")
    print("=" * 70)

    es_backend = ElasticsearchBackend(PARQUET_FILE)
    es_backend.connect()

    if not es_backend.health_check():
        print("ERROR: Could not connect to Elasticsearch")
        exit(1)

    es_results = run_query_benchmarks(
        es_backend,
        "elasticsearch",
        "womens_clothing_reviews",
        num_queries=100
    )

    es_backend.disconnect()

    print("\n" + "=" * 70)
    print("QDRANT QUERY BENCHMARKS")
    print("=" * 70)

    qdrant_backend = QdrantBackend(PARQUET_FILE)
    qdrant_backend.connect()

    if not qdrant_backend.health_check():
        print("ERROR: Could not connect to Qdrant")
        exit(1)

    qdrant_results = run_query_benchmarks(
        qdrant_backend,
        "qdrant",
        "womens_clothing_reviews",
        num_queries=100
    )

    qdrant_backend.disconnect()

    print("\n" + "=" * 70)
    print("QUERY WORKLOAD SUMMARY")
    print("=" * 70)

    for query_type in ["lexical", "vector", "hybrid"]:
        es_result = es_results[query_type]
        qdrant_result = qdrant_results[query_type]

        print(f"\n{query_type.upper()} SEARCH:")
        print(f"  Elasticsearch: {es_result.throughput_ops_per_sec:.2f} queries/sec")
        print(f"  Qdrant: {qdrant_result.throughput_ops_per_sec:.2f} queries/sec")

        if qdrant_result.throughput_ops_per_sec > 0:
            speedup = es_result.throughput_ops_per_sec / qdrant_result.throughput_ops_per_sec
            print(f"  Speedup: {speedup:.2f}x")