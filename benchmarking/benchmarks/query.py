import random

from typing import Dict, List, Tuple

from benchmarking.backends.backends import SearchBackend
from benchmarking.metrics import Timer, CPUMonitor, BenchmarkResult


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
        text = row["title"]
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
            # convert polars series to list
            if hasattr(embedding, 'to_list'):
                # polars series
                embedding = embedding.to_list()
            elif hasattr(embedding, 'tolist'):
                # numpy array
                embedding = embedding.tolist()
            else:
                embedding = list(embedding) if embedding is not None else None

        if embedding is not None:
            vector_queries.append(embedding)

    return lexical_queries, vector_queries


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
    
    return results
