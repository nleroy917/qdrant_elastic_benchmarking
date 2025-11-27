# Elasticsearch vs Qdrant Benchmarking Suite

Comprehensive benchmarking suite to compare Elasticsearch and Qdrant search performance across write and query workloads.

## Quick Start

```bash
# first run creates default config
python runner.py

# edit config.yaml with your backend details
nvim config.yaml

# run benchmarks
python runner.py
```

## What It Benchmarks

- **Write workloads**: Indexing performance with different batch sizes (100, 500, 1000)
- **Query workloads**: Lexical search, vector search, and hybrid search performance

## Configuration

See [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md) for detailed configuration options.


## Output

Results are saved to `benchmark_results/` in JSON and markdown formats with:
- Throughput metrics (ops/sec, queries/sec)
- Latency percentiles (mean, p95, p99)
- CPU and memory usage
- Comparative speedup analysis
