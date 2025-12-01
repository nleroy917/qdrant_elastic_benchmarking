import json
import os

from datetime import datetime
from pathlib import Path
from typing import Dict

from benchmarking.backends.backends import ElasticsearchBackend, QdrantBackend
from benchmarking.benchmarks.write import benchmark_write, INDEX_SCHEMA_ES, INDEX_SCHEMA_QDRANT
from benchmarking.benchmarks.query import run_query_benchmarks
from config import BenchmarkConfig, load_config, create_default_config

# try to load config, create default if not found
try:
    config = load_config()
except FileNotFoundError:
    print("Config file not found. Creating default configuration...")
    create_default_config()
    print("Please configure config.yaml and run again.")
    exit(1)

class BenchmarkRunner:
    """
    Orchestrates benchmarking and reporting
    """

    def __init__(self, benchmark_config: BenchmarkConfig):
        self.config = benchmark_config
        self.results_dir = benchmark_config.results_dir
        self.results_dir.mkdir(exist_ok=True)
        self.all_results: Dict[str, Dict] = {}
        self.timestamp = datetime.now().isoformat()

    def run_all_benchmarks(self) -> None:
        """
        Run all benchmarks for all backends
        """
        print("\n" + "=" * 80)
        print("ELASTIC-QDRANT BENCHMARKING SUITE")
        print("=" * 80)
        print(f"Started: {self.timestamp}\n")

        # create backends with config
        backends = []
        es_backend = None
        qdrant_backend = None

        if self.config.elasticsearch.enabled:
            es_backend = ElasticsearchBackend(
                self.config.parquet_file,
                config=self.config.elasticsearch.config
            )
            backends.append(("elasticsearch", es_backend))

        if self.config.qdrant.enabled:
            qdrant_backend = QdrantBackend(
                self.config.parquet_file,
                config=self.config.qdrant.config
            )
            backends.append(("qdrant", qdrant_backend))

        # connect and health check
        print("Connecting to backends...")
        for name, backend in backends:
            backend.connect()
            if not backend.health_check():
                print(f"ERROR: Could not connect to {name}")
                return

        print("✓ All backends connected successfully\n")

        batch_sizes = self.config.get_batch_sizes()
        num_queries = self.config.get_num_queries()

        # run write benchmarks
        if es_backend:
            print("=" * 80)
            print("PHASE 1: WRITE WORKLOAD BENCHMARKS - ELASTICSEARCH")
            print("=" * 80)

            es_write_results = benchmark_write(
                es_backend,
                "elasticsearch",
                "bench_write",
                INDEX_SCHEMA_ES,
                batch_sizes=batch_sizes
            )
            self.all_results["elasticsearch_write"] = {
                k: v.to_dict() for k, v in es_write_results.items()
            }

        if qdrant_backend:
            print("=" * 80)
            print("PHASE 1: WRITE WORKLOAD BENCHMARKS - QDRANT")
            print("=" * 80)

            qdrant_write_results = benchmark_write(
                qdrant_backend,
                "qdrant",
                "bench_write",
                INDEX_SCHEMA_QDRANT,
                batch_sizes=batch_sizes
            )
            self.all_results["qdrant_write"] = {
                k: v.to_dict() for k, v in qdrant_write_results.items()
            }

        # run query benchmarks
        if es_backend:
            print("\n" + "=" * 80)
            print("PHASE 2: QUERY WORKLOAD BENCHMARKS - ELASTICSEARCH")
            print("=" * 80)

            es_query_results = run_query_benchmarks(
                es_backend,
                "elasticsearch",
                "bench_write",
                num_queries=num_queries
            )
            self.all_results["elasticsearch_query"] = {
                k: v.to_dict() for k, v in es_query_results.items()
            }

        if qdrant_backend:
            print("\n" + "=" * 80)
            print("PHASE 2: QUERY WORKLOAD BENCHMARKS - QDRANT")
            print("=" * 80)

            qdrant_query_results = run_query_benchmarks(
                qdrant_backend,
                "qdrant",
                "bench_write",
                num_queries=num_queries
            )
            self.all_results["qdrant_query"] = {
                k: v.to_dict() for k, v in qdrant_query_results.items()
            }

        # cleanup
        for _, backend in backends:
            backend.disconnect()

        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)

    def generate_reports(self) -> None:
        """
        Generate JSON and markdown reports
        """
        json_report_path = self.results_dir / f"results_{self.timestamp.replace(':', '-')}.json"
        with open(json_report_path, "w") as f:
            json.dump(self.all_results, f, indent=2)
        print(f"\nJSON Report: {json_report_path}")

        # generate markdown report
        md_report_path = self.results_dir / f"report_{self.timestamp.replace(':', '-')}.md"
        self._generate_markdown_report(md_report_path)
        print(f"Markdown Report: {md_report_path}")

    def _generate_markdown_report(self, report_path: Path) -> None:
        """Generate a markdown benchmark report"""
        lines = []
        lines.append("# Elasticsearch vs Qdrant Benchmark Report\n")
        lines.append(f"Generated: {self.timestamp}\n")

        # Write Workload Summary
        lines.append("## Write Workload\n")
        lines.append("| Batch Size | Engine | Throughput (ops/sec) | Duration (s) | Avg CPU (%) | Peak Mem (MB) |")
        lines.append("|------------|--------|----------------------|--------------|-------------|---------------|")

        for batch_size in [100, 500, 1000]:
            es_key = f"elasticsearch_write:{batch_size}"
            qdrant_key = f"qdrant_write:{batch_size}"

            # try to get results from nested dict
            es_result_dict = self.all_results.get("elasticsearch_write", {}).get(batch_size)
            qdrant_result_dict = self.all_results.get("qdrant_write", {}).get(batch_size)

            if es_result_dict:
                lines.append(
                    f"| {batch_size} | Elasticsearch | "
                    f"{es_result_dict.get('throughput_ops_per_sec', 0):.2f} | "
                    f"{es_result_dict.get('duration_seconds', 0):.2f} | "
                    f"{es_result_dict.get('avg_cpu_usage_percent', 0):.1f} | "
                    f"{es_result_dict.get('peak_memory_mb', 0):.1f} |"
                )

            if qdrant_result_dict:
                lines.append(
                    f"| {batch_size} | Qdrant | "
                    f"{qdrant_result_dict.get('throughput_ops_per_sec', 0):.2f} | "
                    f"{qdrant_result_dict.get('duration_seconds', 0):.2f} | "
                    f"{qdrant_result_dict.get('avg_cpu_usage_percent', 0):.1f} | "
                    f"{qdrant_result_dict.get('peak_memory_mb', 0):.1f} |"
                )

        # Query Workload Summary
        lines.append("\n## Query Workload\n")
        lines.append("| Query Type | Engine | Throughput (queries/sec) | Mean Latency (ms) | P99 Latency (ms) |")
        lines.append("|------------|--------|--------------------------|-------------------|------------------|")

        for query_type in ["lexical", "vector", "hybrid"]:
            es_result_dict = self.all_results.get("elasticsearch_query", {}).get(query_type)
            qdrant_result_dict = self.all_results.get("qdrant_query", {}).get(query_type)

            if es_result_dict:
                latency_metrics = es_result_dict.get("latency_metrics", {})
                lines.append(
                    f"| {query_type.capitalize()} | Elasticsearch | "
                    f"{es_result_dict.get('throughput_ops_per_sec', 0):.2f} | "
                    f"{latency_metrics.get('mean_ms', 0):.2f} | "
                    f"{latency_metrics.get('p99_ms', 0):.2f} |"
                )

            if qdrant_result_dict:
                latency_metrics = qdrant_result_dict.get("latency_metrics", {})
                lines.append(
                    f"| {query_type.capitalize()} | Qdrant | "
                    f"{qdrant_result_dict.get('throughput_ops_per_sec', 0):.2f} | "
                    f"{latency_metrics.get('mean_ms', 0):.2f} | "
                    f"{latency_metrics.get('p99_ms', 0):.2f} |"
                )

        # comparative analysis
        lines.append("\n## Comparative Analysis\n")
        lines.append("### Speedup (Elasticsearch vs Qdrant)\n")

        # write speedups
        lines.append("**Write Operations:**\n")
        for batch_size in [100, 500, 1000]:
            es_result = self.all_results.get("elasticsearch_write", {}).get(batch_size)
            qdrant_result = self.all_results.get("qdrant_write", {}).get(batch_size)

            if es_result and qdrant_result:
                es_throughput = es_result.get("throughput_ops_per_sec", 1)
                qdrant_throughput = qdrant_result.get("throughput_ops_per_sec", 1)
                if qdrant_throughput > 0:
                    speedup = es_throughput / qdrant_throughput
                    lines.append(
                        f"- Batch Size {batch_size}: "
                        f"Elasticsearch is {speedup:.2f}x "
                        f"{'faster' if speedup > 1 else 'slower'}\n"
                    )

        # query speedups
        lines.append("\n**Query Operations:**\n")
        for query_type in ["lexical", "vector", "hybrid"]:
            es_result = self.all_results.get("elasticsearch_query", {}).get(query_type)
            qdrant_result = self.all_results.get("qdrant_query", {}).get(query_type)

            if es_result and qdrant_result:
                es_throughput = es_result.get("throughput_ops_per_sec", 1)
                qdrant_throughput = qdrant_result.get("throughput_ops_per_sec", 1)
                if qdrant_throughput > 0:
                    speedup = es_throughput / qdrant_throughput
                    lines.append(
                        f"- {query_type.capitalize()}: "
                        f"Elasticsearch is {speedup:.2f}x "
                        f"{'faster' if speedup > 1 else 'slower'}\n"
                    )

        with open(report_path, "w") as f:
            f.writelines(lines)

def verify_environment() -> None:
    """
    Verify that required environment settings are correct
    """
    if not os.getenv("ES_LOCAL_API_KEY"):
        raise EnvironmentError("ES_LOCAL_API_KEY environment variable is required.")

if __name__ == "__main__":
    verify_environment()
    runner = BenchmarkRunner(config)
    try:
        runner.run_all_benchmarks()
        runner.generate_reports()
        print("\n✓ Benchmarking and reporting complete!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)