import time
import psutil
import json
import statistics

from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class LatencyMetrics:
    """
    Stores latency measurements
    """
    measurements: List[float]  # in milliseconds

    @property
    def mean(self) -> float:
        return statistics.mean(self.measurements) if self.measurements else 0

    @property
    def median(self) -> float:
        return statistics.median(self.measurements) if self.measurements else 0

    @property
    def p99(self) -> float:
        if not self.measurements or len(self.measurements) < 100:
            return max(self.measurements) if self.measurements else 0
        sorted_vals = sorted(self.measurements)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[idx]

    @property
    def p95(self) -> float:
        if not self.measurements or len(self.measurements) < 20:
            return max(self.measurements) if self.measurements else 0
        sorted_vals = sorted(self.measurements)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[idx]

    @property
    def min(self) -> float:
        return min(self.measurements) if self.measurements else 0

    @property
    def max(self) -> float:
        return max(self.measurements) if self.measurements else 0

@dataclass
class BenchmarkResult:
    """
    Stores complete benchmark results
    """
    name: str
    engine: str  # "elasticsearch" or "qdrant"
    workload_type: str  # "write", "lexical_query", "vector_query", "hybrid_query"
    duration_seconds: float
    total_operations: int
    latency_metrics: Dict[str, float]
    throughput_ops_per_sec: float
    avg_cpu_usage_percent: float
    peak_cpu_usage_percent: float
    avg_memory_mb: float
    peak_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

class CPUMonitor:
    """
    Monitors CPU and memory usage during benchmarks
    """

    def __init__(self):
        self.process = psutil.Process()
        self.cpu_measurements = []
        self.memory_measurements = []
        self.is_monitoring = False

    def start(self):
        """
        Start monitoring in a separate thread
        """
        self.is_monitoring = True
        self.cpu_measurements = []
        self.memory_measurements = []

    def record(self):
        """
        Record current CPU and memory usage
        """
        if self.is_monitoring:
            try:
                cpu_percent = self.process.cpu_percent(interval=None)
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.cpu_measurements.append(cpu_percent)
                self.memory_measurements.append(memory_mb)
            except Exception:
                pass

    def stop(self) -> Dict[str, float]:
        """
        Stop monitoring and return statistics
        """
        self.is_monitoring = False

        result = {
            "avg_cpu_percent": statistics.mean(self.cpu_measurements) if self.cpu_measurements else 0,
            "peak_cpu_percent": max(self.cpu_measurements) if self.cpu_measurements else 0,
            "avg_memory_mb": statistics.mean(self.memory_measurements) if self.memory_measurements else 0,
            "peak_memory_mb": max(self.memory_measurements) if self.memory_measurements else 0,
        }
        return result

class Timer:
    """
    Context manager for timing operations
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.latencies = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        latency_ms = (self.end_time - self.start_time) * 1000
        self.latencies.append(latency_ms)
        return False

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    def get_latency_metrics(self) -> Dict[str, float]:
        """Get summary statistics for all recorded latencies"""
        if not self.latencies:
            return {}

        return {
            "mean_ms": statistics.mean(self.latencies),
            "median_ms": statistics.median(self.latencies),
            "p95_ms": self._percentile(self.latencies, 95),
            "p99_ms": self._percentile(self.latencies, 99),
            "min_ms": min(self.latencies),
            "max_ms": max(self.latencies),
        }

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """
        Calculate percentile of data
        """
        if not data or len(data) < 100:
            return max(data) if data else 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * percentile / 100)
        return sorted_data[idx]