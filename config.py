import os

from pathlib import Path
from typing import Dict, Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")

class BackendConfig:
    """
    Configuration for a search backend
    """

    def __init__(self, backend_type: str, config_dict: Dict[str, Any]):
        self.backend_type = backend_type
        self.config = config_dict

    @property
    def type(self) -> str:
        return self.backend_type

    @property
    def enabled(self) -> bool:
        return self.config.get("enabled", True)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def __repr__(self) -> str:
        return f"BackendConfig({self.backend_type}, enabled={self.enabled})"


class BenchmarkConfig:
    """
    Main configuration for benchmarking suite
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict

    @property
    def parquet_file(self) -> str:
        return self.config.get("data", {}).get("parquet_file", "data/ecommerce_products_with_embeddings.parquet")

    @property
    def results_dir(self) -> Path:
        results_dir = self.config.get("output", {}).get("results_dir", "benchmark_results")
        return Path(results_dir)

    @property
    def elasticsearch(self) -> BackendConfig:
        es_config = self.config.get("backends", {}).get("elasticsearch", {})
        return BackendConfig("elasticsearch", es_config)

    @property
    def qdrant(self) -> BackendConfig:
        qdrant_config = self.config.get("backends", {}).get("qdrant", {})
        return BackendConfig("qdrant", qdrant_config)

    @property
    def write_workload(self) -> Dict[str, Any]:
        return self.config.get("workloads", {}).get("write", {})

    @property
    def query_workload(self) -> Dict[str, Any]:
        return self.config.get("workloads", {}).get("query", {})

    def get_batch_sizes(self) -> list:
        return self.write_workload.get("batch_sizes", [100, 500, 1000])

    def get_num_queries(self) -> int:
        return self.query_workload.get("num_queries", 100)

    def get_result_limit(self) -> int:
        return self.query_workload.get("result_limit", 10)


def load_config(config_path: Optional[Path] = None) -> BenchmarkConfig:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config.yaml file. If None, uses DEFAULT_CONFIG_PATH.
                    Can also be overridden with CONFIG_PATH environment variable.

    Returns:
        BenchmarkConfig instance
    """
    # check environment variable first
    env_config_path = os.getenv("CONFIG_PATH")
    if env_config_path:
        config_path = Path(env_config_path)
    elif config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create a config.yaml file or set CONFIG_PATH environment variable"
        )

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if not config_dict:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return BenchmarkConfig(config_dict)


def create_default_config(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """
    Create a default configuration file

    Args:
        config_path: Path where to save the default config
    """
    default_config = {
        "data": {
            "parquet_file": "data/ecommerce_products_with_embeddings.parquet",
        },
        "output": {
            "results_dir": "benchmark_results",
        },
        "backends": {
            "elasticsearch": {
                "enabled": True,
                "host": "http://localhost:9200",
                "api_key": None,  # Or set via ES_LOCAL_API_KEY environment variable
            },
            "qdrant": {
                "enabled": True,
                "host": "localhost",
                "port": 6333,
                # For remote/managed Qdrant, use:
                # "url": "https://your-qdrant-instance.com",
                # "api_key": "your-api-key",
            },
        },
        "workloads": {
            "write": {
                "batch_sizes": [100, 500, 1000],
            },
            "query": {
                "num_queries": 100,
                "result_limit": 10,
            },
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    print(f"Created default configuration at: {config_path}")
    print("Please update the configuration with your backend details.")
