# Configuration Guide

The benchmarking suite uses YAML configuration files to specify backend connections and workload parameters.

## Quick Start

When you first run the benchmarks, if `config.yaml` doesn't exist, it will automatically create a default configuration. Simply run:

```bash
python runner.py
```

A default `config.yaml` will be created that points to local instances. Update it with your backend details.

## Configuration Structure

### Data Configuration

```yaml
data:
  parquet_file: "data/ecommerce_products_with_embeddings.parquet"
```

Specify the path to your parquet file with the data to index.

### Output Configuration

```yaml
output:
  results_dir: "benchmark_results"
```

Directory where JSON and markdown reports will be saved.

### Backend Configuration

#### Elasticsearch

**Local Docker:**
```yaml
backends:
  elasticsearch:
    enabled: true
    host: "http://localhost:9200"
    api_key: null  # Or set via ES_LOCAL_API_KEY environment variable
```

**Elastic Cloud:**
```yaml
backends:
  elasticsearch:
    enabled: true
    host: "https://your-deployment.es.us-west2.aws.elastic-cloud.com:9243"
    api_key: "your-api-key"
```

**Self-hosted with Authentication:**
```yaml
backends:
  elasticsearch:
    enabled: true
    host: "https://your-elasticsearch-server.com:9200"
    api_key: "your-api-key"
```

#### Qdrant

**Local Docker or Self-hosted:**
```yaml
backends:
  qdrant:
    enabled: true
    host: "localhost"
    port: 6333
```

**Qdrant Cloud (Managed):**
```yaml
backends:
  qdrant:
    enabled: true
    url: "https://your-instance.qdrant.io"
    api_key: "your-api-key"
```

**Remote Self-hosted:**
```yaml
backends:
  qdrant:
    enabled: true
    host: "your-qdrant-server.com"
    port: 6333
```

### Workload Configuration

```yaml
workloads:
  write:
    # Batch sizes to test during indexing
    batch_sizes: [100, 500, 1000]

  query:
    # Number of queries to sample and execute
    num_queries: 100
    # Number of results to return per query
    result_limit: 10
```

## Environment Variables

You can also set configuration via environment variables:

- `CONFIG_PATH`: Path to configuration file (defaults to `config.yaml`)
- `ES_LOCAL_API_KEY`: Elasticsearch API key (if not in config)

Example:
```bash
export CONFIG_PATH=/path/to/custom_config.yaml
export ES_LOCAL_API_KEY=your-api-key
python runner.py
```

## Disable Backends

Set `enabled: false` to skip a backend:

```yaml
backends:
  elasticsearch:
    enabled: true
  qdrant:
    enabled: false  # Skip Qdrant benchmarks
```

## Complete Example Configuration

```yaml
data:
  parquet_file: "data/ecommerce_products_with_embeddings.parquet"

output:
  results_dir: "benchmark_results"

backends:
  elasticsearch:
    enabled: true
    host: "http://localhost:9200"
    api_key: null

  qdrant:
    enabled: true
    host: "localhost"
    port: 6333

workloads:
  write:
    batch_sizes: [100, 500, 1000]

  query:
    num_queries: 100
    result_limit: 10
```

## Connecting to Different Deployments

### Docker Compose Setup

If running both services locally via docker-compose:

```yaml
backends:
  elasticsearch:
    enabled: true
    host: "http://elasticsearch:9200"
    api_key: null

  qdrant:
    enabled: true
    host: "qdrant"
    port: 6333
```

### Kubernetes

For Kubernetes deployments, use the service names:

```yaml
backends:
  elasticsearch:
    enabled: true
    host: "http://elasticsearch-service:9200"
    api_key: "your-api-key"

  qdrant:
    enabled: true
    host: "qdrant-service"
    port: 6333
```

### Mixed Environment

Run Elasticsearch locally and Qdrant in the cloud:

```yaml
backends:
  elasticsearch:
    enabled: true
    host: "http://localhost:9200"
    api_key: null

  qdrant:
    enabled: true
    url: "https://your-qdrant-cloud.qdrant.io"
    api_key: "your-api-key"
```
