import os

import polars as pl

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# configuration
PARQUET_FILE = "data/ecommerce_products_with_embeddings.parquet"
INDEX_NAME = "womens_clothing_reviews"
ELASTIC_HOST = "http://localhost:9200"
API_KEY = os.getenv("ES_LOCAL_API_KEY")

es = Elasticsearch([ELASTIC_HOST], api_key=API_KEY)

# check if connection is successful
try:
    info = es.info()
    print(f"Connected to Elasticsearch: {info['version']['number']}")
except Exception as e:
    print(f"Failed to connect to Elasticsearch: {e}")
    exit(1)

index_mapping = {
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

# delete existing index if it exists
try:
    es.indices.delete(index=INDEX_NAME)
    print(f"Deleted existing index: {INDEX_NAME}")
except Exception as _e:
    print(f"No existing index to delete: {INDEX_NAME}")
    pass


# create index with mapping
es.indices.create(index=INDEX_NAME, body=index_mapping)
print(f"Created index: {INDEX_NAME}")

# read parquet file
df = pl.read_parquet(PARQUET_FILE)
print(f"Loaded {len(df)} records from {PARQUET_FILE}")

# prepare documents for bulk indexing
def generate_docs():
    for idx, row in enumerate(df.iter_rows(named=True)):
        doc = {
            "_index": INDEX_NAME,
            "_id": str(idx),
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

# bulk index documents
print("Indexing documents...")
success, failed = bulk(es, generate_docs(), chunk_size=500, raise_on_error=False)
print(f"Successfully indexed {success} documents, {failed} failed")

# get index stats
stats = es.indices.stats(index=INDEX_NAME)
doc_count = stats["indices"][INDEX_NAME]["primaries"]["docs"]["count"]
print(f"Total documents in index: {doc_count}")
