import os
import polars as pl

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# configuration
PARQUET_FILE = "data/ecommerce_products_with_embeddings.parquet"
COLLECTION_NAME = "womens_clothing_reviews"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# check if connection is successful
try:
    info = client.get_collections()
    print(f"Connected to Qdrant: {len(info.collections)} collections found")
except Exception as e:
    print(f"Failed to connect to Qdrant: {e}")
    exit(1)

# delete existing collection if it exists
try:
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")
except Exception as _e:
    print(f"No existing collection to delete: {COLLECTION_NAME}")
    pass

# create collection with vector configuration
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print(f"Created collection: {COLLECTION_NAME}")

# read parquet file
df = pl.read_parquet(PARQUET_FILE)
print(f"Loaded {len(df)} records from {PARQUET_FILE}")

# prepare points for bulk indexing
def generate_points():
    for idx, row in enumerate(df.iter_rows(named=True)):
        point = PointStruct(
            id=idx,
            vector=row["embedding"],
            payload={
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
                "umap2": row["umap2"],
            },
        )
        yield point

# bulk upsert points
print("Indexing documents...")
points = list(generate_points())
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True,
)
print(f"Successfully indexed {len(points)} documents")

# get collection stats
collection_info = client.get_collection(collection_name=COLLECTION_NAME)
doc_count = collection_info.points_count
print(f"Total documents in collection: {doc_count}")