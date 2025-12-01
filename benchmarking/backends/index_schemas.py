INDEX_SCHEMA_ES = {
    "mappings": {
        "properties": {
            "main_category": {"type": "keyword"},
            "title": {"type": "text"},
            "average_rating": {"type": "float"},
            "rating_number": {"type": "integer"},
            "features": {"type": "text"},
            "description": {"type": "text"},
            "price": {"type": "float"},
            "store": {"type": "keyword"},
            "categories": {"type": "keyword"},
            "brand": {"type": "keyword"},
            "manufacturer": {"type": "keyword"},
            "brand_name": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "index_options": {
                    "type": "hnsw"
                },
                "similarity": "cosine"
            }
        }
    }
}

INDEX_SCHEMA_QDRANT = {
    "vector_size": 384,
}