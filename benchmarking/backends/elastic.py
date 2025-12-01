import os

from typing import List, Dict, Any, Generator

from .base import SearchBackend

INDEX_SCHEMA_ES = {
    "mappings": {
        "properties": {
            "features": {"type": "text"},
            "description": {"type": "text"},
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

class ElasticsearchBackend(SearchBackend):
    """
    Elasticsearch implementation
    """

    def __init__(self, parquet_file: str, host: str = None, api_key: str = None, config=None):
        super().__init__(parquet_file)

        # support both direct parameters and BackendConfig objects
        if config is not None:
            self.host = config.get("host", "http://localhost:9200")
            self.api_key = config.get("api_key")
            if not self.api_key:
                self.api_key = os.getenv("ES_LOCAL_API_KEY")
        else:
            self.host = host or "http://localhost:9200"
            self.api_key = api_key or os.getenv("ES_LOCAL_API_KEY")

        self.client = None

    def connect(self) -> None:
        from elasticsearch import Elasticsearch
        self.client = Elasticsearch([self.host], api_key=self.api_key)

    def disconnect(self) -> None:
        if self.client:
            self.client.close()

    def health_check(self) -> bool:
        try:
            _info = self.client.info()
            return True
        except Exception as e:
            print(f"Elasticsearch health check failed: {e}")
            return False

    def reset_index(self, index_name: str) -> None:
        try:
            self.client.indices.delete(index=index_name)
            print(f"Deleted existing index: {index_name}")
        except Exception:
            print(f"No existing index to delete: {index_name}")

    def create_index(self, index_name: str, schema: Dict[str, Any]) -> None:
        self.client.indices.create(index=index_name, body=schema)
        print(f"Created index: {index_name}")

    def index_documents(self, index_name: str, documents: Generator, batch_size: int = 500) -> int:
        from elasticsearch.helpers import bulk

        def doc_generator():
            for doc in documents:
                yield {
                    "_index": index_name,
                    "_id": doc["_id"],
                    "_source": doc["_source"],
                }

        success, failed = bulk(
            self.client,
            doc_generator(),
            chunk_size=batch_size,
            raise_on_error=False,
        )
        print(f"Successfully indexed {success} documents, {failed} failed")
        return success

    def get_doc_count(self, index_name: str) -> int:
        stats = self.client.indices.stats(index=index_name)
        return stats["indices"][index_name]["primaries"]["docs"]["count"]
    
    def lexical_search(self, index_name: str, query: str, limit: int = 10) -> List[Dict]:
        response = self.client.search(
            index=index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["description"]
                    }
                },
                "size": limit,
            },
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]

    def vector_search(self, index_name: str, vector: List[float], limit: int = 10) -> List[Dict]:
        response = self.client.search(
            index=index_name,
            body={
                "query": {
                    "knn": {
                        "field": "embedding",
                        "query_vector": vector[0],
                        "k": limit,
                    }
                },
                "size": limit,
            },
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]

    def hybrid_search(self, index_name, query, vector, limit = 10):
        response = self.client.search(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": query,
                                "fields": ["description"]
                            }
                        },
                        "should": {
                            "knn": {
                                "field": "embedding",
                                "query_vector": vector,
                                "k": limit,
                            }
                        }
                    }
                },
                "size": limit,
            },
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]