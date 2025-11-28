from typing import List, Dict, Any, Generator, Optional, Union
import os

from benchmarking.backends.base import SearchBackend


class ElasticsearchBackend(SearchBackend):
    """
    Elasticsearch implementation
    """

    def __init__(self, parquet_file: str, host: str = None, api_key: str = None, config=None):
        super().__init__(parquet_file)

        # Support both direct parameters and BackendConfig objects
        if config is not None:
            self.host = config.get("host", "http://localhost:9200")
            self.api_key = config.get("api_key")
            # Check environment variable if api_key not set in config
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
            info = self.client.info()
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
                        "fields": ["title", "text"],
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
                        "embedding": {
                            "vector": list(vector),
                            "k": limit,
                        }
                    }
                },
                "size": limit,
            },
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]

    def hybrid_search(self, index_name: str, query: str, vector: List[float], limit: int = 10) -> List[Dict]:
        response = self.client.search(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title", "text"],
                                    "boost": 1.0,
                                }
                            },
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": vector,
                                        "k": limit,
                                    }
                                }
                            },
                        ]
                    }
                },
                "size": limit,
            },
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]


class QdrantBackend(SearchBackend):
    """
    Qdrant implementation
    """

    def __init__(self, parquet_file: str, host: str = None, port: int = None, url: str = None, api_key: str = None, config=None):
        super().__init__(parquet_file)

        # Support both direct parameters and BackendConfig objects
        if config is not None:
            self.host = config.get("host", "localhost")
            self.port = config.get("port", 6333)
            self.url = config.get("url")
            self.api_key = config.get("api_key")
        else:
            self.host = host or "localhost"
            self.port = port or 6333
            self.url = url
            self.api_key = api_key

        self.client = None

    def connect(self) -> None:
        from qdrant_client import QdrantClient

        # Support both local and remote connections
        if self.url:
            # Remote/managed Qdrant Cloud
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            # Local or self-hosted Qdrant
            self.client = QdrantClient(host=self.host, port=self.port)

    def disconnect(self) -> None:
        if self.client:
            self.client.close()

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"Qdrant health check failed: {e}")
            return False

    def reset_index(self, index_name: str) -> None:
        try:
            self.client.delete_collection(collection_name=index_name)
            print(f"Deleted existing collection: {index_name}")
        except Exception:
            print(f"No existing collection to delete: {index_name}")

    def create_index(self, index_name: str, schema: Dict[str, Any]) -> None:
        from qdrant_client.models import Distance, VectorParams

        vector_size = schema.get("vector_size", 384)
        self.client.create_collection(
            collection_name=index_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection: {index_name}")

    def index_documents(self, index_name: str, documents: Generator, batch_size: int = 500) -> int:
        from qdrant_client.models import PointStruct

        points = []
        count = 0

        for doc in documents:
            point = PointStruct(
                id=doc["_id"],
                vector=doc["vector"],
                payload=doc["payload"],
            )
            points.append(point)

            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=index_name,
                    points=points,
                    wait=True,
                )
                count += len(points)
                points = []

        # insert remaining points
        if points:
            self.client.upsert(
                collection_name=index_name,
                points=points,
                wait=True,
            )
            count += len(points)

        print(f"Successfully indexed {count} documents")
        return count

    def get_doc_count(self, index_name: str) -> int:
        collection_info = self.client.get_collection(collection_name=index_name)
        return collection_info.points_count

    def lexical_search(self, index_name: str, query: str, limit: int = 10) -> List[Dict]:
        """
        Qdrant doesn't have native full-text search.
        This is a naive implementation that filters by text match in payload.
        For production use, integrate with a text search engine.
        """
        # this is a simplified implementation
        # in production, you'd want to integrate with a dedicated text search solution
        results = []
        try:
            # scroll through collection to find matches
            points, _ = self.client.scroll(
                collection_name=index_name,
                limit=limit * 10,  # get more to filter
            )

            for point in points:
                text = point.payload.get("text", "") or ""
                title = point.payload.get("title", "") or ""
                if query.lower() in text.lower() or query.lower() in title.lower():
                    results.append(point.payload)
                    if len(results) >= limit:
                        break
        except Exception as e:
            print(f"Lexical search failed: {e}")

        return results

    def vector_search(self, index_name: str, vector: List[float], limit: int = 10) -> List[Dict]:
        try:
            results = self.client.search(
                collection_name=index_name,
                query_vector=vector,
                limit=limit,
            )
            return [result.payload for result in results]
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

    def hybrid_search(self, index_name: str, query: str, vector: List[float], limit: int = 10) -> List[Dict]:
        """
        Hybrid search combining lexical and vector search.
        This is a naive implementation that merges results from both searches.
        """
        lexical_results = self.lexical_search(index_name, query, limit)
        vector_results = self.vector_search(index_name, vector, limit)

        # merge results, preferring vector results but including lexical matches
        seen_ids = set()
        merged = []

        for result in vector_results:
            # use some unique identifier - assumes 'id' or index position
            merged.append(result)
            seen_ids.add(id(result))

        for result in lexical_results:
            if id(result) not in seen_ids:
                merged.append(result)

            if len(merged) >= limit:
                break

        return merged[:limit]