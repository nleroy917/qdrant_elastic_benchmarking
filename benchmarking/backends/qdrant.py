from typing import List, Dict, Any, Generator

from fastembed import SparseTextEmbedding

from .base import SearchBackend

INDEX_SCHEMA_QDRANT = {
    "vector_size": 384,
}

smodel = SparseTextEmbedding("Qdrant/bm25")

class QdrantBackend(SearchBackend):
    """
    Qdrant implementation
    """

    def __init__(self, parquet_file: str, host: str = None, port: int = None, url: str = None, api_key: str = None, config=None):
        super().__init__(parquet_file)
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
        from qdrant_client.models import Distance, VectorParams, SparseVectorParams, Modifier

        vector_size = schema.get("vector_size", 384)
        self.client.create_collection(
            collection_name=index_name,
            vectors_config={
                "embedding": VectorParams(size=vector_size, distance=Distance.COSINE)
            },
            # we use sparse vectors for textual fields
            # to perform lexical/keyword search alongside vector search
            sparse_vectors_config={
                "bm25": SparseVectorParams(modifier=Modifier.IDF)
            }
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
    
    def lexical_search(self, index_name: str, query: str, limit = 10):
        from qdrant_client.models import Document
        response = self.client.query_points(
            collection_name=index_name,
            query=Document(
                text=query,
                model="Qdrant/bm25"
            ),
            using="bm25",
            limit=limit,
        )
        return [result.payload for result in response.points]

    def vector_search(self, index_name: str, vector: List[float], limit: int = 10) -> List[Dict]:
        try:
            results = self.client.query_points(
                collection_name=index_name,
                query=vector[0],
                using="embedding",
                limit=limit,
            )
            return [result.payload for result in results.points]
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []
    
    def hybrid_search(self, index_name: str, query: str, vector: List[float], limit: int = 10) -> List[Dict]:
        from qdrant_client.models import Document, Prefetch, FusionQuery, Fusion
        try:
            response = self.client.query_points(
                collection_name=index_name,
                prefetch = [
                    Prefetch(
                        query=vector[0],
                        using="embedding",
                        limit=100,
                    ),
                    Prefetch(
                        query=Document(
                            text=query,
                            model="Qdrant/bm25"
                        ),
                        using="bm25",
                        limit=100,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
            )
            return [result.payload for result in response.points]
        except Exception as e:
            print(f"Hybrid search failed: {e}")
            return []
    