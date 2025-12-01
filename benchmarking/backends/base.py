from typing import List, Dict, Any, Generator

from abc import ABC, abstractmethod

import polars as pl

class SearchBackend(ABC):
    """
    Abstract base class for search backends
    """

    def __init__(self, parquet_file: str):
        self.parquet_file = parquet_file
        self.df = pl.read_parquet(parquet_file)

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the backend
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close connection to the backend
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if backend is healthy and connected
        """
        pass

    @abstractmethod
    def reset_index(self, index_name: str) -> None:
        """
        Delete and recreate index/collection
        """
        pass

    @abstractmethod
    def create_index(self, index_name: str, schema: Dict[str, Any]) -> None:
        """
        Create index with specified schema
        """
        pass

    @abstractmethod
    def index_documents(self, index_name: str, documents: Generator, batch_size: int = 500) -> int:
        """
        Index documents and return number successfully indexed

        Args:
            index_name: Name of index/collection
            documents: Generator yielding document dicts
            batch_size: Size of batches for bulk operations

        Returns:
            Number of successfully indexed documents
        """
        pass

    @abstractmethod
    def get_doc_count(self, index_name: str) -> int:
        """
        Get total number of documents in index
        """
        pass

    @abstractmethod
    def lexical_search(self, index_name: str, query: str, limit: int = 10) -> List[Dict]:
        """
        Perform lexical/textual search

        Args:
            index_name: Name of index/collection
            query: Text query
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        pass


    @abstractmethod
    def vector_search(self, index_name: str, vector: List[float], limit: int = 10) -> List[Dict]:
        """
        Perform vector/ANN search

        Args:
            index_name: Name of index/collection
            vector: Query vector
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        pass

    @abstractmethod
    def hybrid_search(self, index_name: str, query: str, vector: List[float], limit: int = 10) -> List[Dict]:
        """
        Perform hybrid lexical + vector search

        Args:
            index_name: Name of index/collection
            query: Text query
            vector: Query vector
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        pass