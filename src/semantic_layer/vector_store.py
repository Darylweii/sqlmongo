"""
Vector Store implementations for the Semantic Layer.

This module provides the abstract base class and concrete implementations
for synonym vector storage, supporting semantic similarity search.

Requirements:
- 1.5: Support multiple vector storage backends (FAISS local, Milvus distributed)
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.semantic_layer.embedding import DashScopeEmbedding
from src.semantic_layer.exceptions import VectorStoreError
from src.semantic_layer.schema import DataTypeDefinition

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Search result from vector store.
    
    Attributes:
        type_id: Data type identifier (e.g., "ep", "i")
        synonym: The matched synonym text
        score: Similarity score (0.0 to 1.0, higher is more similar)
        data_type: Complete DataTypeDefinition for the matched type
    """
    type_id: str
    synonym: str
    score: float
    data_type: DataTypeDefinition


class SynonymVectorStore(ABC):
    """
    Abstract base class for synonym vector storage.
    
    Defines the interface for vector stores that support:
    - Initialization with data type definitions
    - Semantic similarity search
    - Index rebuilding for hot reload
    
    Implementations must handle:
    - Vector storage and retrieval
    - Similarity computation
    - Index persistence (if applicable)
    """
    
    @abstractmethod
    def initialize(self, data_types: List[DataTypeDefinition], embedding: DashScopeEmbedding) -> None:
        """
        Initialize the vector store with data type definitions.
        
        Loads all synonyms from the provided data types, vectorizes them
        using the embedding model, and builds the search index.
        
        Args:
            data_types: List of data type definitions to index
            embedding: Embedding model for vectorization
        
        Raises:
            VectorStoreError: If initialization fails
        """
        pass
    
    @abstractmethod
    def search(self, query: str, embedding: DashScopeEmbedding, top_k: int = 10) -> List[SearchResult]:
        """
        Search for semantically similar data types.
        
        Vectorizes the query and finds the most similar synonyms
        in the vector store.
        
        Args:
            query: Search query text
            embedding: Embedding model for query vectorization
            top_k: Maximum number of results to return
        
        Returns:
            List of SearchResult ordered by similarity (highest first)
        
        Raises:
            VectorStoreError: If search fails
        """
        pass
    
    @abstractmethod
    def rebuild_index(self, data_types: List[DataTypeDefinition], embedding: DashScopeEmbedding) -> None:
        """
        Rebuild the search index with new data types.
        
        Used for hot reload when configuration changes.
        Should be atomic - either fully succeeds or keeps old index.
        
        Args:
            data_types: New list of data type definitions
            embedding: Embedding model for vectorization
        
        Raises:
            VectorStoreError: If rebuild fails
        """
        pass
    
    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the vector store has been initialized."""
        pass
    
    @property
    @abstractmethod
    def entry_count(self) -> int:
        """Get the number of indexed entries."""
        pass



@dataclass
class VectorIndexEntry:
    """
    Internal entry in the vector index.
    
    Attributes:
        id: Index position
        type_id: Data type identifier
        synonym: Synonym text
        data_type: Reference to the full DataTypeDefinition
    """
    id: int
    type_id: str
    synonym: str
    data_type: DataTypeDefinition


class FAISSVectorStore(SynonymVectorStore):
    """
    FAISS-based local vector store implementation.
    
    Uses Facebook AI Similarity Search (FAISS) for efficient
    similarity search on local machine.
    
    Features:
    - Inner product similarity (normalized vectors = cosine similarity)
    - Index persistence to disk
    - Atomic index rebuilding
    
    Attributes:
        index_path: Path to save/load the FAISS index
        dimensions: Vector dimensions (must match embedding model)
    """
    
    def __init__(
        self,
        index_path: str = "data/semantic_layer.faiss",
        dimensions: int = 1024,
    ):
        """
        Initialize FAISSVectorStore.
        
        Args:
            index_path: Path to save/load the index file
            dimensions: Vector dimensions
        """
        self.index_path = index_path
        self.dimensions = dimensions
        self._index: Optional[object] = None  # faiss.IndexFlatIP
        self._entries: List[VectorIndexEntry] = []
        self._data_types_map: dict[str, DataTypeDefinition] = {}
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if the vector store has been initialized."""
        return self._initialized
    
    @property
    def entry_count(self) -> int:
        """Get the number of indexed entries."""
        return len(self._entries)
    
    def initialize(self, data_types: List[DataTypeDefinition], embedding: DashScopeEmbedding) -> None:
        """
        Initialize the vector store with data type definitions.
        
        Attempts to load existing index from disk first.
        If not found or invalid, builds a new index.
        
        Args:
            data_types: List of data type definitions to index
            embedding: Embedding model for vectorization
        
        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            import faiss
        except ImportError:
            raise VectorStoreError(
                "FAISS is not installed. Install with: pip install faiss-cpu",
                details={"package": "faiss-cpu"}
            )
        
        # Build data types map
        self._data_types_map = {dt.id: dt for dt in data_types}
        
        # Try to load existing index
        if self._try_load_index():
            logger.info(f"Loaded existing FAISS index with {self.entry_count} entries")
            self._initialized = True
            return
        
        # Build new index
        self._build_index(data_types, embedding)
        self._initialized = True
    
    def _try_load_index(self) -> bool:
        """
        Try to load existing index from disk.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            import faiss
            
            index_file = Path(self.index_path)
            metadata_file = Path(self.index_path + ".meta")
            
            if not index_file.exists() or not metadata_file.exists():
                return False
            
            # Load FAISS index
            self._index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            
            # Reconstruct entries with current data types
            self._entries = []
            for entry_data in metadata.get("entries", []):
                type_id = entry_data["type_id"]
                if type_id in self._data_types_map:
                    self._entries.append(VectorIndexEntry(
                        id=entry_data["id"],
                        type_id=type_id,
                        synonym=entry_data["synonym"],
                        data_type=self._data_types_map[type_id],
                    ))
            
            # Validate index dimensions
            if self._index.d != self.dimensions:
                logger.warning(
                    f"Index dimension mismatch: expected {self.dimensions}, got {self._index.d}"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            return False
    
    def _build_index(self, data_types: List[DataTypeDefinition], embedding: DashScopeEmbedding) -> None:
        """
        Build a new FAISS index from data types.
        
        Args:
            data_types: Data type definitions to index
            embedding: Embedding model
        
        Raises:
            VectorStoreError: If index building fails
        """
        try:
            import faiss
            
            # Collect all synonyms
            all_texts: List[str] = []
            entries: List[VectorIndexEntry] = []
            
            for dt in data_types:
                for term in dt.get_all_searchable_terms():
                    entries.append(VectorIndexEntry(
                        id=len(entries),
                        type_id=dt.id,
                        synonym=term,
                        data_type=dt,
                    ))
                    all_texts.append(term)
            
            if not all_texts:
                logger.warning("No synonyms to index")
                self._index = faiss.IndexFlatIP(self.dimensions)
                self._entries = []
                return
            
            logger.info(f"Vectorizing {len(all_texts)} synonyms...")
            
            # Vectorize all texts
            vectors = embedding.embed_texts(all_texts)
            
            # Convert to numpy array and normalize for cosine similarity
            vectors_np = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors_np)
            
            # Create FAISS index (Inner Product = Cosine Similarity for normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimensions)
            self._index.add(vectors_np)
            
            self._entries = entries
            
            # Save index to disk
            self._save_index()
            
            logger.info(f"Built FAISS index with {len(entries)} entries")
            
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Failed to build FAISS index: {e}",
                details={"error_type": type(e).__name__}
            )
    
    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        try:
            import faiss
            
            # Ensure directory exists
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self._index, self.index_path)
            
            # Save metadata
            metadata = {
                "entries": [
                    {
                        "id": e.id,
                        "type_id": e.type_id,
                        "synonym": e.synonym,
                    }
                    for e in self._entries
                ],
                "dimensions": self.dimensions,
            }
            
            with open(self.index_path + ".meta", "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved FAISS index to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def search(self, query: str, embedding: DashScopeEmbedding, top_k: int = 10) -> List[SearchResult]:
        """
        Search for semantically similar data types.
        
        Args:
            query: Search query text
            embedding: Embedding model for query vectorization
            top_k: Maximum number of results to return
        
        Returns:
            List of SearchResult ordered by similarity (highest first)
        
        Raises:
            VectorStoreError: If search fails
        """
        if not self._initialized:
            raise VectorStoreError("Vector store not initialized")
        
        if not self._entries:
            return []
        
        try:
            import faiss
            
            # Vectorize query
            query_vector = embedding.embed_text(query)
            query_np = np.array([query_vector], dtype=np.float32)
            faiss.normalize_L2(query_np)
            
            # Search
            k = min(top_k, len(self._entries))
            scores, indices = self._index.search(query_np, k)
            
            # Build results
            results: List[SearchResult] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._entries):
                    continue
                
                entry = self._entries[idx]
                results.append(SearchResult(
                    type_id=entry.type_id,
                    synonym=entry.synonym,
                    score=float(score),  # Inner product score (cosine similarity for normalized)
                    data_type=entry.data_type,
                ))
            
            return results
            
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Search failed: {e}",
                details={"query": query, "error_type": type(e).__name__}
            )
    
    def rebuild_index(self, data_types: List[DataTypeDefinition], embedding: DashScopeEmbedding) -> None:
        """
        Rebuild the search index with new data types.
        
        Atomic operation - keeps old index if rebuild fails.
        
        Args:
            data_types: New list of data type definitions
            embedding: Embedding model for vectorization
        
        Raises:
            VectorStoreError: If rebuild fails
        """
        # Store old state for rollback
        old_index = self._index
        old_entries = self._entries
        old_data_types_map = self._data_types_map
        
        try:
            # Update data types map
            self._data_types_map = {dt.id: dt for dt in data_types}
            
            # Build new index
            self._build_index(data_types, embedding)
            
            logger.info("Successfully rebuilt FAISS index")
            
        except Exception as e:
            # Rollback on failure
            self._index = old_index
            self._entries = old_entries
            self._data_types_map = old_data_types_map
            
            raise VectorStoreError(
                f"Failed to rebuild index: {e}",
                details={"error_type": type(e).__name__}
            )
    
    def clear(self) -> None:
        """Clear the index and remove persisted files."""
        try:
            import faiss
            
            self._index = faiss.IndexFlatIP(self.dimensions)
            self._entries = []
            self._initialized = False
            
            # Remove persisted files
            index_file = Path(self.index_path)
            metadata_file = Path(self.index_path + ".meta")
            
            if index_file.exists():
                index_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
                
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")


class MilvusVectorStore(SynonymVectorStore):
    """
    Milvus-based distributed vector store implementation.
    
    Uses Milvus for scalable, distributed similarity search.
    Suitable for production deployments with high availability requirements.
    
    Features:
    - Distributed vector storage
    - Automatic connection management
    - Graceful error handling with fallback support
    
    Attributes:
        host: Milvus server host
        port: Milvus server port
        collection_name: Name of the Milvus collection
        dimensions: Vector dimensions (must match embedding model)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "semantic_layer",
        dimensions: int = 1024,
    ):
        """
        Initialize MilvusVectorStore.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection
            dimensions: Vector dimensions
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimensions = dimensions
        self._entries: List[VectorIndexEntry] = []
        self._data_types_map: dict[str, DataTypeDefinition] = {}
        self._initialized = False
        self._collection = None
    
    @property
    def is_initialized(self) -> bool:
        """Check if the vector store has been initialized."""
        return self._initialized
    
    @property
    def entry_count(self) -> int:
        """Get the number of indexed entries."""
        return len(self._entries)
    
    def _connect(self) -> None:
        """
        Connect to Milvus server.
        
        Raises:
            VectorStoreError: If connection fails
        """
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            
        except ImportError:
            raise VectorStoreError(
                "pymilvus is not installed. Install with: pip install pymilvus",
                details={"package": "pymilvus"}
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to connect to Milvus: {e}",
                details={"host": self.host, "port": self.port, "error_type": type(e).__name__}
            )
    
    def _create_collection(self) -> None:
        """
        Create or get the Milvus collection.
        
        Raises:
            VectorStoreError: If collection creation fails
        """
        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
                self._collection.load()
                logger.info(f"Loaded existing collection: {self.collection_name}")
                return
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="type_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="synonym", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimensions),
            ]
            schema = CollectionSchema(fields=fields, description="Semantic layer synonyms")
            
            # Create collection
            self._collection = Collection(
                name=self.collection_name,
                schema=schema,
            )
            
            # Create index for vector field
            index_params = {
                "metric_type": "IP",  # Inner Product (cosine similarity for normalized vectors)
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            self._collection.create_index(field_name="embedding", index_params=index_params)
            self._collection.load()
            
            logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to create/load collection: {e}",
                details={"collection": self.collection_name, "error_type": type(e).__name__}
            )
    
    def initialize(self, data_types: List[DataTypeDefinition], embedding: DashScopeEmbedding) -> None:
        """
        Initialize the vector store with data type definitions.
        
        Args:
            data_types: List of data type definitions to index
            embedding: Embedding model for vectorization
        
        Raises:
            VectorStoreError: If initialization fails
        """
        # Build data types map
        self._data_types_map = {dt.id: dt for dt in data_types}
        
        # Connect to Milvus
        self._connect()
        
        # Create or load collection
        self._create_collection()
        
        # Check if collection already has data
        if self._collection.num_entities > 0:
            logger.info(f"Collection already has {self._collection.num_entities} entities")
            self._load_entries_from_collection()
            self._initialized = True
            return
        
        # Build index with new data
        self._build_index(data_types, embedding)
        self._initialized = True
    
    def _load_entries_from_collection(self) -> None:
        """Load entry metadata from existing collection."""
        try:
            # Query all entries
            results = self._collection.query(
                expr="id >= 0",
                output_fields=["id", "type_id", "synonym"],
            )
            
            self._entries = []
            for item in results:
                type_id = item["type_id"]
                if type_id in self._data_types_map:
                    self._entries.append(VectorIndexEntry(
                        id=item["id"],
                        type_id=type_id,
                        synonym=item["synonym"],
                        data_type=self._data_types_map[type_id],
                    ))
            
            logger.info(f"Loaded {len(self._entries)} entries from collection")
            
        except Exception as e:
            logger.warning(f"Failed to load entries from collection: {e}")
            self._entries = []
    
    def _build_index(self, data_types: List[DataTypeDefinition], embedding: DashScopeEmbedding) -> None:
        """
        Build index by inserting data into Milvus.
        
        Args:
            data_types: Data type definitions to index
            embedding: Embedding model
        
        Raises:
            VectorStoreError: If index building fails
        """
        try:
            import numpy as np
            
            # Collect all synonyms
            all_texts: List[str] = []
            type_ids: List[str] = []
            
            for dt in data_types:
                for term in dt.get_all_searchable_terms():
                    all_texts.append(term)
                    type_ids.append(dt.id)
            
            if not all_texts:
                logger.warning("No synonyms to index")
                self._entries = []
                return
            
            logger.info(f"Vectorizing {len(all_texts)} synonyms for Milvus...")
            
            # Vectorize all texts
            vectors = embedding.embed_texts(all_texts)
            
            # Normalize vectors for cosine similarity
            vectors_np = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
            vectors_np = vectors_np / norms
            
            # Insert into Milvus
            entities = [
                type_ids,  # type_id
                all_texts,  # synonym
                vectors_np.tolist(),  # embedding
            ]
            
            insert_result = self._collection.insert(entities)
            self._collection.flush()
            
            # Build entries list
            self._entries = []
            for i, (type_id, synonym) in enumerate(zip(type_ids, all_texts)):
                self._entries.append(VectorIndexEntry(
                    id=insert_result.primary_keys[i] if hasattr(insert_result, 'primary_keys') else i,
                    type_id=type_id,
                    synonym=synonym,
                    data_type=self._data_types_map[type_id],
                ))
            
            logger.info(f"Inserted {len(self._entries)} entries into Milvus")
            
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Failed to build Milvus index: {e}",
                details={"error_type": type(e).__name__}
            )
    
    def search(self, query: str, embedding: DashScopeEmbedding, top_k: int = 10) -> List[SearchResult]:
        """
        Search for semantically similar data types.
        
        Args:
            query: Search query text
            embedding: Embedding model for query vectorization
            top_k: Maximum number of results to return
        
        Returns:
            List of SearchResult ordered by similarity (highest first)
        
        Raises:
            VectorStoreError: If search fails
        """
        if not self._initialized:
            raise VectorStoreError("Vector store not initialized")
        
        if not self._entries:
            return []
        
        try:
            import numpy as np
            
            # Vectorize query
            query_vector = embedding.embed_text(query)
            query_np = np.array([query_vector], dtype=np.float32)
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(query_np)
            query_np = query_np / norm
            
            # Search in Milvus
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10},
            }
            
            results = self._collection.search(
                data=query_np.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["type_id", "synonym"],
            )
            
            # Build results
            search_results: List[SearchResult] = []
            for hits in results:
                for hit in hits:
                    type_id = hit.entity.get("type_id")
                    synonym = hit.entity.get("synonym")
                    
                    if type_id in self._data_types_map:
                        search_results.append(SearchResult(
                            type_id=type_id,
                            synonym=synonym,
                            score=float(hit.score),
                            data_type=self._data_types_map[type_id],
                        ))
            
            return search_results
            
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"Milvus search failed: {e}",
                details={"query": query, "error_type": type(e).__name__}
            )
    
    def rebuild_index(self, data_types: List[DataTypeDefinition], embedding: DashScopeEmbedding) -> None:
        """
        Rebuild the search index with new data types.
        
        Drops and recreates the collection with new data.
        
        Args:
            data_types: New list of data type definitions
            embedding: Embedding model for vectorization
        
        Raises:
            VectorStoreError: If rebuild fails
        """
        try:
            from pymilvus import utility
            
            # Update data types map
            self._data_types_map = {dt.id: dt for dt in data_types}
            
            # Drop existing collection
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped collection: {self.collection_name}")
            
            # Recreate collection and build index
            self._create_collection()
            self._build_index(data_types, embedding)
            
            logger.info("Successfully rebuilt Milvus index")
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to rebuild Milvus index: {e}",
                details={"error_type": type(e).__name__}
            )
    
    def close(self) -> None:
        """Close the Milvus connection."""
        try:
            from pymilvus import connections
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {e}")


__all__ = ["SynonymVectorStore", "FAISSVectorStore", "MilvusVectorStore", "SearchResult", "VectorIndexEntry"]
