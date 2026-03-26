"""
Semantic Layer main class for the AI Data Router Agent.

This module provides the SemanticLayer class that coordinates all semantic layer
components including VirtualSchemaManager, VectorStore, Embedding, and Rerank.

Requirements:
- 1.2: Return semantically closest data type for colloquial queries
- 3.1: Use vector search to find most relevant data types before intent parsing
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.semantic_layer.config import SemanticLayerConfig
from src.semantic_layer.embedding import DashScopeEmbedding
from src.semantic_layer.exceptions import (
    ConfigurationError,
    EmbeddingError,
    RerankError,
    SemanticLayerError,
    VectorStoreError,
)
from src.semantic_layer.rerank import DashScopeRerank, RerankResult
from src.semantic_layer.schema import DataTypeDefinition, VirtualSchemaManager
from src.semantic_layer.vector_store import (
    FAISSVectorStore,
    MilvusVectorStore,
    SearchResult,
    SynonymVectorStore,
)

# Import DATA_TYPE_PREFIXES for fallback exact matching
from src.router.collection_router import DATA_TYPE_PREFIXES

logger = logging.getLogger(__name__)


@dataclass
class SemanticSearchResult:
    """
    Complete result from semantic search operation.
    
    Attributes:
        query: Original query text
        candidates: List of candidate search results
        best_match: Best matching result (if any)
        confidence: Confidence score (0.0 to 1.0)
        fallback_used: Whether fallback to exact match was used
        processing_time_ms: Processing time in milliseconds
    """
    query: str
    candidates: List[SearchResult]
    best_match: Optional[SearchResult]
    confidence: float
    fallback_used: bool
    processing_time_ms: float


class SemanticLayer:
    """
    Semantic Layer main class - coordinates all semantic layer components.
    
    This class provides:
    - Initialization and coordination of VirtualSchemaManager, VectorStore, Embedding, Rerank
    - Data type resolution from colloquial queries
    - Collection prefix lookup with backward compatibility
    - Enhanced context generation for Intent Parser
    - Configuration hot reload support
    
    Attributes:
        config: SemanticLayerConfig instance
        schema_manager: VirtualSchemaManager for data type definitions
        vector_store: SynonymVectorStore for semantic search
        embedding: DashScopeEmbedding for text vectorization
        reranker: DashScopeRerank for result reranking
    
    Example:
        config = SemanticLayerConfig.from_env()
        semantic_layer = SemanticLayer(config)
        
        if semantic_layer.initialize():
            data_type, confidence, candidates = semantic_layer.resolve_data_type("吃电情况")
            print(f"Resolved: {data_type} (confidence: {confidence})")
    """
    
    def __init__(self, config: SemanticLayerConfig):
        """
        Initialize the SemanticLayer.
        
        Args:
            config: SemanticLayerConfig instance with all configuration
        """
        self.config = config
        self.schema_manager: Optional[VirtualSchemaManager] = None
        self.vector_store: Optional[SynonymVectorStore] = None
        self.embedding: Optional[DashScopeEmbedding] = None
        self.reranker: Optional[DashScopeRerank] = None
        self._initialized: bool = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if the semantic layer has been initialized."""
        return self._initialized
    
    def initialize(self) -> bool:
        """
        Initialize the semantic layer components.
        
        Initializes all components in order:
        1. VirtualSchemaManager - loads YAML configuration
        2. DashScopeEmbedding - creates embedding client
        3. DashScopeRerank - creates rerank client
        4. VectorStore - builds/loads vector index
        
        If hot reload is enabled, starts the background reload thread.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if not self.config.enabled:
            logger.info("Semantic layer is disabled")
            return False
        
        try:
            # Validate configuration
            is_valid, errors = self.config.validate()
            if not is_valid:
                logger.error(f"Invalid configuration: {errors}")
                return False
            
            # Initialize schema manager
            logger.info(f"Loading virtual schema from {self.config.config_path}")
            self.schema_manager = VirtualSchemaManager(self.config.config_path)
            try:
                self.schema_manager.load_config()
            except ConfigurationError as e:
                logger.warning(f"Failed to load config file, using defaults: {e}")
                # Continue without config file - will use empty data types
            
            # Initialize embedding client
            logger.info("Initializing DashScope embedding client")
            self.embedding = DashScopeEmbedding(
                api_key=self.config.dashscope_api_key,
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions,
            )
            
            # Initialize reranker client
            logger.info("Initializing DashScope rerank client")
            self.reranker = DashScopeRerank(
                api_key=self.config.dashscope_api_key,
                model=self.config.rerank_model,
            )
            
            # Initialize vector store
            logger.info(f"Initializing {self.config.vector_store_type} vector store")
            if self.config.vector_store_type == "milvus":
                self.vector_store = MilvusVectorStore(
                    host=self.config.milvus_host,
                    port=self.config.milvus_port,
                    collection_name=self.config.milvus_collection,
                    dimensions=self.config.embedding_dimensions,
                )
            else:
                self.vector_store = FAISSVectorStore(
                    index_path=self.config.faiss_index_path,
                    dimensions=self.config.embedding_dimensions,
                )
            
            # Initialize vector store with data types
            data_types = self.schema_manager.get_all_data_types()
            if data_types:
                self.vector_store.initialize(data_types, self.embedding)
                logger.info(f"Vector store initialized with {len(data_types)} data types")
            else:
                logger.warning("No data types loaded, vector store will be empty")
            
            # Start hot reload if enabled
            if self.config.hot_reload_enabled and self.schema_manager:
                self.schema_manager.start_hot_reload(self.config.hot_reload_interval_seconds)
            
            self._initialized = True
            logger.info("Semantic layer initialized successfully")
            return True
            
        except (EmbeddingError, VectorStoreError, ConfigurationError) as e:
            logger.error(f"Failed to initialize semantic layer: {e}")
            self._cleanup()
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing semantic layer: {e}")
            self._cleanup()
            return False
    
    def _cleanup(self) -> None:
        """Clean up resources on initialization failure."""
        if self.embedding:
            self.embedding.close()
            self.embedding = None
        if self.reranker:
            self.reranker.close()
            self.reranker = None
        if self.schema_manager:
            self.schema_manager.stop_hot_reload()
            self.schema_manager = None
        self.vector_store = None
        self._initialized = False
    
    def close(self) -> None:
        """Close all resources and stop background threads."""
        logger.info("Closing semantic layer")
        self._cleanup()
    
    def __enter__(self) -> "SemanticLayer":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def resolve_data_type(
        self,
        query: str,
        fallback_to_exact: bool = True,
    ) -> Tuple[Optional[str], float, List[SearchResult]]:
        """
        Resolve data type from a query using semantic search.
        
        Performs semantic search to find the most relevant data type for the query.
        If semantic search fails or returns no results, falls back to exact matching
        against DATA_TYPE_PREFIXES.
        
        Args:
            query: User query text (e.g., "吃电情况", "设备稳不稳定")
            fallback_to_exact: Whether to fall back to exact match if semantic search fails
        
        Returns:
            Tuple of (data_type_id, confidence, candidates):
            - data_type_id: Resolved data type ID (e.g., "ep", "qf") or None
            - confidence: Confidence score (0.0 to 1.0)
            - candidates: List of candidate SearchResults
        
        Example:
            data_type, confidence, candidates = semantic_layer.resolve_data_type("吃电情况")
            # data_type = "ep", confidence = 0.85, candidates = [...]
        """
        if not self._initialized:
            logger.warning("Semantic layer not initialized, using fallback")
            if fallback_to_exact:
                return self._exact_match_fallback(query)
            return None, 0.0, []
        
        try:
            # Perform vector search
            candidates = self.vector_store.search(
                query=query,
                embedding=self.embedding,
                top_k=self.config.top_k_retrieval,
            )
            
            if not candidates:
                logger.debug(f"No candidates found for query: {query}")
                if fallback_to_exact:
                    return self._exact_match_fallback(query)
                return None, 0.0, []
            
            # Filter by similarity threshold
            filtered_candidates = [
                c for c in candidates
                if c.score >= self.config.similarity_threshold
            ]
            
            if not filtered_candidates:
                logger.debug(f"All candidates below threshold for query: {query}")
                if fallback_to_exact:
                    return self._exact_match_fallback(query)
                return None, 0.0, candidates
            
            # Rerank candidates
            reranked_candidates = self._rerank_candidates(query, filtered_candidates)
            
            # Limit to top_n_rerank
            final_candidates = reranked_candidates[:self.config.top_n_rerank]
            
            if final_candidates:
                best = final_candidates[0]
                return best.type_id, best.score, final_candidates
            
            if fallback_to_exact:
                return self._exact_match_fallback(query)
            return None, 0.0, []
            
        except (EmbeddingError, VectorStoreError, RerankError) as e:
            logger.error(f"Error during semantic search: {e}")
            if fallback_to_exact:
                return self._exact_match_fallback(query)
            return None, 0.0, []
    
    def _rerank_candidates(
        self,
        query: str,
        candidates: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Rerank candidates using the rerank model.
        
        Args:
            query: Original query
            candidates: List of candidates to rerank
        
        Returns:
            Reranked list of candidates
        """
        if not self.reranker or len(candidates) <= 1:
            return candidates
        
        try:
            # Prepare documents for reranking
            documents = []
            for c in candidates:
                # Include name and synonym for better context
                doc = f"{c.data_type.name}: {c.synonym}"
                if c.data_type.description:
                    doc += f" - {c.data_type.description}"
                documents.append(doc)
            
            # Call rerank API
            rerank_results = self.reranker.rerank(
                query=query,
                documents=documents,
                top_n=len(candidates),
            )
            
            # Reorder candidates based on rerank results
            reranked = []
            for rr in rerank_results:
                if 0 <= rr.index < len(candidates):
                    candidate = candidates[rr.index]
                    # Update score with rerank score
                    reranked.append(SearchResult(
                        type_id=candidate.type_id,
                        synonym=candidate.synonym,
                        score=rr.score,
                        data_type=candidate.data_type,
                    ))
            
            return reranked
            
        except RerankError as e:
            logger.warning(f"Rerank failed, using original order: {e}")
            return candidates
    
    def _exact_match_fallback(
        self,
        query: str,
    ) -> Tuple[Optional[str], float, List[SearchResult]]:
        """
        Fall back to exact string matching against DATA_TYPE_PREFIXES.
        
        Args:
            query: Query text to match
        
        Returns:
            Tuple of (data_type_id, confidence, candidates)
        """
        query_lower = query.lower().strip()
        
        # Try exact match
        if query_lower in DATA_TYPE_PREFIXES:
            # Find the canonical type ID
            prefix = DATA_TYPE_PREFIXES[query_lower]
            # Extract type ID from prefix (e.g., "source_data_ep_" -> "ep")
            type_id = self._extract_type_id_from_prefix(prefix)
            logger.debug(f"Exact match fallback: {query} -> {type_id}")
            return type_id, 1.0, []
        
        if query in DATA_TYPE_PREFIXES:
            prefix = DATA_TYPE_PREFIXES[query]
            type_id = self._extract_type_id_from_prefix(prefix)
            logger.debug(f"Exact match fallback (case-sensitive): {query} -> {type_id}")
            return type_id, 1.0, []
        
        logger.debug(f"No exact match found for: {query}")
        return None, 0.0, []
    
    def _extract_type_id_from_prefix(self, prefix: str) -> str:
        """
        Extract type ID from collection prefix.
        
        Args:
            prefix: Collection prefix (e.g., "source_data_ep_")
        
        Returns:
            Type ID (e.g., "ep")
        """
        # Remove "source_data_" prefix and trailing "_"
        if prefix.startswith("source_data_") and prefix.endswith("_"):
            return prefix[12:-1]
        return prefix
    
    def get_collection_prefix(self, data_type: str) -> str:
        """
        Get collection prefix for a data type.
        
        Provides backward compatibility with the existing get_collection_prefix()
        function in collection_router.py.
        
        Args:
            data_type: Data type identifier or name
        
        Returns:
            Collection prefix string (e.g., "source_data_ep_")
        """
        # First try schema manager if initialized
        if self._initialized and self.schema_manager:
            dt = self.schema_manager.get_data_type(data_type)
            if dt:
                return dt.collection_prefix
        
        # Fall back to DATA_TYPE_PREFIXES
        data_type_lower = data_type.lower().strip()
        if data_type_lower in DATA_TYPE_PREFIXES:
            return DATA_TYPE_PREFIXES[data_type_lower]
        if data_type in DATA_TYPE_PREFIXES:
            return DATA_TYPE_PREFIXES[data_type]
        
        # Default
        return "source_data_ep_"
    
    def get_enhanced_context(self, query: str) -> Dict[str, Any]:
        """
        Get enhanced context information for Intent Parser.
        
        Performs semantic search and returns structured context that can be
        used to enhance the LLM prompt for intent parsing.
        
        Args:
            query: User query text
        
        Returns:
            Dictionary containing:
            - query: Original query
            - candidates: List of candidate data types with scores
            - best_match: Best matching data type (if any)
            - confidence: Confidence score
            - fallback_used: Whether fallback was used
            - processing_time_ms: Processing time
        
        Example:
            context = semantic_layer.get_enhanced_context("吃电情况")
            # {
            #     "query": "吃电情况",
            #     "candidates": [
            #         {"type_id": "ep", "name": "电量", "score": 0.85, ...},
            #         ...
            #     ],
            #     "best_match": {"type_id": "ep", "name": "电量", ...},
            #     "confidence": 0.85,
            #     "fallback_used": False,
            #     "processing_time_ms": 123.45
            # }
        """
        start_time = time.time()
        
        data_type_id, confidence, candidates = self.resolve_data_type(query)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build candidate list for context
        candidate_list = []
        for c in candidates:
            candidate_list.append({
                "type_id": c.type_id,
                "name": c.data_type.name,
                "synonym": c.synonym,
                "score": c.score,
                "unit": c.data_type.unit,
                "description": c.data_type.description,
            })
        
        # Build best match info
        best_match = None
        if data_type_id and candidates:
            best = candidates[0]
            best_match = {
                "type_id": best.type_id,
                "name": best.data_type.name,
                "collection_prefix": best.data_type.collection_prefix,
                "unit": best.data_type.unit,
                "description": best.data_type.description,
            }
        elif data_type_id:
            # Fallback was used
            best_match = {
                "type_id": data_type_id,
                "name": data_type_id,
                "collection_prefix": self.get_collection_prefix(data_type_id),
            }
        
        return {
            "query": query,
            "candidates": candidate_list,
            "best_match": best_match,
            "confidence": confidence,
            "fallback_used": len(candidates) == 0 and data_type_id is not None,
            "processing_time_ms": processing_time_ms,
        }
    
    def reload_config(self) -> bool:
        """
        Reload configuration and rebuild vector index.
        
        Reloads the YAML configuration and rebuilds the vector index
        with the new data types. If reload fails, keeps the previous
        configuration.
        
        Returns:
            True if reload was successful, False otherwise
        """
        if not self._initialized:
            logger.warning("Cannot reload: semantic layer not initialized")
            return False
        
        try:
            # Reload schema configuration
            if not self.schema_manager.reload_config():
                logger.error("Failed to reload schema configuration")
                return False
            
            # Rebuild vector index
            data_types = self.schema_manager.get_all_data_types()
            if data_types:
                self.vector_store.rebuild_index(data_types, self.embedding)
                logger.info(f"Rebuilt vector index with {len(data_types)} data types")
            
            return True
            
        except (VectorStoreError, EmbeddingError) as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False


def create_semantic_layer(
    config: Optional[SemanticLayerConfig] = None,
    auto_initialize: bool = True,
) -> Optional[SemanticLayer]:
    """
    Factory function to create and optionally initialize a SemanticLayer.
    
    Handles initialization failures gracefully by returning None,
    allowing the system to fall back to traditional mode.
    
    Args:
        config: SemanticLayerConfig instance. If None, loads from environment.
        auto_initialize: Whether to automatically initialize the semantic layer.
    
    Returns:
        Initialized SemanticLayer instance, or None if:
        - Semantic layer is disabled in config
        - Initialization fails
    
    Example:
        # Create with auto-initialization
        semantic_layer = create_semantic_layer()
        if semantic_layer:
            result = semantic_layer.resolve_data_type("吃电情况")
        else:
            # Fall back to traditional mode
            pass
        
        # Create without auto-initialization
        semantic_layer = create_semantic_layer(auto_initialize=False)
        if semantic_layer and semantic_layer.initialize():
            # Use semantic layer
            pass
    """
    # Load config from environment if not provided
    if config is None:
        config = SemanticLayerConfig.from_env()
    
    # Check if enabled
    if not config.enabled:
        logger.info("Semantic layer is disabled in configuration")
        return None
    
    # Create instance
    semantic_layer = SemanticLayer(config)
    
    # Auto-initialize if requested
    if auto_initialize:
        if not semantic_layer.initialize():
            logger.warning("Semantic layer initialization failed, returning None")
            return None
    
    return semantic_layer


__all__ = [
    "SemanticLayer",
    "SemanticSearchResult",
    "create_semantic_layer",
]
