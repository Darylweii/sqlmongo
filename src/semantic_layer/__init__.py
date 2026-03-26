"""
Semantic Layer Module for AI Data Router Agent.

This module provides semantic understanding capabilities for the DAG orchestrator,
including synonym vector search, virtual schema configuration, and enhanced intent parsing.

Components:
- SemanticLayerConfig: Configuration management for the semantic layer
- SemanticLayerError: Exception hierarchy for semantic layer errors
- VirtualSchemaManager: YAML-based data type configuration management
- SynonymVectorStore: Vector storage for semantic similarity search
- DashScopeEmbedding: Alibaba Cloud DashScope embedding integration
- DashScopeRerank: Alibaba Cloud DashScope rerank integration
- SemanticLayer: Main orchestration class

Usage:
    from src.semantic_layer import SemanticLayer, SemanticLayerConfig
    
    config = SemanticLayerConfig.from_env()
    semantic_layer = SemanticLayer(config)
    semantic_layer.initialize()
    
    result = semantic_layer.resolve_data_type("吃电情况")
"""

from src.semantic_layer.config import SemanticLayerConfig
from src.semantic_layer.exceptions import (
    SemanticLayerError,
    ConfigurationError,
    EmbeddingError,
    VectorStoreError,
    RerankError,
)
from src.semantic_layer.schema import (
    DataTypeDefinition,
    VirtualSchemaSettings,
    VirtualSchemaManager,
)
from src.semantic_layer.embedding import DashScopeEmbedding
from src.semantic_layer.rerank import DashScopeRerank, RerankResult
from src.semantic_layer.vector_store import (
    SynonymVectorStore,
    FAISSVectorStore,
    MilvusVectorStore,
    SearchResult,
    VectorIndexEntry,
)
from src.semantic_layer.semantic_layer import (
    SemanticLayer,
    SemanticSearchResult,
    create_semantic_layer,
)

__all__ = [
    "SemanticLayerConfig",
    "SemanticLayerError",
    "ConfigurationError",
    "EmbeddingError",
    "VectorStoreError",
    "RerankError",
    "DataTypeDefinition",
    "VirtualSchemaSettings",
    "VirtualSchemaManager",
    "DashScopeEmbedding",
    "DashScopeRerank",
    "RerankResult",
    "SynonymVectorStore",
    "FAISSVectorStore",
    "MilvusVectorStore",
    "SearchResult",
    "VectorIndexEntry",
    "SemanticLayer",
    "SemanticSearchResult",
    "create_semantic_layer",
]
