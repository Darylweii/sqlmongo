"""
Configuration management for the Semantic Layer.

This module provides the SemanticLayerConfig dataclass for managing
all semantic layer related configurations including vector store,
embedding model, and retrieval parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SemanticLayerConfig:
    """
    Semantic Layer configuration.
    
    Attributes:
        enabled: Whether the semantic layer is enabled
        config_path: Path to the virtual schema YAML configuration file
        vector_store_type: Type of vector store ("faiss" or "milvus")
        faiss_index_path: Path to save/load FAISS index
        milvus_host: Milvus server host
        milvus_port: Milvus server port
        milvus_collection: Milvus collection name
        dashscope_api_key: Alibaba Cloud DashScope API key
        embedding_model: Embedding model name
        embedding_dimensions: Embedding vector dimensions
        rerank_model: Rerank model name
        similarity_threshold: Minimum similarity score for valid matches
        top_k_retrieval: Number of candidates from vector search
        top_n_rerank: Number of results after reranking
        hot_reload_enabled: Whether to enable configuration hot reload
        hot_reload_interval_seconds: Interval for checking config changes
    """
    
    # Core settings
    enabled: bool = False
    config_path: str = "config/semantic_layer.yaml"
    
    # Vector store configuration
    vector_store_type: str = "faiss"
    faiss_index_path: str = "data/semantic_layer.faiss"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "semantic_layer"
    
    # Alibaba Cloud DashScope configuration
    dashscope_api_key: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    rerank_model: str = "gte-rerank-v2"
    
    # Retrieval configuration
    similarity_threshold: float = 0.5
    top_k_retrieval: int = 10
    top_n_rerank: int = 5
    
    # Hot reload configuration
    hot_reload_enabled: bool = True
    hot_reload_interval_seconds: int = 60
    
    @classmethod
    def from_env(cls) -> "SemanticLayerConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            SEMANTIC_LAYER_ENABLED: Enable/disable semantic layer (true/false)
            SEMANTIC_LAYER_CONFIG_PATH: Path to virtual schema YAML config
            VECTOR_STORE_TYPE: Vector store type ("faiss" or "milvus")
            FAISS_INDEX_PATH: Path to FAISS index file
            MILVUS_HOST: Milvus server host
            MILVUS_PORT: Milvus server port
            MILVUS_COLLECTION: Milvus collection name
            DASHSCOPE_API_KEY: Alibaba Cloud DashScope API key
            EMBEDDING_MODEL: Embedding model name
            EMBEDDING_DIMENSIONS: Embedding vector dimensions
            RERANK_MODEL: Rerank model name
            SIMILARITY_THRESHOLD: Minimum similarity threshold
            TOP_K_RETRIEVAL: Number of vector search candidates
            TOP_N_RERANK: Number of results after reranking
            HOT_RELOAD_ENABLED: Enable configuration hot reload
            HOT_RELOAD_INTERVAL: Hot reload check interval in seconds
        
        Returns:
            SemanticLayerConfig instance with values from environment
        """
        return cls(
            enabled=os.getenv("SEMANTIC_LAYER_ENABLED", "false").lower() == "true",
            config_path=os.getenv("SEMANTIC_LAYER_CONFIG_PATH", "config/semantic_layer.yaml"),
            vector_store_type=os.getenv("VECTOR_STORE_TYPE", "faiss"),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "data/semantic_layer.faiss"),
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=int(os.getenv("MILVUS_PORT", "19530")),
            milvus_collection=os.getenv("MILVUS_COLLECTION", "semantic_layer"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
            rerank_model=os.getenv("RERANK_MODEL", "gte-rerank-v2"),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.5")),
            top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "10")),
            top_n_rerank=int(os.getenv("TOP_N_RERANK", "5")),
            hot_reload_enabled=os.getenv("HOT_RELOAD_ENABLED", "true").lower() == "true",
            hot_reload_interval_seconds=int(os.getenv("HOT_RELOAD_INTERVAL", "60")),
        )
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration values.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if self.enabled and not self.dashscope_api_key:
            errors.append("DASHSCOPE_API_KEY is required when semantic layer is enabled")
        
        if self.vector_store_type not in ("faiss", "milvus"):
            errors.append(f"vector_store_type must be 'faiss' or 'milvus', got '{self.vector_store_type}'")
        
        if self.similarity_threshold < 0.0 or self.similarity_threshold > 1.0:
            errors.append(f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}")
        
        if self.top_k_retrieval <= 0:
            errors.append(f"top_k_retrieval must be positive, got {self.top_k_retrieval}")
        
        if self.top_n_rerank <= 0:
            errors.append(f"top_n_rerank must be positive, got {self.top_n_rerank}")
        
        if self.top_n_rerank > self.top_k_retrieval:
            errors.append(f"top_n_rerank ({self.top_n_rerank}) should not exceed top_k_retrieval ({self.top_k_retrieval})")
        
        if self.embedding_dimensions <= 0:
            errors.append(f"embedding_dimensions must be positive, got {self.embedding_dimensions}")
        
        if self.hot_reload_interval_seconds <= 0:
            errors.append(f"hot_reload_interval_seconds must be positive, got {self.hot_reload_interval_seconds}")
        
        return (len(errors) == 0, errors)
