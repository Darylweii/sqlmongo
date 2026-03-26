"""
Custom exceptions for the Semantic Layer.

This module defines the exception hierarchy for semantic layer errors,
providing specific exception types for different failure scenarios.
"""


class SemanticLayerError(Exception):
    """
    Base exception for all semantic layer errors.
    
    All semantic layer specific exceptions inherit from this class,
    allowing for broad exception handling when needed.
    """
    
    def __init__(self, message: str, details: dict | None = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(SemanticLayerError):
    """
    Exception raised for configuration-related errors.
    
    This includes:
    - Invalid YAML syntax in configuration files
    - Missing required configuration fields
    - Invalid configuration values
    - Configuration file not found or inaccessible
    """
    pass


class EmbeddingError(SemanticLayerError):
    """
    Exception raised for embedding-related errors.
    
    This includes:
    - DashScope API call failures
    - Invalid API key
    - Rate limiting errors
    - Network timeouts
    - Invalid embedding responses
    """
    pass


class VectorStoreError(SemanticLayerError):
    """
    Exception raised for vector store-related errors.
    
    This includes:
    - FAISS index corruption or loading failures
    - Milvus connection failures
    - Index building errors
    - Search operation failures
    - Dimension mismatch errors
    """
    pass


class RerankError(SemanticLayerError):
    """
    Exception raised for reranking-related errors.
    
    This includes:
    - DashScope rerank API call failures
    - Invalid rerank responses
    - Rate limiting errors
    - Network timeouts
    """
    pass
