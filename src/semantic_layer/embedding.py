"""
DashScope Embedding integration for the Semantic Layer.

This module provides the DashScopeEmbedding class for text vectorization
using Alibaba Cloud DashScope's text-embedding-v4 model via OpenAI-compatible API.

Requirements:
- 1.1: Support vectorizing business terms and storing in Synonym_Vector_Store
- 1.6: Use Alibaba Cloud DashScope text-embedding-v4 model for Chinese text vectorization
"""

import logging
from typing import List, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.semantic_layer.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL = "text-embedding-v4"
DEFAULT_DIMENSIONS = 1024
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3


class DashScopeEmbedding:
    """
    Alibaba Cloud DashScope Embedding model wrapper.
    
    Uses the OpenAI-compatible API to call text-embedding-v4 model
    for Chinese text vectorization.
    
    Attributes:
        api_key: DashScope API key
        model: Embedding model name (default: text-embedding-v4)
        dimensions: Output vector dimensions (default: 1024)
        base_url: API base URL
        timeout: Request timeout in seconds
    
    Example:
        embedding = DashScopeEmbedding(api_key="your-api-key")
        vector = embedding.embed_text("电量")
        vectors = embedding.embed_texts(["电量", "电流", "电压"])
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the DashScope Embedding client.
        
        Args:
            api_key: DashScope API key (required)
            model: Embedding model name
            dimensions: Output vector dimensions
            base_url: API base URL
            timeout: Request timeout in seconds
        
        Raises:
            EmbeddingError: If api_key is empty
        """
        if not api_key:
            raise EmbeddingError("DashScope API key is required")
        
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
    
    @property
    def client(self) -> httpx.Client:
        """Lazy initialization of HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
    
    def __enter__(self) -> "DashScopeEmbedding":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call the DashScope embedding API.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        
        Raises:
            EmbeddingError: If API call fails
        """
        try:
            response = self.client.post(
                "/embeddings",
                json={
                    "model": self.model,
                    "input": texts,
                    "dimensions": self.dimensions,
                    "encoding_format": "float",
                },
            )
            
            if response.status_code == 401:
                raise EmbeddingError(
                    "Invalid DashScope API key",
                    details={"status_code": 401}
                )
            
            if response.status_code == 429:
                raise EmbeddingError(
                    "DashScope API rate limit exceeded",
                    details={"status_code": 429}
                )
            
            if response.status_code != 200:
                raise EmbeddingError(
                    f"DashScope API error: {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "response": response.text[:500] if response.text else None,
                    }
                )
            
            data = response.json()
            
            if "data" not in data:
                raise EmbeddingError(
                    "Invalid API response: missing 'data' field",
                    details={"response": data}
                )
            
            # Sort by index to ensure correct order
            embeddings_data = sorted(data["data"], key=lambda x: x.get("index", 0))
            vectors = [item["embedding"] for item in embeddings_data]
            
            # Validate vector dimensions
            for i, vec in enumerate(vectors):
                if len(vec) != self.dimensions:
                    logger.warning(
                        f"Vector dimension mismatch at index {i}: "
                        f"expected {self.dimensions}, got {len(vec)}"
                    )
            
            return vectors
            
        except httpx.TimeoutException as e:
            logger.error(f"DashScope API timeout: {e}")
            raise
        except httpx.NetworkError as e:
            logger.error(f"DashScope API network error: {e}")
            raise
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Unexpected error calling DashScope API: {e}",
                details={"error_type": type(e).__name__}
            )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Vectorize a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        
        Raises:
            EmbeddingError: If embedding fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        vectors = self._call_api([text])
        
        if not vectors:
            raise EmbeddingError(
                "Empty response from embedding API",
                details={"input": text}
            )
        
        return vectors[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Vectorize multiple texts in batch.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        
        Raises:
            EmbeddingError: If embedding fails
        """
        if not texts:
            return []
        
        # Filter out empty texts and track their indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping empty text at index {i}")
        
        if not valid_texts:
            raise EmbeddingError("All input texts are empty")
        
        # Call API with valid texts
        vectors = self._call_api(valid_texts)
        
        # Reconstruct result list with None for skipped texts
        result = [None] * len(texts)
        for idx, vec in zip(valid_indices, vectors):
            result[idx] = vec
        
        # Replace None with empty vectors for skipped texts
        empty_vector = [0.0] * self.dimensions
        result = [vec if vec is not None else empty_vector for vec in result]
        
        return result


__all__ = ["DashScopeEmbedding"]
