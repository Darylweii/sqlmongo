"""
DashScope Rerank integration for the Semantic Layer.

This module provides the DashScopeRerank class for reranking search results
using Alibaba Cloud DashScope's gte-rerank-v2 model.

Requirements:
- 1.7: Use Alibaba Cloud DashScope gte-rerank-v2 model for reranking search results
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.semantic_layer.exceptions import RerankError

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL = "gte-rerank-v2"
DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3


@dataclass
class RerankResult:
    """
    Rerank result for a single document.
    
    Attributes:
        index: Original index of the document in the input list
        text: Document text
        score: Relevance score (higher is more relevant)
    """
    index: int
    text: str
    score: float


class DashScopeRerank:
    """
    Alibaba Cloud DashScope Rerank model wrapper.
    
    Uses the DashScope rerank API to reorder documents by relevance to a query.
    
    Attributes:
        api_key: DashScope API key
        model: Rerank model name (default: gte-rerank-v2)
        endpoint: API endpoint URL
        timeout: Request timeout in seconds
    
    Example:
        reranker = DashScopeRerank(api_key="your-api-key")
        results = reranker.rerank(
            query="电量查询",
            documents=["电量", "电流", "电压"],
            top_n=2
        )
        for result in results:
            print(f"{result.text}: {result.score}")
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the DashScope Rerank client.
        
        Args:
            api_key: DashScope API key (required)
            model: Rerank model name
            endpoint: API endpoint URL
            timeout: Request timeout in seconds
        
        Raises:
            RerankError: If api_key is empty
        """
        if not api_key:
            raise RerankError("DashScope API key is required")
        
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
    
    @property
    def client(self) -> httpx.Client:
        """Lazy initialization of HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
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
    
    def __enter__(self) -> "DashScopeRerank":
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
    def _call_api(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Call the DashScope rerank API.
        
        Args:
            query: Query text
            documents: List of documents to rerank
            top_n: Number of top results to return (None for all)
        
        Returns:
            List of RerankResult sorted by score descending
        
        Raises:
            RerankError: If API call fails
        """
        try:
            # Build request payload
            payload = {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "return_documents": True,
                },
            }
            
            if top_n is not None:
                payload["parameters"]["top_n"] = top_n
            
            response = self.client.post(self.endpoint, json=payload)
            
            if response.status_code == 401:
                raise RerankError(
                    "Invalid DashScope API key",
                    details={"status_code": 401}
                )
            
            if response.status_code == 429:
                raise RerankError(
                    "DashScope API rate limit exceeded",
                    details={"status_code": 429}
                )
            
            if response.status_code != 200:
                raise RerankError(
                    f"DashScope API error: {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "response": response.text[:500] if response.text else None,
                    }
                )
            
            data = response.json()
            
            # Check for API-level errors
            if "code" in data and data["code"] != "Success":
                raise RerankError(
                    f"DashScope API error: {data.get('message', 'Unknown error')}",
                    details={"code": data.get("code"), "request_id": data.get("request_id")}
                )
            
            # Extract results from output
            output = data.get("output", {})
            results_data = output.get("results", [])
            
            if not results_data:
                logger.warning("Empty results from rerank API")
                return []
            
            # Convert to RerankResult objects
            results = []
            for item in results_data:
                result = RerankResult(
                    index=item.get("index", 0),
                    text=item.get("document", {}).get("text", documents[item.get("index", 0)]),
                    score=item.get("relevance_score", 0.0),
                )
                results.append(result)
            
            # Sort by score descending (should already be sorted, but ensure)
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except httpx.TimeoutException as e:
            logger.error(f"DashScope rerank API timeout: {e}")
            raise
        except httpx.NetworkError as e:
            logger.error(f"DashScope rerank API network error: {e}")
            raise
        except RerankError:
            raise
        except Exception as e:
            raise RerankError(
                f"Unexpected error calling DashScope rerank API: {e}",
                details={"error_type": type(e).__name__}
            )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to a query.
        
        Args:
            query: Query text to compare against
            documents: List of document texts to rerank
            top_n: Number of top results to return (None for all)
        
        Returns:
            List of RerankResult sorted by relevance score (descending)
        
        Raises:
            RerankError: If reranking fails
        
        Example:
            results = reranker.rerank(
                query="吃电情况",
                documents=["电量", "电流", "功率因数"],
                top_n=2
            )
            # Returns top 2 most relevant documents
        """
        if not query or not query.strip():
            raise RerankError("Query cannot be empty")
        
        if not documents:
            return []
        
        # Filter out empty documents
        valid_docs = []
        valid_indices = []
        for i, doc in enumerate(documents):
            if doc and doc.strip():
                valid_docs.append(doc)
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping empty document at index {i}")
        
        if not valid_docs:
            raise RerankError("All documents are empty")
        
        # Call API
        results = self._call_api(query, valid_docs, top_n)
        
        # Map indices back to original document list
        for result in results:
            if 0 <= result.index < len(valid_indices):
                result.index = valid_indices[result.index]
        
        return results


__all__ = ["DashScopeRerank", "RerankResult"]
