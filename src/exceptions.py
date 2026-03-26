"""
Exception class hierarchy for AI Data Router Agent.

This module defines all custom exceptions used throughout the system,
organized in a hierarchical structure for proper error handling.
"""

from typing import List, Any, Optional


class DataRouterError(Exception):
    """Base exception class for all Data Router errors."""
    pass


# Collection Router Exceptions
class CollectionRouterError(DataRouterError):
    """Base exception for Collection Router related errors."""
    pass


class CircuitBreakerError(CollectionRouterError):
    """
    Circuit breaker exception - raised when query scope is too large.
    
    Triggered when the number of collections to query exceeds the maximum
    allowed threshold (default: 50 collections).
    """
    def __init__(self, collection_count: int, max_allowed: int):
        self.collection_count = collection_count
        self.max_allowed = max_allowed
        super().__init__(
            f"查询范围过大：需要扫描 {collection_count} 张表，"
            f"超过最大允许值 {max_allowed}。请缩小时间范围。"
        )


class InvalidDateRangeError(CollectionRouterError):
    """
    Invalid date range exception.
    
    Raised when date format is invalid or date range is illogical
    (e.g., start date after end date).
    """
    def __init__(self, message: str):
        super().__init__(message)


# Metadata Engine Exceptions
class MetadataEngineError(DataRouterError):
    """Base exception for Metadata Engine related errors."""
    pass


class DatabaseConnectionError(MetadataEngineError):
    """
    Database connection exception.
    
    Raised when unable to connect to the database or connection times out.
    """
    def __init__(self, message: str = "数据库连接失败"):
        super().__init__(message)


# Data Fetcher Exceptions
class DataFetcherError(DataRouterError):
    """Base exception for Data Fetcher related errors."""
    pass


class PartialQueryFailureError(DataFetcherError):
    """
    Partial query failure exception (non-fatal).
    
    Raised when some collections fail to query but others succeed.
    Contains both the successful data and list of failed collections.
    """
    def __init__(
        self, 
        successful_data: List[Any], 
        failed_collections: List[str],
        message: Optional[str] = None
    ):
        self.successful_data = successful_data
        self.failed_collections = failed_collections
        if message is None:
            message = f"部分查询失败：{len(failed_collections)} 个集合查询失败"
        super().__init__(message)


# Cache Exceptions
class CacheError(DataRouterError):
    """Base exception for Cache related errors."""
    pass


class CacheConnectionError(CacheError):
    """Raised when unable to connect to Redis cache."""
    def __init__(self, message: str = "Redis 缓存连接失败"):
        super().__init__(message)


# Tool Exceptions
class ToolError(DataRouterError):
    """Base exception for Tool related errors."""
    pass


class MissingParameterError(ToolError):
    """Raised when required parameters are missing for a tool call."""
    def __init__(self, parameter_name: str):
        self.parameter_name = parameter_name
        super().__init__(f"缺少必需参数：{parameter_name}")


# Agent Exceptions
class AgentError(DataRouterError):
    """Base exception for Agent related errors."""
    pass


class IntentRecognitionError(AgentError):
    """Raised when intent recognition fails."""
    pass
