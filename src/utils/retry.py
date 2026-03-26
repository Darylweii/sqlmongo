"""
Retry utilities with exponential backoff for database operations.

This module provides retry decorators using tenacity library for handling
transient failures in database queries.

Requirements: 8.2 - WHEN 数据库连接超时 THEN 系统 SHALL 进行有限次数的重试
"""

from functools import wraps
from typing import Callable, Type, Tuple, Any
import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)


# Default retry configuration
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_WAIT_MULTIPLIER = 1
DEFAULT_WAIT_MIN = 1
DEFAULT_WAIT_MAX = 10


def retry_with_exponential_backoff(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    wait_multiplier: float = DEFAULT_WAIT_MULTIPLIER,
    wait_min: float = DEFAULT_WAIT_MIN,
    wait_max: float = DEFAULT_WAIT_MAX,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator factory for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        wait_multiplier: Multiplier for exponential wait time (default: 1)
        wait_min: Minimum wait time in seconds (default: 1)
        wait_max: Maximum wait time in seconds (default: 10)
        retryable_exceptions: Tuple of exception types to retry on
    
    Returns:
        Decorator function
    
    Example:
        @retry_with_exponential_backoff(max_attempts=3)
        def query_database():
            # database operation
            pass
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_multiplier, min=wait_min, max=wait_max),
        retry=retry_if_exception_type(retryable_exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def database_retry(func: Callable = None, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Callable:
    """
    Decorator for database operations with default retry settings.
    
    This decorator applies exponential backoff retry logic specifically
    configured for database operations (connection timeouts, transient errors).
    
    Args:
        func: Function to decorate (when used without parentheses)
        max_attempts: Maximum number of retry attempts
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @database_retry
        def query_with_retry(collection, query):
            return collection.find(query)
        
        @database_retry(max_attempts=5)
        def important_query(collection, query):
            return collection.find(query)
    """
    # Common database exceptions to retry on
    from pymongo.errors import (
        ConnectionFailure,
        ServerSelectionTimeoutError,
        AutoReconnect,
        NetworkTimeout,
    )
    from sqlalchemy.exc import (
        OperationalError,
        InterfaceError,
        TimeoutError as SQLAlchemyTimeoutError,
    )
    
    retryable_db_exceptions = (
        ConnectionFailure,
        ServerSelectionTimeoutError,
        AutoReconnect,
        NetworkTimeout,
        OperationalError,
        InterfaceError,
        SQLAlchemyTimeoutError,
        ConnectionError,
        TimeoutError,
    )
    
    decorator = retry_with_exponential_backoff(
        max_attempts=max_attempts,
        retryable_exceptions=retryable_db_exceptions,
    )
    
    if func is not None:
        # Called without parentheses: @database_retry
        return decorator(func)
    
    # Called with parentheses: @database_retry(max_attempts=5)
    return decorator


def async_database_retry(
    func: Callable = None,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
) -> Callable:
    """
    Async version of database_retry decorator.
    
    Args:
        func: Async function to decorate
        max_attempts: Maximum number of retry attempts
    
    Returns:
        Decorated async function with retry logic
    
    Example:
        @async_database_retry
        async def async_query(collection, query):
            return await collection.find(query).to_list(None)
    """
    from pymongo.errors import (
        ConnectionFailure,
        ServerSelectionTimeoutError,
        AutoReconnect,
        NetworkTimeout,
    )
    
    retryable_db_exceptions = (
        ConnectionFailure,
        ServerSelectionTimeoutError,
        AutoReconnect,
        NetworkTimeout,
        ConnectionError,
        TimeoutError,
    )
    
    decorator = retry_with_exponential_backoff(
        max_attempts=max_attempts,
        retryable_exceptions=retryable_db_exceptions,
    )
    
    if func is not None:
        return decorator(func)
    
    return decorator


__all__ = [
    "retry_with_exponential_backoff",
    "database_retry",
    "async_database_retry",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_WAIT_MULTIPLIER",
    "DEFAULT_WAIT_MIN",
    "DEFAULT_WAIT_MAX",
]
