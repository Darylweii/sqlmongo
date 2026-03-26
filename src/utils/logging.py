"""
Logging configuration for the AI Data Router Agent system.

This module provides centralized logging configuration with support for
exception logging and structured output.

Requirements: 8.4 - THE 系统 SHALL 记录所有异常到日志系统
"""

import logging
import sys
from typing import Optional
from datetime import datetime


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Logger names for different components
LOGGER_NAMES = {
    "root": "ai_data_router",
    "router": "ai_data_router.router",
    "metadata": "ai_data_router.metadata",
    "fetcher": "ai_data_router.fetcher",
    "cache": "ai_data_router.cache",
    "compressor": "ai_data_router.compressor",
    "agent": "ai_data_router.agent",
    "tools": "ai_data_router.tools",
}


def configure_logging(
    level: int = logging.INFO,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure the root logger for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_format: Log message format string
        date_format: Date format string
        log_file: Optional file path for file logging
    
    Returns:
        Configured root logger
    
    Example:
        logger = configure_logging(level=logging.DEBUG)
        logger.info("Application started")
    """
    root_logger = logging.getLogger(LOGGER_NAMES["root"])
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(component: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        component: Component name (router, metadata, fetcher, cache, compressor, agent, tools)
    
    Returns:
        Logger instance for the component
    
    Example:
        logger = get_logger("router")
        logger.info("Processing collection routing")
    """
    logger_name = LOGGER_NAMES.get(component, f"{LOGGER_NAMES['root']}.{component}")
    return logging.getLogger(logger_name)


def log_exception(
    logger: logging.Logger,
    exception: Exception,
    context: Optional[str] = None,
    extra_data: Optional[dict] = None,
) -> None:
    """
    Log an exception with full context and traceback.
    
    This function ensures all exceptions are properly logged with
    relevant context information for debugging.
    
    Args:
        logger: Logger instance to use
        exception: The exception to log
        context: Optional context description
        extra_data: Optional dictionary of additional data to log
    
    Example:
        try:
            risky_operation()
        except Exception as e:
            log_exception(logger, e, context="During data fetch", extra_data={"query": query})
    """
    error_message = f"Exception occurred: {type(exception).__name__}: {str(exception)}"
    
    if context:
        error_message = f"[{context}] {error_message}"
    
    if extra_data:
        extra_info = ", ".join(f"{k}={v}" for k, v in extra_data.items())
        error_message = f"{error_message} | Extra: {extra_info}"
    
    logger.exception(error_message)


class ExceptionLogger:
    """
    Context manager for automatic exception logging.
    
    Example:
        with ExceptionLogger(logger, "database query"):
            result = db.query(...)
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        reraise: bool = True,
        extra_data: Optional[dict] = None,
    ):
        """
        Initialize the exception logger context manager.
        
        Args:
            logger: Logger instance to use
            operation: Description of the operation being performed
            reraise: Whether to re-raise the exception after logging
            extra_data: Optional additional data to include in logs
        """
        self.logger = logger
        self.operation = operation
        self.reraise = reraise
        self.extra_data = extra_data or {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            log_exception(
                self.logger,
                exc_val,
                context=self.operation,
                extra_data=self.extra_data,
            )
            return not self.reraise
        return False


__all__ = [
    "configure_logging",
    "get_logger",
    "log_exception",
    "ExceptionLogger",
    "LOGGER_NAMES",
    "DEFAULT_FORMAT",
    "DEFAULT_DATE_FORMAT",
]
