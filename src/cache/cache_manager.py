"""
Cache Manager module for Redis caching.

This module provides a CacheManager class that handles Redis caching
for sensor data queries, including key generation, get/set operations,
and recent query detection.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, List, Optional

import redis

from src.exceptions import CacheConnectionError


class CacheManager:
    """
    Redis 缓存管理器
    
    Manages caching of sensor data queries with consistent key generation
    and TTL-based expiration. Supports detection of recent queries that
    are suitable for caching.
    
    Attributes:
        client: Redis client instance
        default_ttl: Default time-to-live for cached entries in seconds
    """
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        """
        Initialize the CacheManager.
        
        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379")
            default_ttl: Default TTL for cached entries in seconds (default: 3600)
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._client: Optional[redis.Redis] = None
    
    @property
    def client(self) -> redis.Redis:
        """Lazy initialization of Redis client."""
        if self._client is None:
            try:
                self._client = redis.from_url(self.redis_url)
                # Test connection
                self._client.ping()
            except redis.ConnectionError as e:
                raise CacheConnectionError(f"无法连接到 Redis: {e}")
        return self._client
    
    def _generate_key(
        self,
        device_codes: List[str],
        start_time: str,
        end_time: str,
        *,
        data_type: str = "",
        tags: Optional[List[str]] = None,
        tg_values: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 0,
        output_format: str = "",
        value_filter: Optional[dict] = None,
    ) -> str:
        """
        生成缓存 key
        
        Generates a consistent cache key based on device codes and time range.
        The key is deterministic regardless of the order of device_codes.
        
        Args:
            device_codes: List of device codes to query
            start_time: Start time string (any format)
            end_time: End time string (any format)
        
        Returns:
            A unique cache key string prefixed with "sensor_data:"
        """
        # Sort qualifiers to ensure consistent keys regardless of order.
        sorted_codes = sorted(device_codes)
        sorted_tags = sorted(str(tag).strip() for tag in (tags or []) if str(tag).strip())
        sorted_tgs = sorted(str(tg).strip() for tg in (tg_values or []) if str(tg).strip())
        normalized_filter = json.dumps(value_filter or {}, ensure_ascii=False, sort_keys=True)
        content = json.dumps(
            {
                "device_codes": sorted_codes,
                "start_time": start_time,
                "end_time": end_time,
                "data_type": str(data_type or "").strip(),
                "tags": sorted_tags,
                "tg_values": sorted_tgs,
                "page": int(page or 1),
                "page_size": int(page_size or 0),
                "output_format": str(output_format or "").strip(),
                "value_filter": normalized_filter,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        hash_value = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"sensor_data:{hash_value}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Retrieves a cached value by key.
        
        Args:
            key: The cache key to look up
        
        Returns:
            The cached value (deserialized from JSON) or None if not found
        
        Raises:
            CacheConnectionError: If unable to connect to Redis
        """
        try:
            value = self.client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except redis.ConnectionError as e:
            raise CacheConnectionError(f"Redis 连接失败: {e}")
        except json.JSONDecodeError:
            # If value is not valid JSON, return as string
            return value.decode() if isinstance(value, bytes) else value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存
        
        Stores a value in the cache with optional TTL.
        
        Args:
            key: The cache key
            value: The value to cache (will be serialized to JSON)
            ttl: Time-to-live in seconds (uses default_ttl if not specified)
        
        Raises:
            CacheConnectionError: If unable to connect to Redis
        """
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            serialized_value = json.dumps(value, default=str)
            self.client.setex(key, ttl, serialized_value)
        except redis.ConnectionError as e:
            raise CacheConnectionError(f"Redis 连接失败: {e}")
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Removes a cached value by key.
        
        Args:
            key: The cache key to delete
        
        Returns:
            True if the key was deleted, False if it didn't exist
        
        Raises:
            CacheConnectionError: If unable to connect to Redis
        """
        try:
            return bool(self.client.delete(key))
        except redis.ConnectionError as e:
            raise CacheConnectionError(f"Redis 连接失败: {e}")
    
    def is_recent_query(
        self, 
        start_time: str, 
        end_time: str, 
        days: int = 7
    ) -> bool:
        """
        判断是否为最近 N 天的查询（适合缓存）
        
        Determines if a query's time range falls within the recent N days,
        which makes it suitable for caching (hot data).
        
        Args:
            start_time: Start time string in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format
            end_time: End time string in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format
            days: Number of days to consider as "recent" (default: 7)
        
        Returns:
            True if the query time range is within the recent N days
        """
        try:
            # Parse dates - support both date and datetime formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    end_dt = datetime.strptime(end_time, fmt)
                    break
                except ValueError:
                    continue
            else:
                # If no format matches, return False (not cacheable)
                return False
            
            # Calculate the threshold date
            now = datetime.now()
            threshold = now - timedelta(days=days)
            
            # Query is recent if end_time is within the last N days
            return end_dt >= threshold
            
        except (ValueError, TypeError):
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: The cache key to check
        
        Returns:
            True if the key exists in cache
        
        Raises:
            CacheConnectionError: If unable to connect to Redis
        """
        try:
            return bool(self.client.exists(key))
        except redis.ConnectionError as e:
            raise CacheConnectionError(f"Redis 连接失败: {e}")
    
    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
