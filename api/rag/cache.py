"""
Response cache module for RAG system.

This module implements caching of responses to reduce API costs and latency.
Supports both in-memory and Redis-based caching.
"""

import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        """Set value in cache with TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache implementation."""

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            return None

        entry = self.cache[key]
        if time.time() > entry["expires_at"]:
            del self.cache[key]
            return None

        return entry["value"]

    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        """Set value in cache with TTL."""
        self.cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
        }

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()


class RedisCache(CacheBackend):
    """Redis cache implementation."""

    def __init__(self, redis_url: str):
        try:
            import redis
            self.redis = redis.from_url(redis_url)
            self.redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from Redis cache."""
        try:
            value = self.redis.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error(f"Error getting from Redis: {e}")
            return None

    def set(self, key: str, value: Dict[str, Any], ttl: int) -> None:
        """Set value in Redis cache with TTL."""
        try:
            self.redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Error setting to Redis: {e}")

    def delete(self, key: str) -> None:
        """Delete value from Redis."""
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Error deleting from Redis: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self.redis.flushdb()
        except Exception as e:
            logger.error(f"Error clearing Redis: {e}")


class ResponseCache:
    """
    High-level cache for RAG responses.
    
    Caches responses based on normalized query to reduce API costs.
    """

    def __init__(
        self,
        backend: CacheBackend,
        ttl: int = 3600,
        enabled: bool = True,
    ):
        """
        Initialize response cache.

        Args:
            backend: Cache backend (Memory or Redis).
            ttl: Time to live for cache entries in seconds.
            enabled: Whether caching is enabled.
        """
        self.backend = backend
        self.ttl = ttl
        self.enabled = enabled
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
        }

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for cache key generation.

        Args:
            query: User query.

        Returns:
            Normalized query string.
        """
        # Convert to lowercase and strip whitespace
        normalized = query.lower().strip()
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return normalized

    def _generate_cache_key(self, query: str, top_k: int) -> str:
        """
        Generate cache key from query.

        Args:
            query: User query.
            top_k: Number of results retrieved.

        Returns:
            Cache key.
        """
        normalized = self._normalize_query(query)
        # Include top_k in key to cache different result counts separately
        key_string = f"{normalized}:{top_k}"
        # Use hash to keep keys short
        return f"rag:response:{hashlib.md5(key_string.encode()).hexdigest()}"

    def get(self, query: str, top_k: int) -> Optional[Dict[str, Any]]:
        """
        Get cached response for query.

        Args:
            query: User query.
            top_k: Number of results to retrieve.

        Returns:
            Cached response or None if not found.
        """
        if not self.enabled:
            return None

        key = self._generate_cache_key(query, top_k)
        result = self.backend.get(key)

        if result is not None:
            self.stats["hits"] += 1
            logger.info(f"Cache hit for query: {query[:50]}...")
        else:
            self.stats["misses"] += 1
            logger.debug(f"Cache miss for query: {query[:50]}...")

        return result

    def set(
        self,
        query: str,
        top_k: int,
        response: str,
        sources: List[Dict[str, Any]],
    ) -> None:
        """
        Cache a response.

        Args:
            query: User query.
            top_k: Number of results retrieved.
            response: Generated response.
            sources: Source documents.
        """
        if not self.enabled:
            return

        key = self._generate_cache_key(query, top_k)
        value = {
            "response": response,
            "sources": sources,
            "timestamp": time.time(),
        }

        self.backend.set(key, value, self.ttl)
        self.stats["sets"] += 1
        logger.debug(f"Cached response for query: {query[:50]}...")

    def invalidate_url(self, url: str) -> None:
        """
        Invalidate cache entries containing a specific URL.

        Note: This requires iterating through cache in Redis.
        For production, consider using cache tags or separate invalidation tracking.

        Args:
            url: URL that was updated.
        """
        logger.info(f"Cache invalidation for URL: {url} (not fully implemented)")
        # TODO: Implement proper cache invalidation strategy
        # Options:
        # 1. Store URL -> cache key mappings
        # 2. Use cache tags
        # 3. Clear all cache on updates (simple but inefficient)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.backend.clear()
        self.stats = {"hits": 0, "misses": 0, "sets": 0}
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }


def create_cache(redis_url: Optional[str] = None, ttl: int = 3600, enabled: bool = True) -> ResponseCache:
    """
    Create a response cache with appropriate backend.

    Args:
        redis_url: Redis connection URL (if None, uses memory cache).
        ttl: Time to live for cache entries.
        enabled: Whether caching is enabled.

    Returns:
        ResponseCache instance.
    """
    if redis_url:
        try:
            backend = RedisCache(redis_url)
            logger.info("Using Redis cache backend")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis, falling back to memory cache: {e}")
            backend = MemoryCache()
    else:
        backend = MemoryCache()
        logger.info("Using in-memory cache backend")

    return ResponseCache(backend=backend, ttl=ttl, enabled=enabled)
