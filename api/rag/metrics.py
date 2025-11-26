"""
Observability and metrics module for RAG system.

This module provides metrics collection, pipeline tracing, and performance monitoring.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available, metrics disabled")


@dataclass
class PipelineTrace:
    """Trace information for a single RAG pipeline execution."""
    
    query: str
    start_time: float
    end_time: Optional[float] = None
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def add_stage(
        self,
        name: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a pipeline stage with timing information."""
        self.stages[name] = {
            "duration": duration,
            "metadata": metadata or {},
        }
    
    def finish(self, error: Optional[str] = None) -> None:
        """Mark trace as finished."""
        self.end_time = time.time()
        self.error = error
    
    def get_total_duration(self) -> float:
        """Get total pipeline duration."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "query": self.query[:100],  # Truncate for privacy
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "total_duration": self.get_total_duration(),
            "stages": self.stages,
            "metadata": self.metadata,
            "error": self.error,
            "success": self.error is None,
        }


class MetricsCollector:
    """
    Collects and exposes metrics for the RAG system.
    
    Tracks:
    - Request counts and latencies
    - Cache hit rates
    - Retrieval metrics
    - Error rates
    - Token usage
    """

    def __init__(self, enabled: bool = True, port: int = 8000):
        """
        Initialize metrics collector.

        Args:
            enabled: Whether metrics collection is enabled.
            port: Port for Prometheus metrics endpoint.
        """
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        self.port = port
        self.traces: List[PipelineTrace] = []
        self.max_traces = 100  # Keep last 100 traces in memory
        
        if self.enabled:
            self._init_prometheus_metrics()
            self._start_metrics_server()
        else:
            self._init_simple_metrics()

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_counter = Counter(
            'rag_requests_total',
            'Total number of RAG requests',
            ['status']
        )
        
        self.request_duration = Histogram(
            'rag_request_duration_seconds',
            'RAG request duration in seconds',
            ['stage'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'rag_cache_hits_total',
            'Number of cache hits'
        )
        
        self.cache_misses = Counter(
            'rag_cache_misses_total',
            'Number of cache misses'
        )
        
        # Retrieval metrics
        self.retrieval_results = Histogram(
            'rag_retrieval_results',
            'Number of results retrieved',
            buckets=[1, 3, 5, 10, 20, 50]
        )
        
        self.retrieval_similarity = Histogram(
            'rag_retrieval_similarity',
            'Similarity scores of retrieved documents',
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        )
        
        # Token usage
        self.token_usage = Counter(
            'rag_tokens_total',
            'Total tokens used',
            ['type']  # 'prompt' or 'completion'
        )
        
        # Error metrics
        self.error_counter = Counter(
            'rag_errors_total',
            'Total number of errors',
            ['type']
        )
        
        # System info
        self.system_info = Info(
            'rag_system',
            'RAG system information'
        )

        logger.info("Prometheus metrics initialized")

    def _init_simple_metrics(self) -> None:
        """Initialize simple in-memory metrics."""
        self.simple_metrics = {
            "requests": {"total": 0, "success": 0, "error": 0},
            "cache": {"hits": 0, "misses": 0},
            "retrieval": {"total_results": 0, "total_queries": 0},
            "tokens": {"prompt": 0, "completion": 0},
            "errors": {},
            "stage_durations": {},
        }

    def _start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        try:
            start_http_server(self.port)
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    @contextmanager
    def trace_pipeline(self, query: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing a complete pipeline execution.

        Args:
            query: User query.
            metadata: Additional metadata.

        Yields:
            PipelineTrace instance.
        """
        trace = PipelineTrace(
            query=query,
            start_time=time.time(),
            metadata=metadata or {},
        )
        
        try:
            yield trace
            trace.finish()
            self._record_request(success=True)
        except Exception as e:
            trace.finish(error=str(e))
            self._record_request(success=False)
            self._record_error(type(e).__name__)
            raise
        finally:
            # Store trace
            self.traces.append(trace)
            if len(self.traces) > self.max_traces:
                self.traces.pop(0)
            
            # Log trace
            logger.info(
                f"Pipeline trace: {trace.get_total_duration():.3f}s - "
                f"Stages: {list(trace.stages.keys())}"
            )

    @contextmanager
    def measure_stage(self, trace: PipelineTrace, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for measuring a pipeline stage.

        Args:
            trace: Parent pipeline trace.
            stage_name: Name of the stage.
            metadata: Additional metadata.

        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            trace.add_stage(stage_name, duration, metadata)
            self._record_stage_duration(stage_name, duration)

    def _record_request(self, success: bool) -> None:
        """Record a request."""
        if self.enabled:
            status = 'success' if success else 'error'
            self.request_counter.labels(status=status).inc()
        else:
            self.simple_metrics["requests"]["total"] += 1
            if success:
                self.simple_metrics["requests"]["success"] += 1
            else:
                self.simple_metrics["requests"]["error"] += 1

    def _record_stage_duration(self, stage: str, duration: float) -> None:
        """Record stage duration."""
        if self.enabled:
            self.request_duration.labels(stage=stage).observe(duration)
        else:
            if stage not in self.simple_metrics["stage_durations"]:
                self.simple_metrics["stage_durations"][stage] = []
            self.simple_metrics["stage_durations"][stage].append(duration)

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        if self.enabled:
            self.cache_hits.inc()
        else:
            self.simple_metrics["cache"]["hits"] += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        if self.enabled:
            self.cache_misses.inc()
        else:
            self.simple_metrics["cache"]["misses"] += 1

    def record_retrieval(self, num_results: int, similarities: List[float]) -> None:
        """
        Record retrieval metrics.

        Args:
            num_results: Number of results retrieved.
            similarities: Similarity scores.
        """
        if self.enabled:
            self.retrieval_results.observe(num_results)
            for sim in similarities:
                self.retrieval_similarity.observe(sim)
        else:
            self.simple_metrics["retrieval"]["total_results"] += num_results
            self.simple_metrics["retrieval"]["total_queries"] += 1

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Record token usage.

        Args:
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
        """
        if self.enabled:
            self.token_usage.labels(type='prompt').inc(prompt_tokens)
            self.token_usage.labels(type='completion').inc(completion_tokens)
        else:
            self.simple_metrics["tokens"]["prompt"] += prompt_tokens
            self.simple_metrics["tokens"]["completion"] += completion_tokens

    def _record_error(self, error_type: str) -> None:
        """Record an error."""
        if self.enabled:
            self.error_counter.labels(type=error_type).inc()
        else:
            if error_type not in self.simple_metrics["errors"]:
                self.simple_metrics["errors"][error_type] = 0
            self.simple_metrics["errors"][error_type] += 1

    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent pipeline traces.

        Args:
            limit: Maximum number of traces to return.

        Returns:
            List of trace dictionaries.
        """
        return [trace.to_dict() for trace in self.traces[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with current stats.
        """
        if not self.enabled:
            # Calculate averages for simple metrics
            stats = dict(self.simple_metrics)
            for stage, durations in stats["stage_durations"].items():
                if durations:
                    stats["stage_durations"][stage] = {
                        "avg": sum(durations) / len(durations),
                        "min": min(durations),
                        "max": max(durations),
                        "count": len(durations),
                    }
            return stats
        
        return {
            "metrics_enabled": True,
            "prometheus_port": self.port,
            "recent_traces_count": len(self.traces),
        }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def init_metrics(enabled: bool = True, port: int = 8000) -> MetricsCollector:
    """
    Initialize global metrics collector.

    Args:
        enabled: Whether metrics are enabled.
        port: Port for Prometheus endpoint.

    Returns:
        MetricsCollector instance.
    """
    global _metrics_collector
    _metrics_collector = MetricsCollector(enabled=enabled, port=port)
    return _metrics_collector


def get_metrics() -> Optional[MetricsCollector]:
    """
    Get global metrics collector.

    Returns:
        MetricsCollector instance or None if not initialized.
    """
    return _metrics_collector
