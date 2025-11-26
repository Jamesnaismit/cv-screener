"""
Configuration module for the RAG application.

This module handles loading and validating configuration from environment variables.
"""

import os
from typing import Optional


class AppConfig:
    """
    Configuration class for the RAG application.

    Loads and manages all application settings from environment variables,
    organized into logical groups for better maintainability.
    """

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        # ═══════════════════ Core API Configuration ═══════════════════
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.database_url: str = os.getenv("DATABASE_URL", "")

        # ═══════════════════ Model Configuration ═══════════════════
        self.model_name: str = os.getenv("APP_MODEL_NAME", "gpt-4o-mini")
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.temperature: float = 0.7
        self.max_tokens: int = 1000

        # ═══════════════════ Application Settings ═══════════════════
        self.port: int = int(os.getenv("APP_PORT", "8000"))
        self.log_level: str = os.getenv("APP_LOG_LEVEL", "INFO")

        # ═══════════════════ RAG Configuration ═══════════════════
        self.max_history: int = int(os.getenv("APP_MAX_HISTORY", "10"))
        self.top_k_results: int = int(os.getenv("APP_TOP_K_RESULTS", "5"))

        # ═══════════════════ Cache Configuration ═══════════════════
        self.cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
        self.redis_url: Optional[str] = os.getenv("REDIS_URL")

        # ═══════════════════ Re-ranking Configuration ═══════════════════
        self.rerank_enabled: bool = os.getenv("RERANK_ENABLED", "true").lower() == "true"
        self.rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "20"))

        # ═══════════════════ Metrics Configuration ═══════════════════
        self.metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
        self.metrics_port: int = int(os.getenv("METRICS_PORT", "9000"))

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        errors = []

        # Validate required fields
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        if not self.database_url:
            errors.append("DATABASE_URL is required")

        # Validate port ranges
        if self.port <= 0 or self.port > 65535:
            errors.append(f"APP_PORT must be between 1 and 65535 (got {self.port})")
        if self.metrics_enabled and (self.metrics_port <= 0 or self.metrics_port > 65535):
            errors.append(f"METRICS_PORT must be between 1 and 65535 (got {self.metrics_port})")

        # Validate RAG parameters
        if self.max_history < 0:
            errors.append(f"APP_MAX_HISTORY must be non-negative (got {self.max_history})")
        if self.top_k_results <= 0:
            errors.append(f"APP_TOP_K_RESULTS must be positive (got {self.top_k_results})")
        if self.rerank_top_k <= 0:
            errors.append(f"RERANK_TOP_K must be positive (got {self.rerank_top_k})")

        # Validate model parameters
        if self.temperature < 0 or self.temperature > 2:
            errors.append(f"Temperature must be between 0 and 2 (got {self.temperature})")
        if self.max_tokens <= 0:
            errors.append(f"Max tokens must be positive (got {self.max_tokens})")

        # Validate cache configuration
        if self.cache_enabled and self.cache_ttl <= 0:
            errors.append(f"CACHE_TTL must be positive when cache is enabled (got {self.cache_ttl})")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  • {err}" for err in errors)
            raise ValueError(error_msg)


def get_config() -> AppConfig:
    """
    Get validated application configuration.

    Returns:
        AppConfig: Validated configuration instance.
    """
    config = AppConfig()
    config.validate()
    return config
