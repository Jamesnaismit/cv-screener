"""
Configuration module for the embedder service.

This module handles loading and validating configuration from environment variables.
"""

import os
from pathlib import Path
from typing import Optional


class EmbedderConfig:
    """Configuration class for the embedder service."""

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.database_url: str = os.getenv("DATABASE_URL", "")
        self.input_dir: Path = Path(
            os.getenv("EMBEDDER_INPUT_DIR", "/data/feed")
        )
        self.batch_size: int = int(os.getenv("EMBEDDER_BATCH_SIZE", "100"))
        self.chunk_size: int = int(os.getenv("EMBEDDER_CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("EMBEDDER_CHUNK_OVERLAP", "200"))
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.embedding_dimension: int = int(
            os.getenv("EMBEDDING_DIMENSION", "1536")
        )
        self.log_level: str = os.getenv("EMBEDDER_LOG_LEVEL", "INFO")
        self.max_retries: int = 3
        self.retry_delay: int = 2

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        if self.batch_size <= 0:
            raise ValueError("EMBEDDER_BATCH_SIZE must be positive")
        if self.chunk_size <= 0:
            raise ValueError("EMBEDDER_CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("EMBEDDER_CHUNK_OVERLAP must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("EMBEDDER_CHUNK_OVERLAP must be less than EMBEDDER_CHUNK_SIZE")
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")


def get_config() -> EmbedderConfig:
    """
    Get validated embedder configuration.

    Returns:
        EmbedderConfig: Validated configuration instance.
    """
    config = EmbedderConfig()
    config.validate()
    return config
