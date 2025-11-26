"""
API entry point for the CV Screener RAG system.

This module initializes FastAPI and exposes endpoints to query the
vector-backed knowledge base populated with CVs from the feed folder.
"""

import logging
import sys
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from config import get_config
from rag import VectorRetriever, ConversationalRAGChain
from rag.cache import create_cache
from rag.metrics import init_metrics
from rag.reranker import HybridRetriever
from rag.prompts import PromptOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Payload for a query request."""

    question: str = Field(..., min_length=1, description="User question about a candidate")
    top_k: Optional[int] = Field(None, ge=1, description="Override default number of sources to retrieve")


class Source(BaseModel):
    """Source document returned alongside answers."""

    title: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    similarity: Optional[float] = None
    metadata: Optional[dict] = None


class QueryResponse(BaseModel):
    """Standard response for RAG queries."""

    answer: str
    sources: List[Source]
    metadata: dict


def _print_startup_banner(config) -> None:
    """Print application startup banner."""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                    CV Screener API                         ║
    ║        Conversational RAG over candidate resumes           ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info("Starting CV Screener API v1.0")
    logger.info("─" * 60)


def _print_configuration(config) -> None:
    """Print configuration summary."""
    logger.info("Configuration:")
    logger.info(f"  • Model: {config.model_name}")
    logger.info(f"  • Embedding: {config.embedding_model}")
    logger.info(f"  • Top-K Results: {config.top_k_results}")
    logger.info(f"  • Max History: {config.max_history}")
    logger.info(f"  • Cache: {'✓' if config.cache_enabled else '✗'}")
    logger.info(f"  • Re-ranking: {'✓' if config.rerank_enabled else '✗'}")
    logger.info(f"  • Metrics: {'✓' if config.metrics_enabled else '✗'}")
    logger.info("─" * 60)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    logger.setLevel(config.log_level)

    _print_startup_banner(config)
    _print_configuration(config)

    try:
        logger.info("Initializing system components...")

        metrics = None
        if config.metrics_enabled:
            logger.info("  ► Metrics collector...")
            metrics = init_metrics(enabled=True, port=config.metrics_port)
            logger.info("    ✓ Metrics ready")

        cache = None
        if config.cache_enabled:
            logger.info("  ► Response cache...")
            cache = create_cache(
                redis_url=config.redis_url,
                ttl=config.cache_ttl,
                enabled=True,
            )
            logger.info("    ✓ Cache ready")

        logger.info("  ► Vector retriever...")
        vector_retriever = VectorRetriever(
            database_url=config.database_url,
            openai_api_key=config.openai_api_key,
            embedding_model=config.embedding_model,
            top_k=config.top_k_results,
        )
        logger.info("    ✓ Vector retriever ready")

        if config.rerank_enabled:
            logger.info("  ► Hybrid retriever with re-ranking...")
            retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                database_url=config.database_url,
                alpha=0.5,
                top_k_vector=config.rerank_top_k,
                top_k_bm25=config.rerank_top_k,
            )
            logger.info("    ✓ Hybrid retriever ready")
        else:
            retriever = vector_retriever

        logger.info("  ► OpenAI client...")
        openai_client = OpenAI(api_key=config.openai_api_key)
        logger.info("    ✓ OpenAI client ready")

        logger.info("  ► Prompt optimizer (CV tuned)...")
        prompt_optimizer = PromptOptimizer(
            use_few_shot=True,
            validate_output=True,
            auto_augment_short_queries=True,
            openai_client=openai_client,
        )
        logger.info("    ✓ Prompt optimizer ready")

        logger.info("  ► Conversational RAG chain...")
        rag_chain = ConversationalRAGChain(
            retriever=retriever,
            openai_api_key=config.openai_api_key,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_history=config.max_history,
            cache=cache,
            metrics=metrics,
            prompt_optimizer=prompt_optimizer,
        )
        logger.info("    ✓ RAG chain ready")

        app = FastAPI(
            title="CV Screener API",
            version="1.0.0",
            description="RAG API for querying candidate resumes ingested from /feed.",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.state.config = config
        app.state.rag_chain = rag_chain
        app.state.retriever = retriever
        app.state.vector_retriever = vector_retriever

        @app.get("/health")
        async def health() -> dict:
            """Simple readiness endpoint."""
            return {"status": "ok", "message": "CV Screener API ready"}

        @app.post("/query", response_model=QueryResponse)
        async def query(payload: QueryRequest) -> QueryResponse:
            """Execute a RAG query over the embedded CVs."""
            question = payload.question.strip()
            if not question:
                raise HTTPException(status_code=400, detail="Question cannot be empty")

            try:
                answer, sources = rag_chain.query(
                    question,
                    top_k=payload.top_k or config.top_k_results,
                )
                return QueryResponse(
                    answer=answer,
                    sources=[Source(**src) for src in sources],
                    metadata={
                        "model": config.model_name,
                        "retrieved": len(sources),
                        "top_k": payload.top_k or config.top_k_results,
                    },
                )
            except Exception as exc:
                logger.error("Error processing query: %s", exc, exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to process query")

        @app.on_event("shutdown")
        async def shutdown_event() -> None:
            """Clean up resources on shutdown."""
            logger.info("Shutting down services...")
            if hasattr(app.state, "vector_retriever"):
                app.state.vector_retriever.close()
                logger.info("  ✓ Vector retriever closed")
            if hasattr(app.state, "retriever") and hasattr(app.state.retriever, "close"):
                app.state.retriever.close()
                logger.info("  ✓ Retriever closed")

        logger.info("─" * 60)
        logger.info("API ready:")
        logger.info(f"   REST: http://localhost:{config.port}/query")
        if config.metrics_enabled:
            logger.info(f"   Metrics: http://localhost:{config.metrics_port}")
        logger.info("─" * 60)

        return app

    except Exception as e:
        logger.error("─" * 60)
        logger.error(f"❌ Application error: {e}", exc_info=True)
        logger.error("─" * 60)
        sys.exit(1)


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=app.state.config.port,
        log_level=logging.getLevelName(logger.level).lower(),
    )
