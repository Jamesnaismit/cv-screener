"""
Hybrid retrieval and re-ranking module.

This module implements hybrid search combining:
1. Vector similarity (semantic search)
2. BM25 keyword search (lexical search)
3. Re-ranking to combine both approaches
"""

import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import psycopg2

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining vector similarity and BM25 keyword search.
    """

    def __init__(
        self,
        vector_retriever,
        database_url: str,
        alpha: float = 0.5,
        top_k_vector: int = 20,
        top_k_bm25: int = 20,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: VectorRetriever instance for semantic search.
            database_url: PostgreSQL connection URL.
            alpha: Weight for combining scores (0=BM25 only, 1=vector only).
            top_k_vector: Number of results to retrieve from vector search.
            top_k_bm25: Number of results to retrieve from BM25 search.
        """
        self.vector_retriever = vector_retriever
        self.database_url = database_url
        self.alpha = alpha
        self.top_k_vector = top_k_vector
        self.top_k_bm25 = top_k_bm25
        self.conn = None
        self.bm25_index = None
        self.documents_cache = []
        self._connect()

    def _connect(self) -> None:
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(self.database_url)
            logger.info("Hybrid retriever connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _ensure_connection(self) -> None:
        """Ensure database connection is active."""
        if self.conn is None or self.conn.closed:
            logger.warning("Database connection lost, reconnecting...")
            self._connect()

    def _load_documents_for_bm25(self) -> None:
        """
        Load all documents into memory for BM25 indexing.
        
        Note: For large datasets, consider using a dedicated search engine
        like Elasticsearch or using document sampling.
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.id,
                    e.chunk_text,
                    e.metadata,
                    d.url,
                    d.title
                FROM embeddings e
                JOIN documents d ON e.document_id = d.id
                """
            )

            self.documents_cache = []
            tokenized_corpus = []

            for row in cur.fetchall():
                doc = {
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "url": row[3],
                    "title": row[4],
                }
                self.documents_cache.append(doc)
                # Tokenize for BM25 (simple whitespace tokenization)
                tokenized_corpus.append(doc["content"].lower().split())

        # Build BM25 index
        if tokenized_corpus:
            self.bm25_index = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 index built with {len(self.documents_cache)} documents")
        else:
            logger.warning("No documents found for BM25 indexing")

    def _ensure_bm25_index(self) -> None:
        """Ensure BM25 index is loaded."""
        if self.bm25_index is None or not self.documents_cache:
            self._load_documents_for_bm25()

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of documents with BM25 scores.
        """
        self._ensure_bm25_index()

        if not self.bm25_index:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant results
                doc = self.documents_cache[idx].copy()
                doc["bm25_score"] = float(scores[idx])
                results.append(doc)

        logger.info(f"BM25 search returned {len(results)} results")
        return results

    def _normalize_scores(self, results: List[Dict[str, Any]], score_key: str) -> None:
        """
        Normalize scores to [0, 1] range using min-max scaling.

        Args:
            results: List of results with scores.
            score_key: Key name for the score to normalize.
        """
        if not results:
            return

        scores = [r[score_key] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score > 0:
            for r in results:
                r[f"{score_key}_normalized"] = (r[score_key] - min_score) / (
                    max_score - min_score
                )
        else:
            for r in results:
                r[f"{score_key}_normalized"] = 1.0

    def _merge_and_rerank(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge and re-rank results from vector and BM25 search.

        Args:
            vector_results: Results from vector similarity search.
            bm25_results: Results from BM25 search.

        Returns:
            Merged and re-ranked results.
        """
        # Normalize scores
        self._normalize_scores(vector_results, "similarity")
        self._normalize_scores(bm25_results, "bm25_score")

        # Create lookup for BM25 scores by document ID
        bm25_lookup = {r["id"]: r["bm25_score_normalized"] for r in bm25_results}
        
        # Create lookup for vector scores by content (since BM25 uses different IDs)
        vector_lookup = {r["content"]: r["similarity_normalized"] for r in vector_results}

        # Merge results
        merged = {}

        # Add vector results
        for r in vector_results:
            key = r["content"]
            merged[key] = {
                **r,
                "vector_score": r.get("similarity_normalized", 0),
                "bm25_score_norm": 0,
            }

        # Add BM25 results and update scores
        for r in bm25_results:
            key = r["content"]
            if key in merged:
                # Document found in both - update BM25 score
                merged[key]["bm25_score_norm"] = r["bm25_score_normalized"]
            else:
                # Document only in BM25
                merged[key] = {
                    **r,
                    "vector_score": 0,
                    "bm25_score_norm": r["bm25_score_normalized"],
                }

        # Calculate hybrid scores
        for doc in merged.values():
            doc["hybrid_score"] = (
                self.alpha * doc["vector_score"]
                + (1 - self.alpha) * doc["bm25_score_norm"]
            )

        # Sort by hybrid score
        reranked = sorted(
            merged.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        logger.info(f"Merged and re-ranked {len(reranked)} unique results")
        return reranked

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Search query.
            top_k: Number of final results to return.

        Returns:
            List of re-ranked documents.
        """
        final_top_k = top_k or self.top_k_vector

        logger.info(f"Starting hybrid retrieval for: {query[:50]}...")

        # 1. Vector similarity search
        vector_results = self.vector_retriever.retrieve(
            query, top_k=self.top_k_vector
        )

        # 2. BM25 keyword search
        bm25_results = self._bm25_search(query, top_k=self.top_k_bm25)

        # 3. Merge and re-rank
        hybrid_results = self._merge_and_rerank(vector_results, bm25_results)

        # 4. Return top-k
        final_results = hybrid_results[:final_top_k]

        logger.info(
            f"Hybrid retrieval returned {len(final_results)} results "
            f"(vector: {len(vector_results)}, BM25: {len(bm25_results)})"
        )

        return final_results

    def refresh_bm25_index(self) -> None:
        """Refresh BM25 index with latest documents."""
        logger.info("Refreshing BM25 index...")
        self._load_documents_for_bm25()

    def close(self) -> None:
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Hybrid retriever database connection closed")


class SimpleReranker:
    """
    Simple re-ranker that can be applied after retrieval.
    
    Implements various re-ranking strategies without requiring BM25.
    """

    @staticmethod
    def rerank_by_diversity(
        results: List[Dict[str, Any]],
        top_k: int,
        similarity_threshold: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results to promote diversity by removing very similar documents.

        Args:
            results: List of retrieved documents.
            top_k: Number of results to return.
            similarity_threshold: Threshold for considering documents similar.

        Returns:
            Re-ranked diverse results.
        """
        if not results:
            return []

        diverse_results = [results[0]]  # Always include top result
        
        for doc in results[1:]:
            if len(diverse_results) >= top_k:
                break
                
            # Simple diversity check based on URL
            is_diverse = True
            for selected in diverse_results:
                if doc.get("url") == selected.get("url"):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(doc)

        logger.info(f"Diversity re-ranking: {len(results)} -> {len(diverse_results)}")
        return diverse_results

    @staticmethod
    def rerank_by_recency(
        results: List[Dict[str, Any]],
        recency_weight: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results considering recency (if metadata contains timestamps).

        Args:
            results: List of retrieved documents.
            recency_weight: Weight for recency factor.

        Returns:
            Re-ranked results.
        """
        # This is a placeholder - implement based on your metadata structure
        # For now, just return original results
        logger.info("Recency re-ranking (not implemented)")
        return results
