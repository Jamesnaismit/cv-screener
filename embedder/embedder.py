"""
Main embedder module for processing CV documents and generating embeddings.

This module reads CV PDFs, chunks them, generates embeddings,
and stores them in the PostgreSQL vector database.
"""

import argparse
import logging
import sys
from typing import List, Dict, Optional

from config import get_config
from processors import DocumentChunker, EmbeddingGenerator
from database import VectorStore
from loaders import CVLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """
    Main embedder class for processing documents and generating embeddings.
    """

    def __init__(self, config) -> None:
        """
        Initialize the document embedder.

        Args:
            config: EmbedderConfig instance.
        """
        self.config = config
        self.loader = CVLoader(config.input_dir)
        self.chunker = DocumentChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.embedding_generator = EmbeddingGenerator(
            api_key=config.openai_api_key,
            model=config.embedding_model,
            batch_size=config.batch_size,
        )
        self.vector_store = VectorStore(config.database_url)

    def load_documents(self) -> List[Dict]:
        """
        Load documents from the input directory.

        Returns:
            List of document dictionaries.
        """
        documents = []

        return self.loader.load()

    def process_documents(
            self, documents: List[Dict], force_reprocess: bool = False
    ) -> None:
        """
        Process documents: chunk, embed, and store.

        Args:
            documents: List of document dictionaries.
            force_reprocess: If True, reprocess even if document exists in DB.
        """
        if not documents:
            logger.warning("No documents to process")
            return

        logger.info(f"Processing {len(documents)} documents")

        total_chunks = 0
        total_embeddings = 0

        for idx, document in enumerate(documents):
            url = document.get("url", "unknown")
            content_hash = document.get("content_hash", "")
            existing: Optional[Dict] = None

            if url != "unknown":
                existing = self.vector_store.get_document_info(url)

            logger.info(f"Processing document {idx + 1}/{len(documents)}: {url}")

            try:

                if (
                        existing
                        and not force_reprocess
                        and existing.get("content_hash") == content_hash
                ):
                    logger.info(f"Document unchanged, skipping: {url}")
                    continue

                chunks = self.chunker.chunk_document(document)
                if not chunks:
                    logger.warning(f"No chunks created for document: {url}")
                    continue

                total_chunks += len(chunks)

                chunks_with_embeddings = self.embedding_generator.generate_embeddings(
                    chunks
                )

                document_id = self.vector_store.insert_document(
                    url=document["url"],
                    title=document.get("title", ""),
                    content=document.get("content", ""),
                    metadata=document.get("metadata", {}),
                    content_hash=content_hash,
                )

                if force_reprocess or (existing and existing.get("content_hash") != content_hash):
                    self.vector_store.delete_embeddings_for_document(document_id)

                inserted = self.vector_store.insert_embeddings(chunks_with_embeddings)
                total_embeddings += inserted

                logger.info(
                    f"Processed document {idx + 1}/{len(documents)}: "
                    f"{len(chunks)} chunks, {inserted} embeddings"
                )

            except Exception as e:
                logger.error(f"Error processing document {url}: {e}")

        logger.info(
            f"Processing complete. "
            f"Total chunks: {total_chunks}, "
            f"Total embeddings: {total_embeddings}"
        )

    def show_stats(self) -> None:
        """Display database statistics."""
        doc_count = self.vector_store.get_document_count()
        emb_count = self.vector_store.get_embedding_count()

        logger.info("=" * 50)
        logger.info("Database Statistics")
        logger.info("=" * 50)
        logger.info(f"Total documents: {doc_count}")
        logger.info(f"Total embeddings: {emb_count}")
        logger.info("=" * 50)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.vector_store:
            self.vector_store.close()


def main() -> None:
    """Main entry point for the embedder."""
    parser = argparse.ArgumentParser(description="CV Screener Embedder")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all documents",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics only",
    )
    args = parser.parse_args()

    config = get_config()
    logger.setLevel(config.log_level)

    embedder = DocumentEmbedder(config)

    try:
        if args.stats:
            embedder.show_stats()
        else:
            documents = embedder.load_documents()
            embedder.process_documents(documents, force_reprocess=args.force)
            embedder.show_stats()

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        embedder.cleanup()


if __name__ == "__main__":
    main()
