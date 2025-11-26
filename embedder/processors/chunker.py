"""
Document chunking module for splitting text into semantic chunks.

This module uses LangChain's text splitters to break documents into
appropriately sized chunks for embedding.
"""

import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Handles splitting documents into chunks for embedding.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Initialize the document chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"Initialized chunker with size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Split a document into chunks.

        Args:
            document: Document dictionary with 'url', 'title', 'content', and 'metadata'.

        Returns:
            List of chunk dictionaries, each containing:
                - chunk_text: The text of the chunk
                - chunk_index: Index of the chunk within the document
                - url: Original document URL
                - title: Original document title
                - metadata: Original metadata plus chunk-specific info
        """
        content = document.get("content", "")

        if not content or not content.strip():
            logger.warning(f"Empty content for document: {document.get('url', 'unknown')}")
            return []

        chunks = self.splitter.split_text(content)

        chunk_dicts = []
        for idx, chunk_text in enumerate(chunks):
            chunk_dict = {
                "chunk_text": chunk_text,
                "chunk_index": idx,
                "url": document.get("url", ""),
                "title": document.get("title", ""),
                "metadata": {
                    **document.get("metadata", {}),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk_text),
                },
            }
            chunk_dicts.append(chunk_dict)

        logger.debug(
            f"Split document {document.get('url', 'unknown')} into {len(chunks)} chunks"
        )

        return chunk_dicts

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of document dictionaries.

        Returns:
            List of all chunks from all documents.
        """
        all_chunks = []

        for document in documents:
            try:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(
                    f"Error chunking document {document.get('url', 'unknown')}: {e}"
                )

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        return all_chunks
