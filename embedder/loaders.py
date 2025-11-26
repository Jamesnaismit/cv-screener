"""
Loaders for ingesting raw documents into the embedding pipeline.

The CV loader reads PDF resumes from the configured feed directory and
normalizes them into the JSON structure expected by the embedder.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class CVLoader:
    """Load CVs from PDF files stored in a directory."""

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir

    def load(self) -> List[Dict]:
        """Load and normalize all PDF files in the input directory."""
        documents: List[Dict] = []

        pdf_files = sorted(self.input_dir.glob("*.pdf"))
        logger.info("Found %s PDF resumes in %s", len(pdf_files), self.input_dir)

        if not pdf_files:
            logger.warning("No PDF files found in %s", self.input_dir)
            return documents

        for pdf_path in pdf_files:
            try:
                text, page_count = self._extract_text(pdf_path)

                if not text.strip():
                    logger.warning("Empty text extracted from %s, skipping", pdf_path.name)
                    continue

                candidate_name = self._guess_candidate_name(pdf_path)
                content_hash = self._compute_hash(text)

                documents.append({
                    "url": f"cv://{pdf_path.stem}",
                    "title": candidate_name,
                    "content": text,
                    "metadata": {
                        "candidate": candidate_name,
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "document_type": "resume",
                        "pages": page_count,
                    },
                    "content_hash": content_hash,
                })

            except Exception as exc:
                logger.error("Failed to process %s: %s", pdf_path.name, exc)

        logger.info("Loaded %s CVs for embedding", len(documents))
        return documents

    @staticmethod
    def _extract_text(path: Path) -> tuple[str, int]:
        """Extract text from a PDF file."""
        reader = PdfReader(str(path))
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            cleaned = page_text.strip()
            if cleaned:
                pages.append(cleaned)

        combined_text = "\n\n".join(pages)
        return combined_text, len(reader.pages)

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute deterministic hash for change detection."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _guess_candidate_name(path: Path) -> str:
        """Infer a readable candidate name from the file name."""
        stem = path.stem
        cleaned = re.sub(r"^cv[-_]*\d*[-_]*", "", stem, flags=re.IGNORECASE)
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        cleaned = " ".join(part for part in cleaned.split() if part)
        return cleaned.title() if cleaned else stem
