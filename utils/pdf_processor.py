"""PDF text extraction and page-aware word chunking utilities."""

from __future__ import annotations

from typing import Any

import fitz


def _as_pdf_bytes(pdf_file: Any) -> bytes:
    """Accept bytes, a bytearray, or a Streamlit UploadedFile-like object."""
    if isinstance(pdf_file, bytes):
        return pdf_file
    if isinstance(pdf_file, bytearray):
        return bytes(pdf_file)
    if hasattr(pdf_file, "getvalue"):
        return pdf_file.getvalue()
    raise TypeError("pdf_file must be bytes or an object exposing getvalue().")


def extract_text_from_pdf(pdf_file: Any) -> list[dict[str, Any]]:
    """Extract selectable text page by page, preserving original 1-based page numbers.

    Empty pages are omitted from the returned list, but the page_number field
    always refers to the page's original position in the PDF.
    """
    pdf_bytes = _as_pdf_bytes(pdf_file)
    pages: list[dict[str, Any]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page_number": page_number, "text": text})
    return pages


def create_chunks(
    pages: list[dict[str, Any]], chunk_size: int = 500, overlap: int = 100
) -> list[dict[str, Any]]:
    """Split each page into overlapping word chunks without crossing page boundaries."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be an integer >= 0 and smaller than chunk_size.")

    chunks: list[dict[str, Any]] = []
    chunk_id = 0
    step = chunk_size - overlap

    for page in pages:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        if "page_number" not in page:
            raise ValueError("Each page must contain a page_number field.")
        words = text.split()
        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                break
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": int(page["page_number"]),
                    "text": " ".join(chunk_words),
                }
            )
            chunk_id += 1
            if start + chunk_size >= len(words):
                break
    return chunks
