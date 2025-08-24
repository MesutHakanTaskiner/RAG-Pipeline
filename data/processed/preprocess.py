#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: PDF Preprocessing & Chunking Pipeline (English, marker-free)

This script:
- Extracts text and layout blocks with PyMuPDF (font size, bold hints)
- Detects & removes repeating headers/footers across pages
- Falls back to OCR for image-heavy/low-text pages (if pytesseract is available)
- Extracts table text with pdfplumber (if installed) and merges inline WITHOUT tags
- Infers a basic section path from heading-like blocks (size/bold heuristics)
- Splits text into semantic chunks with overlap
- Emits JSONL with rich metadata per chunk (no extra markers in the text)
"""

from __future__ import annotations
import argparse
import dataclasses
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Required
try:
    import fitz  # PyMuPDF
except Exception:
    print("PyMuPDF (fitz) is required: pip install pymupdf", file=sys.stderr)
    raise

# Optional
try:
    import pdfplumber  # table extraction
except Exception:
    pdfplumber = None

try:
    import camelot  # advanced table extraction (not used by default)
except Exception:
    camelot = None

try:
    import pytesseract  # OCR
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

TOKEN_PER_CHAR = 1.0 / 4.0  # rough token estimate


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) * TOKEN_PER_CHAR))


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def infer_year_from_path(path: Path) -> Optional[int]:
    m = re.search(r"(20\d{2})", str(path))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


@dataclasses.dataclass
class HeaderFooter:
    header_lines: List[str]
    footer_lines: List[str]


def detect_repeating_header_footer(
    page_text_lines: List[List[str]], top_k: int = 2, bottom_k: int = 2, min_repeat: float = 0.6
) -> HeaderFooter:
    """
    Find top/bottom lines that repeat on most pages.
    - page_text_lines: list of page->list of lines
    - top_k/bottom_k: how many first/last lines to check
    - min_repeat: fraction threshold (e.g., 0.6 => on ≥60% of pages)
    """
    if not page_text_lines:
        return HeaderFooter([], [])

    total_pages = len(page_text_lines)
    top_candidates: Dict[str, int] = {}
    bottom_candidates: Dict[str, int] = {}

    for lines in page_text_lines:
        if not lines:
            continue
        top = [normalize_line(x) for x in lines[:top_k] if x.strip()]
        bot = [normalize_line(x) for x in lines[-bottom_k:] if x.strip()]
        for t in top:
            top_candidates[t] = top_candidates.get(t, 0) + 1
        for b in bot:
            bottom_candidates[b] = bottom_candidates.get(b, 0) + 1

    header = [k for k, v in top_candidates.items() if v / total_pages >= min_repeat]
    footer = [k for k, v in bottom_candidates.items() if v / total_pages >= min_repeat]
    return HeaderFooter(header, footer)


@dataclasses.dataclass
class PageContent:
    page_num: int  # 1-based
    text: str
    blocks: List[dict]
    tables_text: str


FONT_BOLD_HINTS = {"Bold", "Semibold", "Demi", "Black"}


def block_is_heading(block: dict) -> bool:
    """Heading if avg font size is large or any span looks bold."""
    spans = []
    for line in block.get('lines', []):
        spans.extend(line.get('spans', []))
    if not spans:
        return False
    sizes = [s.get('size', 0) for s in spans if 'size' in s]
    fonts = [s.get('font', '') for s in spans]
    avg_size = sum(sizes) / max(1, len(sizes))
    is_bold = any(any(b in f for b in FONT_BOLD_HINTS) for f in fonts)
    return is_bold or avg_size >= 12.5


def extract_tables_with_pdfplumber(pdf_path: Path) -> Dict[int, str]:
    """Return per-page merged table text (TSV-like), if available."""
    if pdfplumber is None:
        return {}
    tables: Dict[int, str] = {}
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tbs = page.extract_tables() or []
                    texts = []
                    for tb in tbs:
                        rows = ["\t".join([c if c is not None else "" for c in row]) for row in tb]
                        if rows:
                            texts.append("\n".join(rows))
                    if texts:
                        tables[i] = "\n\n".join(texts)
                except Exception:
                    continue
    except Exception:
        return {}
    return tables


def ocr_pixmap_to_text(pix) -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def extract_page_content(doc, page_index: int, tables_map: Dict[int, str]) -> PageContent:
    page = doc.load_page(page_index)
    page_num = page_index + 1

    page_dict = page.get_text("dict")
    blocks = [b for b in page_dict.get('blocks', []) if b.get('type', 0) == 0]

    text = page.get_text("text") or ""
    text_norm = re.sub(r"\s+", " ", text).strip()
    if len(text_norm) < 30:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        ocr_text = ocr_pixmap_to_text(pix)
        if len(ocr_text.strip()) > len(text_norm):
            text = ocr_text

    tables_text = tables_map.get(page_num, "")
    return PageContent(page_num=page_num, text=text, blocks=blocks, tables_text=tables_text)
