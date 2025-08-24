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
