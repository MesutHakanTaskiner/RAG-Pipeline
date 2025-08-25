#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF Preprocessing & Chunking (marker-free)
- Walks data/raw/<YEAR>/ directories (e.g., 2022, 2023, 2024)
- For each PDF, writes its OWN JSONL under data/processed/<YEAR>/chunks_<DOCID>.jsonl
- No argparse; configuration is defined as constants below
"""

from __future__ import annotations
import dataclasses
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================
# Configuration
# =============================
RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
MIN_CHUNK_CHARS = 800
MAX_CHUNK_CHARS = 1200
OVERLAP = 180
YEAR_DIR_PATTERN = re.compile(r"^(20\d{2})$")  # e.g., 2022, 2023, 2024

# ===============
# Dependencies
# ===============
try:
    import fitz  # PyMuPDF
except Exception:
    print("PyMuPDF (fitz) is required: pip install pymupdf", file=sys.stderr)
    raise

try:
    import pdfplumber  # optional: table extraction
except Exception:
    pdfplumber = None

try:
    import pytesseract  # optional: OCR
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# ===============
# Helpers
# ===============
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


def sanitize_name(s: str) -> str:
    """Make a filesystem- and URL-friendly identifier from a filename stem."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_").lower()

# ===============
# Header/Footer detection
# ===============
@dataclasses.dataclass
class HeaderFooter:
    header_lines: List[str]
    footer_lines: List[str]


def detect_repeating_header_footer(page_text_lines: List[List[str]], top_k: int = 2, bottom_k: int = 2,
                                   min_repeat: float = 0.6) -> HeaderFooter:
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

# ===============
# Page extraction + OCR + tables
# ===============
@dataclasses.dataclass
class PageContent:
    page_num: int  # 1-based
    text: str
    blocks: List[dict]
    tables_text: str


FONT_BOLD_HINTS = {"Bold", "Semibold", "Demi", "Black"}


def block_is_heading(block: dict) -> bool:
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
                        rows = ["\t".join([(c or "") for c in row]) for row in tb]
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

# ===============
# Section paths + chunking
# ===============
@dataclasses.dataclass
class Chunk:
    id: str
    doc_id: str
    year: Optional[int]
    page_start: int
    page_end: int
    section_path: str
    text: str
    tokens_est: int
    source_path: str
    sha256: str


def build_section_paths(page_contents: List[PageContent]) -> List[Tuple[str, int, int, str]]:
    paths: List[Tuple[str, int, int, str]] = []
    current_path: List[str] = []
    for pc in page_contents:
        for bi, b in enumerate(pc.blocks):
            txt = " ".join(span.get('text', '') for line in b.get('lines', []) for span in line.get('spans', []))
            txt = normalize_line(txt)
            if not txt:
                continue
            if block_is_heading(b):
                if re.match(r"^(\d+[\.|\)]\s+.*)$", txt):
                    current_path = [txt]
                else:
                    if len(current_path) >= 3:
                        current_path = current_path[:2]
                    current_path.append(txt)
                paths.append((" / ".join(current_path), pc.page_num, bi, txt))
    return paths


def split_text_with_overlap(text: str, min_chars: int, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[\.!?…])\s+", text)
    chunks: List[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 <= max_chars:
            buf = (buf + " " + sent).strip()
        else:
            if len(buf) >= min_chars:
                chunks.append(buf)
                carry = buf[-overlap:] if overlap > 0 and len(buf) > overlap else ""
                buf = (carry + " " + sent).strip()
            else:
                while len(buf + " " + sent) > max_chars:
                    part = (buf + " " + sent)[:max_chars]
                    chunks.append(part.strip())
                    carry = part[-overlap:] if overlap > 0 and len(part) > overlap else ""
                    sent = carry + (buf + " " + sent)[max_chars:]
                    buf = ""
                buf = (buf + " " + sent).strip()
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def chunk_document(doc_id: str, year: Optional[int], page_contents: List[PageContent],
                   sha256sum: str, source_path: str,
                   min_chars: int, max_chars: int, overlap: int) -> List[Chunk]:
    paths = build_section_paths(page_contents)

    page_texts: List[Tuple[int, str]] = []
    page_lines_for_header: List[List[str]] = []
    for pc in page_contents:
        combined = pc.text
        if pc.tables_text:
            combined = (combined + "\n\n" + pc.tables_text).strip()
        lines = [l for l in pc.text.splitlines() if normalize_line(l)]
        page_lines_for_header.append(lines)
        page_texts.append((pc.page_num, combined))

    hf = detect_repeating_header_footer(page_lines_for_header)

    def clean_hf(text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        for l in lines:
            nl = normalize_line(l)
            if nl and (nl in hf.header_lines or nl in hf.footer_lines):
                continue
            cleaned.append(l)
        out = "\n".join(cleaned)
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    all_text_parts = []
    for (_, txt) in page_texts:
        cleaned = clean_hf(txt)
        if cleaned:
            all_text_parts.append(cleaned)
    full_text = "\n\n".join(all_text_parts).strip()

    page_start = page_contents[0].page_num if page_contents else 1
    page_end = page_contents[-1].page_num if page_contents else 1

    raw_chunks = split_text_with_overlap(full_text, min_chars, max_chars, overlap)

    sections_sorted = sorted(paths, key=lambda x: (x[1], x[2]))

    def infer_section_for_chunk(ch_text: str) -> str:
        best = ""
        for spath, _, _, title in sections_sorted:
            if title and title in ch_text:
                best = spath
        return best or (sections_sorted[-1][0] if sections_sorted else "")

    chunks: List[Chunk] = []
    for i, ch in enumerate(raw_chunks, start=1):
        section_path = infer_section_for_chunk(ch)
        chunks.append(Chunk(
            id=f"{doc_id}::chunk_{i:04d}",
            doc_id=doc_id,
            year=year,
            page_start=page_start,
            page_end=page_end,
            section_path=section_path,
            text=ch,
            tokens_est=estimate_tokens(ch),
            source_path=str(source_path),
            sha256=sha256sum,
        ))
    return chunks

# ===============
# Directory walking: per-PDF JSONL (NO argparse)
# ===============

def process_pdf(path: Path, min_chars: int, max_chars: int, overlap: int) -> List[Chunk]:
    doc = fitz.open(str(path))
    sha = sha256_of_file(path)
    doc_id = sanitize_name(path.stem)
    year = infer_year_from_path(path)

    tables_map = extract_tables_with_pdfplumber(path)

    page_contents: List[PageContent] = []
    for i in range(len(doc)):
        page_contents.append(extract_page_content(doc, i, tables_map))

    return chunk_document(
        doc_id=doc_id,
        year=year,
        page_contents=page_contents,
        sha256sum=sha,
        source_path=str(path),
        min_chars=min_chars,
        max_chars=max_chars,
        overlap=overlap,
    )


def write_jsonl(chunks: List[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for ch in chunks:
            f.write(json.dumps(dataclasses.asdict(ch), ensure_ascii=False) + "\n")


def main() -> None:
    if not RAW_ROOT.exists():
        print(f"RAW_ROOT does not exist: {RAW_ROOT}")
        sys.exit(1)

    year_dirs = [p for p in RAW_ROOT.iterdir() if p.is_dir() and YEAR_DIR_PATTERN.match(p.name)]
    year_dirs.sort(key=lambda p: p.name)

    if not year_dirs:
        print(f"No year directories under {RAW_ROOT}")
        sys.exit(1)

    for ydir in year_dirs:
        year = int(ydir.name)
        pdfs = sorted(ydir.glob("*.pdf"))
        if not pdfs:
            print(f"[SKIP] {ydir} has no PDFs")
            continue

        out_dir = PROCESSED_ROOT / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Year {year} ===")
        for p in pdfs:
            try:
                chunks = process_pdf(p, MIN_CHUNK_CHARS, MAX_CHUNK_CHARS, OVERLAP)
                doc_id = sanitize_name(p.stem)
                out_path = out_dir / f"chunks_{doc_id}.jsonl"
                write_jsonl(chunks, out_path)
                print(f"  [OK] {p.name} → {out_path} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  [ERROR] {p}: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
