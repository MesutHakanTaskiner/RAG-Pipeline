import os
import re
import io
import json
import fitz  # PyMuPDF
from PIL import Image

# Optional deps
try:
    import pdfplumber
    HAVE_PDFPLUMBER = True
except Exception:
    HAVE_PDFPLUMBER = False

try:
    import pytesseract
    OCR_OK = True
except Exception:
    OCR_OK = False

# --- Configure Tesseract path for Windows (adjust if needed) ---
WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and OCR_OK:
    try:
        if os.path.isfile(WINDOWS_TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT_PATH
    except Exception:
        pass

# --- Inputs / Outputs ---
SRC = ["docs/sr_2022_cb.pdf", "docs/sr_2022_db.pdf"]  # set your paths here
OUT_DIR = "extracted_high_quality"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------- Text normalization & cleaning -----------------
def normalize_text(s: str) -> str:
    """Light normalization: unify line breaks, collapse spaces/newlines."""
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_for_llm(text: str) -> str:
    """
    Aggressive cleanup to reduce LLM noise:
    - Remove non-printable/control chars
    - Keep letters, digits, common punctuation, whitespace
    - Collapse spaces/newlines
    """
    if not text:
        return ""
    # Remove non-printable characters
    text = "".join(ch for ch in text if ch.isprintable())

    # Allow only letters (including Turkish), digits, common punctuation & whitespace
    # (Extend the charset as needed)
    allowed_pattern = r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ.,;:!?()\[\]{}\-\–—_\"'/%&$#@ \n\t<>]"
    text = re.sub(allowed_pattern, " ", text)

    # Collapse repeated spaces and long blank runs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------- Header/Footer utilities -----------------
def detect_repeating_lines(pages_text, top_k=10, min_len=8):
    """
    Detect repeating first/last lines across pages (simple frequency heuristic).
    These are likely headers/footers and can be stripped.
    """
    from collections import Counter
    heads, tails = [], []
    for t in pages_text:
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        if not lines:
            continue
        if len(lines[0]) >= min_len:
            heads.append(lines[0])
        if len(lines[-1]) >= min_len:
            tails.append(lines[-1])
    head_common = [l for l, _ in Counter(heads).most_common(top_k)]
    tail_common = [l for l, _ in Counter(tails).most_common(top_k)]
    return set(head_common), set(tail_common)


def strip_headers_footers(text, head_set, tail_set):
    """Remove first/last line if they match detected header/footer sets."""
    lines = [l for l in text.splitlines()]
    if lines and lines[0].strip() in head_set:
        lines = lines[1:]
    if lines and lines[-1].strip() in tail_set:
        lines = lines[:-1]
    return "\n".join(lines)


# ----------------- Reading order / OCR -----------------
def extract_blocks_reading_order(page):
    """
    Create a reasonable reading order for two-column pages:
    - Split blocks by vertical midline into left/right columns
    - Sort by (y, x) within each column
    - Concatenate left then right
    """
    blocks = page.get_text("blocks") or []  # (x0, y0, x1, y1, text, ...)
    W = page.rect.width
    mid = W / 2
    left, right = [], []
    for b in blocks:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, txt = b[:5]
        if not (isinstance(txt, str) and txt.strip()):
            continue
        (left if x0 < mid else right).append((y0, x0, txt))
    left.sort(key=lambda t: (t[0], t[1]))
    right.sort(key=lambda t: (t[0], t[1]))
    ordered = [t[2] for t in left + right]
    return "\n".join(ordered)


def ocr_page(page, dpi=250, lang="eng"):
    """
    Render page to bitmap and run OCR with Tesseract if available.
    Falls back to empty string if OCR is not configured.
    """
    if not OCR_OK:
        return ""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    try:
        img = Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception:
        return ""
    try:
        return pytesseract.image_to_string(img, lang=lang)
    except Exception:
        return ""


# ----------------- Main extraction loop -----------------
for pdf in SRC:
    if not os.path.exists(pdf):
        print(f"[WARN] Missing: {pdf} (please provide a valid path)")
        continue

    stem = os.path.splitext(os.path.basename(pdf))[0]
    out_pages = os.path.join(OUT_DIR, f"{stem}_pages")
    os.makedirs(out_pages, exist_ok=True)

    pages_raw = []

    # Pass 1: text in reading order + OCR fallback for low-text pages
    with fitz.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            text = extract_blocks_reading_order(page)

            # OCR fallback if the page has very little text
            if len(text.strip()) < 100:
                ocr_txt = ocr_page(page, dpi=250, lang="eng")
                if len(ocr_txt.strip()) > len(text.strip()):
                    text = f"{text}\n{ocr_txt}"

            # Normalize early and apply aggressive cleanup for LLM
            text = clean_for_llm(normalize_text(text))
            pages_raw.append(text)

    # Detect and strip headers/footers after initial cleanup
    head_set, tail_set = detect_repeating_lines(pages_raw)
    pages_clean = []
    for i, t in enumerate(pages_raw, start=1):
        t2 = strip_headers_footers(t, head_set, tail_set)
        pages_clean.append(t2)

    # Optional: enrich tables (extract cell text) with pdfplumber
    if HAVE_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf) as pl:
                for i, page in enumerate(pl.pages, start=1):
                    try:
                        tables = page.extract_tables() or []
                        if not tables:
                            continue
                        tbl_lines = []
                        for tbl in tables:
                            for row in tbl:
                                row_text = " | ".join([((c or "").strip()) for c in row])
                                if row_text.strip():
                                    tbl_lines.append(row_text)
                        if tbl_lines:
                            # Clean table text too, to avoid weird symbols
                            table_text = clean_for_llm("\n".join(tbl_lines))
                            pages_clean[i - 1] += "\n\n[TABLE]\n" + table_text
                    except Exception:
                        # Table extraction is best-effort; ignore page-level failures
                        pass
        except Exception:
            # If pdfplumber fails globally, just skip enrichment
            pass

    # Save aggregated full text
    full_path = os.path.join(OUT_DIR, f"{stem}_full.txt")
    with open(full_path, "w", encoding="utf-8") as f:
        for i, t in enumerate(pages_clean, start=1):
            f.write(f"=== [PAGE {i}] ===\n{t}\n\n")

    # Save page stats as JSONL
    jsonl_path = os.path.join(OUT_DIR, f"{stem}_pages.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, t in enumerate(pages_clean, start=1):
            stat = {
                "page": i,
                "n_chars": len(t),
                "is_empty": len(t.strip()) == 0,
            }
            f.write(json.dumps(stat, ensure_ascii=False) + "\n")

    # Save per-page .txt files
    for i, t in enumerate(pages_clean, start=1):
        page_txt = os.path.join(out_pages, f"page_{i:03d}.txt")
        with open(page_txt, "w", encoding="utf-8") as f:
            f.write(t)

    print(f"✓ Processed: {pdf} → {full_path}")
