"""
RC Chassis Extractor + Comparator

Features:
- OCR an RC image (or first page of a PDF) using pytesseract
- Extract likely "Chassis" / "Chassis No" values via regex
- Normalize chassis strings (remove spaces, dashes, uppercase)
- Compare seller input vs extracted using fuzzy ratio (difflib)
- Returns match result, best extracted candidate, and similarity score

Dependencies:
- pip install pytesseract pillow pdf2image
- Install Tesseract OCR (system package):
    - Ubuntu: sudo apt install tesseract-ocr
    - Mac (brew): brew install tesseract
    - Windows: install from https://github.com/tesseract-ocr/tesseract/releases
- For PDF support: install poppler (pdf2image dependency)
    - Ubuntu: sudo apt install poppler-utils
    - Mac (brew): brew install poppler
    - Windows: download poppler binaries and add to PATH

Usage:
    python rc_chassis_check.py
    (or import functions in your backend)
"""

import re
import json
import difflib
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import os

# If using PDFs:
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

# -------------------------
# Utility / OCR functions
# -------------------------
def image_to_text(image_path, preprocess=True):
    """
    Load image and run OCR; returns raw text.
    Preprocessing helps OCR: grayscale, autocontrast, slight sharpening.
    """
    img = Image.open(image_path)
    if preprocess:
        img = img.convert("L")  # grayscale
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.SHARPEN)
    text = pytesseract.image_to_string(img, lang='eng')
    return text

def pdf_first_page_to_image(pdf_path, dpi=200):
    """
    Convert first page of PDF to PIL Image (requires pdf2image and poppler).
    """
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError("pdf2image not installed. Install with `pip install pdf2image` and poppler in system.")
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    if pages:
        return pages[0]
    raise RuntimeError("Could not convert PDF to image (empty).")


# -------------------------
# Extraction + normalization
# -------------------------
# Patterns to find chassis-like tokens; cover common RC labels
_CHASSIS_REGEXES = [
    r"chassis\s*no(?:\.|:)?\s*([A-Z0-9\-\/\\]+)",
    r"chassis\s*number(?:\.|:)?\s*([A-Z0-9\-\/\\]+)",
    r"chassis(?:\.|:)?\s*([A-Z0-9\-\/\\]{6,})",
    r"chasis\s*no(?:\.|:)?\s*([A-Z0-9\-\/\\]+)",  # common misspellings
    r"frame\s*no(?:\.|:)?\s*([A-Z0-9\-\/\\]+)",
    r"chassis\/chasis\s*no(?:\.|:)?\s*([A-Z0-9\-\/\\]+)"
]

def extract_chassis_candidates(raw_text):
    """
    Return list of extracted chassis-like strings from OCR text.
    """
    text = raw_text.replace("\n", " ").replace("\r", " ")
    candidates = []
    for regex in _CHASSIS_REGEXES:
        for m in re.finditer(regex, text, flags=re.IGNORECASE):
            val = m.group(1).strip()
            # clean trailing punctuation
            val = re.sub(r'[^A-Z0-9\-\/\\]', '', val, flags=re.IGNORECASE)
            if val:
                candidates.append(val)
    # As fallback: look for long alphanumeric tokens (length >= 8)
    if not candidates:
        toks = re.findall(r"[A-Z0-9\-\/\\]{8,}", text, flags=re.IGNORECASE)
        candidates.extend(toks)
    # Unique preserve order
    seen = set()
    uniq = []
    for c in candidates:
        cu = c.upper()
        if cu not in seen:
            seen.add(cu)
            uniq.append(cu)
    return uniq

def normalize_chassis(s):
    """
    Normalize chassis string: uppercase, remove spaces, replace similar chars,
    remove punctuation except alnum.
    """
    if s is None:
        return None
    s = s.upper()
    # remove spaces and common separators
    s = re.sub(r'[\s\-\:]', '', s)
    # remove non-alphanumeric characters
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

# -------------------------
# Comparison logic
# -------------------------
def similarity_ratio(a, b):
    """
    Return similarity ratio between two strings (0..1)
    """
    if a is None or b is None:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def compare_chassis(seller_chassis, extracted_candidates, threshold=0.90):
    """
    Compare seller input to extracted candidates.
    Returns dict with:
      - 'best_candidate'
      - 'normalized_seller'
      - 'normalized_candidate'
      - 'similarity' (0..1)
      - 'match' (True/False by threshold)
      - 'all_candidates' : list of (candidate, normalized, similarity)
    """
    s_norm = normalize_chassis(seller_chassis)
    best = None
    best_score = 0.0
    details = []
    for cand in extracted_candidates:
        c_norm = normalize_chassis(cand)
        score = similarity_ratio(s_norm, c_norm)
        details.append({'candidate': cand, 'normalized': c_norm, 'score': score})
        if score > best_score:
            best_score = score
            best = cand
    match = best_score >= threshold
    result = {
        'seller_input': seller_chassis,
        'normalized_seller': s_norm,
        'best_candidate': best,
        'normalized_candidate': normalize_chassis(best) if best else None,
        'similarity': round(best_score, 4),
        'match': match,
        'all_candidates': details
    }
    return result

# -------------------------
# High-level API
# -------------------------
def rc_chassis_check(seller_chassis, file_path, is_pdf=False, preprocess=True, threshold=0.90):
    """
    High-level function:
    - if is_pdf True: convert first page to image (requires pdf2image)
    - runs OCR, extracts chassis candidates, compares with seller input
    - returns a result dict (JSON-serializable)
    """
    # Step 1: get OCR text
    if is_pdf:
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("PDF support unavailable; install pdf2image and poppler.")
        pil_img = pdf_first_page_to_image(file_path)
        # save a temp image to pass to pytesseract (or directly process)
        pil_img = pil_img.convert("RGB")
        # run OCR directly on PIL Image
        text = pytesseract.image_to_string(pil_img)
    else:
        text = image_to_text(file_path, preprocess=preprocess)

    candidates = extract_chassis_candidates(text)
    compare = compare_chassis(seller_chassis, candidates, threshold=threshold)

    return {
        'ocr_text_snippet': (text[:800] + '...') if text else '',
        'extracted_candidates': candidates,
        'comparison': compare
    }

# -------------------------
# Example CLI usage
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RC Chassis Number extractor & comparator")
    parser.add_argument("--seller", required=True, help="Seller-entered chassis number (string)")
    parser.add_argument("--file", required=True, help="Path to RC image or PDF")
    parser.add_argument("--pdf", action="store_true", help="Treat file as PDF (convert first page)")
    parser.add_argument("--threshold", type=float, default=0.90, help="Similarity threshold for match (0..1)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"File not found: {args.file}")

    try:
        result = rc_chassis_check(args.seller, args.file, is_pdf=args.pdf, threshold=args.threshold)
        print(json.dumps(result, indent=2))
        # Simple human-friendly summary
        comp = result['comparison']
        print("\nSummary:")
        print(f"Seller input: {comp['seller_input']}")
        print(f"Best OCR candidate: {comp['best_candidate']} (similarity={comp['similarity']})")
        print("Match status:", "MATCH ✅" if comp['match'] else "NO MATCH ⚠️ - flag for review")
    except Exception as e:
        print("Error:", str(e))
