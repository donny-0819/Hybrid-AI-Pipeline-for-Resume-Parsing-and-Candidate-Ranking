import io
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pymupdf as fitz  # PyMuPDF
import numpy as np
from docx import Document
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None

from src.config import load_config

_WINDOWS_TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
]


def _find_poppler_bin() -> Optional[str]:
    """Locates Poppler's bin folder when it is installed (e.g. via winget) but not yet on PATH."""
    if shutil.which("pdftoppm"):
        return None
    winget_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    for exe in winget_root.glob("oschwartz10612.Poppler*/**/bin/pdftoppm.exe"):
        return str(exe.parent)
    return None


if pytesseract is not None and not shutil.which("tesseract"):
    for _candidate in _WINDOWS_TESSERACT_PATHS:
        if _candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(_candidate)
            break

_POPPLER_PATH = _find_poppler_bin()


def _projection_score(binary_inv: np.ndarray, angle: float) -> float:
    h, w = binary_inv.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(binary_inv, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    row_sums = rotated.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(row_sums) ** 2))


def _compute_skew_angle(binary_inv: np.ndarray, max_angle: float = 15.0) -> float:
    """
    Estimates the rotation (degrees, OpenCV convention) that straightens text lines.
    Projection-profile search: horizontal text lines produce the sharpest row-sum profile,
    so the best angle maximizes the squared differences between neighbouring rows.
    """
    if int(np.count_nonzero(binary_inv)) < 100:
        return 0.0
    h, w = binary_inv.shape[:2]
    scale = min(1.0, 1000.0 / float(max(h, w)))
    small = cv2.resize(binary_inv, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else binary_inv
    small = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    coarse = np.arange(-max_angle, max_angle + 1e-6, 1.0)
    best = max(coarse, key=lambda a: _projection_score(small, a))
    fine = np.arange(best - 1.0, best + 1.0 + 1e-6, 0.1)
    best = max(fine, key=lambda a: _projection_score(small, a))
    return float(round(best, 2))


def preprocess_image_opencv(
    img_bgr: np.ndarray,
    denoise_h: int = 10,
    deskew_enabled: bool = True,
    deskew_max_angle: float = 15.0,
    adaptive_binarization: bool = True,
    block_size: int = 31,
    adaptive_c: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Applies grayscale conversion, non-local means denoising, deskewing, and adaptive Gaussian binarization."""
    meta: Dict[str, Any] = {"steps": []}

    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()
    meta["steps"].append("grayscale")

    if denoise_h > 0:
        gray = cv2.fastNlMeansDenoising(gray, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
        meta["steps"].append(f"denoise(h={denoise_h})")

    skew_deg = 0.0
    if deskew_enabled:
        text_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        skew_deg = _compute_skew_angle(text_mask, max_angle=deskew_max_angle)
        if abs(skew_deg) >= 0.25:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, skew_deg, 1.0)
            gray = cv2.warpAffine(
                gray,
                rot_mat,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            meta["steps"].append(f"deskew({skew_deg:.2f}deg)")
    meta["estimated_skew_deg"] = round(skew_deg, 3)

    if adaptive_binarization:
        blk = block_size if block_size % 2 == 1 else block_size + 1
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blk,
            adaptive_c,
        )
        meta["steps"].append(f"adaptive_binarize(block={blk},C={adaptive_c})")
    else:
        processed = gray

    return processed, meta


def _sort_blocks_layout_aware(blocks: List[tuple], page_width: float) -> List[str]:
    """
    Reconstructs natural reading order for single-column, two-column, and sidebar PDF templates.
    Full-width headers/banners are read first, while narrow column blocks are ordered left-column then right-column.
    """
    text_blocks = [b for b in blocks if len(b) >= 7 and int(b[6]) == 0 and str(b[4]).strip()]
    if not text_blocks:
        return []

    midpoint = page_width * 0.5
    headers: List[tuple] = []
    left_col: List[tuple] = []
    right_col: List[tuple] = []

    for b in text_blocks:
        x0, y0, x1, y1, text = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4]).strip()
        width = x1 - x0
        # Blocks spanning across the midpoint or wider than 65% of page width are treated as full-width sections
        if width >= page_width * 0.65 or (x0 < midpoint * 0.85 and x1 > midpoint * 1.15):
            headers.append((y0, x0, text))
        elif x1 <= midpoint * 1.08:
            left_col.append((y0, x0, text))
        else:
            right_col.append((y0, x0, text))

    # If there is no meaningful two-column split, sort strictly top-to-bottom then left-to-right
    if not right_col or not left_col:
        all_sorted = sorted(
            [(round(float(b[1]) / 8.0), float(b[0]), str(b[4]).strip()) for b in text_blocks],
            key=lambda item: (item[0], item[1]),
        )
        return [item[2] for item in all_sorted]

    headers.sort(key=lambda item: (item[0], item[1]))
    left_col.sort(key=lambda item: (item[0], item[1]))
    right_col.sort(key=lambda item: (item[0], item[1]))

    # Place top header blocks (above columns) first, then left column, then right column, then bottom footers
    min_col_y = min(left_col[0][0], right_col[0][0])
    top_headers = [h[2] for h in headers if h[0] <= min_col_y + 15]
    bottom_headers = [h[2] for h in headers if h[0] > min_col_y + 15]

    ordered = top_headers + [c[2] for c in left_col] + [c[2] for c in right_col] + bottom_headers
    return ordered


def extract_pdf_digital(file_bytes: bytes) -> Tuple[str, int, float]:
    """Extracts digital text from PDF using PyMuPDF spatial block coordinates."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_texts: List[str] = []
    total_chars = 0

    for page in doc:
        blocks = page.get_text("blocks")
        ordered_blocks = _sort_blocks_layout_aware(blocks, page.rect.width)
        page_str = "\n".join(ordered_blocks).strip()
        total_chars += len(page_str)
        if page_str:
            page_texts.append(page_str)

    page_count = max(len(doc), 1)
    doc.close()
    full_text = "\n\n".join(page_texts).strip()
    char_density = total_chars / float(page_count)
    return full_text, page_count, char_density


def extract_docx_digital(file_bytes: bytes) -> Tuple[str, int]:
    """Extracts paragraphs and structured table cells from native DOCX files."""
    doc = Document(io.BytesIO(file_bytes))
    chunks: List[str] = []

    for para in doc.paragraphs:
        txt = para.text.strip()
        if txt:
            chunks.append(txt)

    for table in doc.tables:
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                c_txt = " ".join(cell.text.split()).strip()
                if c_txt and c_txt not in row_cells:
                    row_cells.append(c_txt)
            if row_cells:
                chunks.append(" | ".join(row_cells))

    full_text = "\n".join(chunks).strip()
    return full_text, 1


def _ocr_single_bgr_image(img_bgr: np.ndarray, use_opencv: bool, cfg: dict) -> Tuple[str, Dict[str, Any]]:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract and install Tesseract binary.")

    ocr_cfg = cfg["ingestion"]
    presets = ocr_cfg["opencv_presets"]
    tess_args = f"--oem {ocr_cfg['tesseract_oem']} --psm {ocr_cfg['tesseract_psm']}"

    if use_opencv:
        proc_img, prep_meta = preprocess_image_opencv(
            img_bgr,
            denoise_h=presets["denoise_h"],
            deskew_enabled=presets["deskew_enabled"],
            deskew_max_angle=presets["deskew_max_angle_deg"],
            adaptive_binarization=presets["adaptive_binarization"],
            block_size=presets["adaptive_block_size"],
            adaptive_c=presets["adaptive_c"],
        )
        pil_img = Image.fromarray(proc_img)
    else:
        prep_meta = {"steps": ["raw_unprocessed"], "estimated_skew_deg": 0.0}
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else img_bgr
        pil_img = Image.fromarray(rgb)

    text = pytesseract.image_to_string(pil_img, lang=ocr_cfg.get("tesseract_lang", "eng"), config=tess_args)
    return text.strip(), prep_meta


def extract_via_ocr(file_bytes: bytes, suffix: str, use_opencv: bool = True) -> Tuple[str, int, List[Dict[str, Any]]]:
    """Rasterizes scanned PDFs or decodes camera images and runs OpenCV + Tesseract OCR."""
    cfg = load_config()
    dpi = cfg["ingestion"]["ocr_dpi"]
    pages_bgr: List[np.ndarray] = []

    if suffix == ".pdf":
        pil_pages = []
        if convert_from_bytes is not None:
            try:
                pil_pages = convert_from_bytes(file_bytes, dpi=dpi, poppler_path=_POPPLER_PATH)
            except Exception:
                pil_pages = []
        for pil_p in pil_pages:
            rgb = np.array(pil_p.convert("RGB"))
            pages_bgr.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not pages_bgr:
            # Fallback rasterization via PyMuPDF if Poppler/pdf2image is unavailable
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                pages_bgr.append(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            doc.close()
    else:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            pil_im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            decoded = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        pages_bgr.append(decoded)

    texts: List[str] = []
    meta_list: List[Dict[str, Any]] = []
    for page_img in pages_bgr:
        t_str, m_info = _ocr_single_bgr_image(page_img, use_opencv=use_opencv, cfg=cfg)
        texts.append(t_str)
        meta_list.append(m_info)

    return "\n\n".join(texts).strip(), max(len(pages_bgr), 1), meta_list


def ingest_document(
    file_bytes: bytes,
    filename: str,
    force_ocr: bool = False,
    use_opencv: bool = True,
) -> Dict[str, Any]:
    """
    Tier 1 Entry Point: Routes uploaded document by container type and character density.
    Outputs a standardized JSON-compatible dictionary.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    density_threshold = cfg["ingestion"]["digital_char_density_threshold"]
    suffix = Path(filename).suffix.lower()

    routed_mode = "unknown"
    page_count = 1
    char_density = 0.0
    prep_details: List[Dict[str, Any]] = []

    if suffix == ".docx":
        clean_text, page_count = extract_docx_digital(file_bytes)
        char_density = float(len(clean_text))
        routed_mode = "python-docx (digital)"
    elif suffix == ".txt":
        clean_text = file_bytes.decode("utf-8", errors="replace").strip()
        char_density = float(len(clean_text))
        routed_mode = "plain-text (digital)"
    elif suffix == ".pdf":
        if not force_ocr:
            dig_text, page_count, char_density = extract_pdf_digital(file_bytes)
            if char_density >= density_threshold:
                clean_text = dig_text
                routed_mode = "PyMuPDF spatial layout (digital)"
            else:
                clean_text, page_count, prep_details = extract_via_ocr(file_bytes, suffix, use_opencv=use_opencv)
                routed_mode = "OpenCV + Tesseract OCR (scanned PDF)" if use_opencv else "Raw Tesseract OCR (scanned PDF)"
        else:
            clean_text, page_count, prep_details = extract_via_ocr(file_bytes, suffix, use_opencv=use_opencv)
            routed_mode = "OpenCV + Tesseract OCR (forced)" if use_opencv else "Raw Tesseract OCR (forced)"
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        clean_text, page_count, prep_details = extract_via_ocr(file_bytes, suffix, use_opencv=use_opencv)
        char_density = float(len(clean_text)) / float(page_count)
        routed_mode = "OpenCV + Tesseract OCR (image)" if use_opencv else "Raw Tesseract OCR (image)"
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    latency = round(time.perf_counter() - t0, 4)
    return {
        "filename": filename,
        "container_type": suffix,
        "routed_mode": routed_mode,
        "page_count": page_count,
        "embedded_char_density": round(char_density, 2),
        "preprocessing_metadata": prep_details,
        "clean_text": clean_text,
        "tier1_latency_sec": latency,
    }
