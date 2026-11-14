import argparse
from pathlib import Path

import cv2
import pymupdf as fitz  # PyMuPDF
import numpy as np


def degrade_image_smartphone_simulation(
    img_bgr: np.ndarray,
    skew_deg: float = 4.2,
    noise_sigma: float = 14.0,
    shadow_strength: float = 0.35,
    seed: int = 42,
) -> np.ndarray:
    """
    Applies controlled smartphone scan artifacts:
    1. Rotation skew
    2. Uneven directional illumination gradient
    3. Slight optical defocus blur
    4. Gaussian sensor noise + JPEG compression artifacts
    """
    rng = np.random.default_rng(seed)
    h, w = img_bgr.shape[:2]

    # 1. Rotation skew
    rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), skew_deg, 0.98)
    warped = cv2.warpAffine(img_bgr, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(232, 230, 225))

    # 2. Uneven illumination gradient across page
    x_grad = np.linspace(1.0 - shadow_strength, 1.0, w, dtype=np.float32)
    y_grad = np.linspace(1.0, 1.0 - (shadow_strength * 0.6), h, dtype=np.float32)
    illum_map = np.outer(y_grad, x_grad)[:, :, None]
    lit = np.clip(warped.astype(np.float32) * illum_map, 0, 255)

    # 3. Slight camera blur
    blurred = cv2.GaussianBlur(lit, (3, 3), sigmaX=0.7)

    # 4. Sensor noise
    noise = rng.normal(0.0, noise_sigma, size=blurred.shape).astype(np.float32)
    noisy = np.clip(blurred + noise, 0, 255).astype(np.uint8)

    # 5. Mild JPEG compression artifact
    ok, enc = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    if ok:
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return noisy


def process_directory(input_dir: Path, output_dir: Path, dpi: int = 220, skew_deg: float = 4.2) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for path in sorted(input_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            doc = fitz.open(str(path))
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for page_idx, page in enumerate(doc):
                pix = page.get_pixmap(matrix=mat, alpha=False)
                rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                degraded = degrade_image_smartphone_simulation(bgr, skew_deg=skew_deg, seed=42 + count)
                out_file = output_dir / f"{path.stem}_page{page_idx + 1}_scan.jpg"
                cv2.imwrite(str(out_file), degraded)
                count += 1
            doc.close()
        elif suffix in {".png", ".jpg", ".jpeg"}:
            bgr = cv2.imread(str(path))
            if bgr is not None:
                degraded = degrade_image_smartphone_simulation(bgr, skew_deg=skew_deg, seed=42 + count)
                out_file = output_dir / f"{path.stem}_scan.jpg"
                cv2.imwrite(str(out_file), degraded)
                count += 1

    print(f"Generated {count} degraded smartphone-scan image(s) in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create controlled degraded smartphone scans from clean resumes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with clean PDFs or images")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination folder for degraded scan JPGs")
    parser.add_argument("--skew", type=float, default=4.2, help="Rotation skew angle in degrees")
    args = parser.parse_args()
    process_directory(Path(args.input_dir), Path(args.output_dir), skew_deg=args.skew)
