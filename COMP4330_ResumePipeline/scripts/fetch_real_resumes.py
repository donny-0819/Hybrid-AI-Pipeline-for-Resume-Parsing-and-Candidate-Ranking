"""Downloads a 24-resume test corpus from the public Kaggle Resume Dataset mirror on Hugging Face
(opensporks/resumes, LiveCareer resumes) and writes degraded phone-scan copies plus a grading sheet."""
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluate_benchmarks import _rasterize_first_page  # noqa: E402
from scripts.degrade_scans import degrade_image_smartphone_simulation  # noqa: E402

REPO = "opensporks/resumes"
OUT_DIR = ROOT_DIR / "data" / "real_resumes"

SELECTION = {
    "cyber": ["52246737", "18187364", "27058381", "24913648", "20879311", "19796840"],
    "swe": ["16186411", "51588273", "12763627", "83816738", "21038022", "28126340"],
    "data": ["18067556", "27536013", "30223363", "29075857", "12011623", "13385306"],
}
CONTROL_CATEGORIES = ["CHEF", "TEACHER", "ACCOUNTANT"]
SCAN_IDS = ["52246737", "18187364", "16186411", "12763627", "18067556", "30223363"]


def main() -> None:
    df = pd.read_csv(hf_hub_download(REPO, "Resume/Resume.csv", repo_type="dataset"), dtype={"ID": str})
    category_of = dict(zip(df.ID, df.Category))

    rows = [(rid, lean) for lean, ids in SELECTION.items() for rid in ids]
    for cat in CONTROL_CATEGORIES:
        for rid in df[df.Category == cat].sample(2, random_state=7).ID:
            rows.append((rid, "non-IT control"))

    pdf_dir, scan_dir = OUT_DIR / "pdf", OUT_DIR / "scanned"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    scan_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for rid, lean in rows:
        category = category_of[rid]
        cached = hf_hub_download(REPO, f"data/data/{category}/{rid}.pdf", repo_type="dataset")
        pdf_path = pdf_dir / f"{rid}.pdf"
        shutil.copy(cached, pdf_path)
        if rid in SCAN_IDS:
            degraded = degrade_image_smartphone_simulation(_rasterize_first_page(pdf_path.read_bytes()), seed=int(rid) % 1000)
            cv2.imwrite(str(scan_dir / f"{rid}_scan.jpg"), degraded)
        manifest.append({
            "filename": pdf_path.name,
            "dataset_category": category,
            "selected_as": lean,
            "grade_JD_CYBER": "",
            "grade_JD_SWE": "",
            "grade_JD_DATA": "",
        })

    pd.DataFrame(manifest).to_csv(OUT_DIR / "grading_sheet.csv", index=False)
    print(f"Wrote {len(manifest)} PDFs, {len(SCAN_IDS)} scans, and grading_sheet.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
