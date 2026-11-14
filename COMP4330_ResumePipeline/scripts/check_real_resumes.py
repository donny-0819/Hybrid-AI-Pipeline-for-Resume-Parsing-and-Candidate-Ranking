"""Runs the full pipeline on data/real_resumes and prints a sanity report (no human grades needed)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluate_benchmarks import compute_cer, compute_wer  # noqa: E402
from src.tier1_ingestion import ingest_document  # noqa: E402
from src.tier2_extraction import extract_candidate_profile  # noqa: E402
from src.tier3_matching import rank_candidates  # noqa: E402

DATA_DIR = ROOT_DIR / "data" / "real_resumes"
JD_LEAN = {"JD_CYBER": "cyber", "JD_SWE": "swe", "JD_DATA": "data"}


def main() -> None:
    sheet = pd.read_csv(DATA_DIR / "grading_sheet.csv")
    lean = dict(zip(sheet.filename, sheet.selected_as))

    profiles, texts = [], {}
    for pdf in sorted((DATA_DIR / "pdf").glob("*.pdf")):
        t1 = ingest_document(pdf.read_bytes(), pdf.name)
        texts[pdf.stem] = t1["clean_text"]
        prof = extract_candidate_profile(t1["clean_text"], pdf.name)
        prof["tier1_latency_sec"] = t1["tier1_latency_sec"]
        profiles.append(prof)
    print(f"Parsed {len(profiles)} PDFs; avg skills found per resume: "
          f"{sum(len(p['skills']) for p in profiles) / len(profiles):.1f}")

    print("\nTop-5 by job description (which group each resume was picked for):")
    for jd in json.loads((ROOT_DIR / "data" / "sample_jds.json").read_text(encoding="utf-8")):
        for method in ("keyword", "sbert", "hybrid"):
            ranked = [r["filename"] for r in rank_candidates(profiles, jd["description"], jd["mandatory_skills"], method=method)]
            top5 = [lean[f] for f in ranked[:5]]
            hits = sum(1 for g in top5 if g == JD_LEAN[jd["jd_id"]])
            best_control = min(ranked.index(f) + 1 for f in ranked if lean[f] == "non-IT control")
            print(f"  {jd['jd_id']:9} {method:8} matching-group in top5: {hits}/5  "
                  f"best non-IT rank: {best_control:2}  top5: {top5}")

    print("\nScanned copies vs. their digital PDF (first page):")
    for scan in sorted((DATA_DIR / "scanned").glob("*.jpg")):
        rid = scan.stem.replace("_scan", "")
        for use_cv in (False, True):
            ocr = ingest_document(scan.read_bytes(), scan.name, use_opencv=use_cv)["clean_text"]
            ref = " ".join(texts[rid].split()[: max(len(ocr.split()), 1)])
            print(f"  {rid} {'opencv' if use_cv else 'raw   '} CER {compute_cer(ref, ocr):.3f}  WER {compute_wer(ref, ocr):.3f}")


if __name__ == "__main__":
    main()
