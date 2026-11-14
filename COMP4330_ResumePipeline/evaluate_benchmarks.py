import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import cv2
import pymupdf as fitz  # PyMuPDF
import numpy as np

from scripts.degrade_scans import degrade_image_smartphone_simulation
from src.config import canonicalize_skill
from src.tier1_ingestion import ingest_document
from src.tier2_extraction import extract_candidate_profile
from src.tier3_matching import rank_candidates

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT_DIR / "data" / "sample_resumes"


def _levenshtein_distance(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """Space-efficient dynamic programming edit distance."""
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        a_item = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if a_item == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def compute_cer(ground_truth: str, ocr_hypothesis: str) -> float:
    """Character Error Rate (CER)."""
    gt = " ".join(ground_truth.split())
    hyp = " ".join(ocr_hypothesis.split())
    if not gt:
        return 0.0 if not hyp else 1.0
    return round(_levenshtein_distance(list(gt), list(hyp)) / float(len(gt)), 4)


def compute_wer(ground_truth: str, ocr_hypothesis: str) -> float:
    """Word Error Rate (WER)."""
    gt_words = ground_truth.split()
    hyp_words = ocr_hypothesis.split()
    if not gt_words:
        return 0.0 if not hyp_words else 1.0
    return round(_levenshtein_distance(gt_words, hyp_words) / float(len(gt_words)), 4)


def compute_entity_prf1(predicted: Set[str], gold: Set[str]) -> Dict[str, float]:
    """Computes entity-level Precision, Recall, and F1."""
    pred_norm = {canonicalize_skill(x).lower() for x in predicted if x}
    gold_norm = {canonicalize_skill(x).lower() for x in gold if x}
    if not pred_norm and not gold_norm:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_norm & gold_norm)
    precision = tp / float(len(pred_norm)) if pred_norm else 0.0
    recall = tp / float(len(gold_norm)) if gold_norm else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_precision_at_k(
    ranked_filenames: List[str],
    human_grades: Dict[str, int],
    k: int = 5,
    relevant_threshold: int = 2,
) -> float:
    """Precision@K where human relevance grade >= relevant_threshold (on 0-3 scale) counts as relevant."""
    top_k = ranked_filenames[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for fn in top_k if human_grades.get(fn, 0) >= relevant_threshold)
    return round(hits / float(len(top_k)), 4)


def compute_ndcg_at_k(
    ranked_filenames: List[str],
    human_grades: Dict[str, int],
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain (nDCG@K) per Jarvelin & Kekalainen (2002) for 0-3 relevance grades."""
    def dcg(rels: List[int]) -> float:
        total = 0.0
        for idx, rel in enumerate(rels, start=1):
            total += (2**rel - 1) / math.log2(idx + 1)
        return total

    actual_rels = [int(human_grades.get(fn, 0)) for fn in ranked_filenames[:k]]
    ideal_rels = sorted([int(v) for v in human_grades.values()], reverse=True)[:k]
    ideal_dcg = dcg(ideal_rels)
    if ideal_dcg == 0.0:
        return 0.0
    return round(dcg(actual_rels) / ideal_dcg, 4)


DEMO_CORPUS = [
    {
        "filename": "cand01_swe_synonyms.txt",
        "text": (
            "Alex Wong\nalex.wong@example.com | +852 9123 4567\n"
            "B.Sc in Computer Science, Hong Kong Metropolitan University\n"
            "Senior Cloud Backend Developer at FinTech Labs: Built asynchronous RESTful APIs in Py and FastAPI, "
            "deployed containerized microservices via Docker and K8s on Amazon Web Services, managed Postgres databases, "
            "and built interactive UI views in TS and ReactJS with automated CICD."
        ),
        "gold_skills": ["Python", "FastAPI", "REST API", "Docker", "Kubernetes", "AWS", "PostgreSQL", "TypeScript", "React", "CI/CD"],
        "grades": {"JD_SWE": 3, "JD_CYBER": 0, "JD_DATA": 1},
    },
    {
        "filename": "cand02_eloquent_bluff_prose.txt",
        "text": (
            "Jordan Lee\njordan.lee@example.com | +852 9234 5678\n"
            "Master of Business Administration\n"
            "Visionary digital transformation strategist passionate about scalable cloud-native software architecture, "
            "modern full-stack engineering excellence, enterprise cybersecurity resilience, and executive data storytelling. "
            "Collaborated with cross-functional software teams to align product roadmaps and agile delivery."
        ),
        "gold_skills": [],
        "grades": {"JD_SWE": 0, "JD_CYBER": 0, "JD_DATA": 0},
    },
    {
        "filename": "cand03_cyber_specialist.txt",
        "text": (
            "Samira Patel\nsamira.patel@example.com | +852 9345 6789\n"
            "B.Eng in Information Security\n"
            "SOC Security Analyst: Investigated security incidents in Splunk SIEM, inspected network packet captures in Wireshark, "
            "executed web pentesting aligned with OWASP Top 10, hardened Linux servers, and led Incident Response drills."
        ),
        "gold_skills": ["SIEM", "Splunk", "Wireshark", "Penetration Testing", "OWASP", "Linux", "Incident Response"],
        "grades": {"JD_SWE": 0, "JD_CYBER": 3, "JD_DATA": 0},
    },
    {
        "filename": "cand04_data_analyst.txt",
        "text": (
            "Chloe Chan\nchloe.chan@example.com | +852 9456 7890\n"
            "BSc in Data Science\n"
            "Business Intelligence Analyst: Engineered automated ETL pipelines using Python, Pandas, and SQL. "
            "Built executive KPI dashboards in PowerBI and Tableau, and designed A/B Testing & Statistical Modeling studies."
        ),
        "gold_skills": ["Python", "Pandas", "SQL", "ETL", "Power BI", "Tableau", "A/B Testing", "Statistical Modeling"],
        "grades": {"JD_SWE": 1, "JD_CYBER": 0, "JD_DATA": 3},
    },
    {
        "filename": "cand05_junior_swe.txt",
        "text": (
            "ivan.ho@example.com | +852 9567 8901\nIvan Ho\n"
            "Higher Diploma in Software Engineering\n"
            "Junior Software Developer Intern: Developed web frontends with JavaScript, React, HTML, CSS, "
            "and wrote Python backend scripts with Docker and Git."
        ),
        "gold_skills": ["JavaScript", "React", "HTML", "CSS", "Python", "Docker", "Git"],
        "grades": {"JD_SWE": 2, "JD_CYBER": 0, "JD_DATA": 1},
    },
]


def run_demo_evaluation() -> Dict[str, Any]:
    jds_path = Path(__file__).resolve().parent / "data" / "sample_jds.json"
    jds = json.loads(jds_path.read_text(encoding="utf-8"))

    # 1. NER Baseline vs. Upgraded Comparison
    baseline_f1s: List[float] = []
    upgraded_f1s: List[float] = []
    baseline_profiles = []
    upgraded_profiles = []

    for item in DEMO_CORPUS:
        p_base = extract_candidate_profile(item["text"], filename=item["filename"], mode="baseline")
        p_up = extract_candidate_profile(item["text"], filename=item["filename"], mode="transformer")
        baseline_profiles.append(p_base)
        upgraded_profiles.append(p_up)

        if item["gold_skills"]:
            baseline_f1s.append(compute_entity_prf1(set(p_base["skills"]), set(item["gold_skills"]))["f1"])
            upgraded_f1s.append(compute_entity_prf1(set(p_up["skills"]), set(item["gold_skills"]))["f1"])

    # 2. Ranking Method Comparison across all 3 IT JDs (Keyword vs. Pure SBERT vs. 70/30 Hybrid)
    methods = ["keyword", "sbert", "hybrid"]
    ranking_summary: Dict[str, Dict[str, float]] = {}

    for m in methods:
        p_at_3_list: List[float] = []
        ndcg_at_5_list: List[float] = []
        bluff_ranks: List[int] = []

        for jd in jds:
            jd_id = jd["jd_id"]
            grades = {c["filename"]: c["grades"][jd_id] for c in DEMO_CORPUS}
            ranked = rank_candidates(
                profiles=upgraded_profiles,
                jd_text=jd["description"],
                mandatory_skills=jd["mandatory_skills"],
                method=m,
            )
            ordered_fns = [r["filename"] for r in ranked]
            p_at_3_list.append(compute_precision_at_k(ordered_fns, grades, k=2, relevant_threshold=2))
            ndcg_at_5_list.append(compute_ndcg_at_k(ordered_fns, grades, k=5))
            bluff_rank = next(r["rank"] for r in ranked if r["filename"] == "cand02_eloquent_bluff_prose.txt")
            bluff_ranks.append(bluff_rank)

        ranking_summary[m] = {
            "mean_precision_at_2": round(sum(p_at_3_list) / len(p_at_3_list), 4),
            "mean_ndcg_at_5": round(sum(ndcg_at_5_list) / len(ndcg_at_5_list), 4),
            "eloquent_bluff_avg_rank_position": round(sum(bluff_ranks) / len(bluff_ranks), 2),
        }

    report = {
        "ner_skill_f1_comparison": {
            "baseline_rule_mean_f1": round(sum(baseline_f1s) / len(baseline_f1s), 4),
            "upgraded_trf_gazetteer_mean_f1": round(sum(upgraded_f1s) / len(upgraded_f1s), 4),
        },
        "ranking_benchmarks_across_3_it_jds": ranking_summary,
    }
    print(json.dumps(report, indent=2))
    return report


def _render_resume_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(56, 56, 539, 786), text, fontsize=11, fontname="helv")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _rasterize_first_page(pdf_bytes: bytes, dpi: int = 220) -> np.ndarray:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def generate_sample_resumes(out_dir: Path = SAMPLE_DIR, skew_deg: float = 4.2) -> List[Dict[str, Any]]:
    """Writes each demo resume as a digital PDF, a degraded smartphone-style JPG, and a ground-truth TXT."""
    for sub in ("digital", "scanned", "ground_truth"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    samples = []
    for idx, item in enumerate(DEMO_CORPUS):
        stem = Path(item["filename"]).stem
        pdf_bytes = _render_resume_pdf(item["text"])
        degraded = degrade_image_smartphone_simulation(_rasterize_first_page(pdf_bytes), skew_deg=skew_deg, seed=42 + idx)

        pdf_path = out_dir / "digital" / f"{stem}.pdf"
        jpg_path = out_dir / "scanned" / f"{stem}_scan.jpg"
        pdf_path.write_bytes(pdf_bytes)
        cv2.imwrite(str(jpg_path), degraded)
        (out_dir / "ground_truth" / f"{stem}.txt").write_text(item["text"], encoding="utf-8")
        samples.append({"stem": stem, "pdf": pdf_path, "jpg": jpg_path, "ground_truth": item["text"]})
    return samples


def run_ocr_evaluation(skew_deg: float = 4.2) -> Dict[str, Any]:
    """Hypothesis 3: CER/WER on digital PDFs vs. degraded scans, with and without OpenCV preprocessing."""
    samples = generate_sample_resumes(skew_deg=skew_deg)
    conditions = {
        "digital_pdf_pymupdf": lambda s: ingest_document(s["pdf"].read_bytes(), s["pdf"].name),
        "scan_raw_tesseract": lambda s: ingest_document(s["jpg"].read_bytes(), s["jpg"].name, use_opencv=False),
        "scan_opencv_tesseract": lambda s: ingest_document(s["jpg"].read_bytes(), s["jpg"].name, use_opencv=True),
    }
    summary: Dict[str, Dict[str, float]] = {}
    for name, run in conditions.items():
        cers, wers, lats = [], [], []
        for s in samples:
            out = run(s)
            cers.append(compute_cer(s["ground_truth"], out["clean_text"]))
            wers.append(compute_wer(s["ground_truth"], out["clean_text"]))
            lats.append(out["tier1_latency_sec"])
        summary[name] = {
            "mean_cer": round(sum(cers) / len(cers), 4),
            "mean_wer": round(sum(wers) / len(wers), 4),
            "mean_latency_sec": round(sum(lats) / len(lats), 3),
        }
    report = {
        "ocr_accuracy_digital_vs_scanned": summary,
        "simulated_skew_deg": skew_deg,
        "sample_files_written_to": str(SAMPLE_DIR),
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COMP 4330SEF Benchmark Evaluator (CER/WER, NER F1, P@K, nDCG@K)")
    parser.add_argument("--run-demo", action="store_true", help="Run NER F1 and ranking (P@K, nDCG@K) benchmarks")
    parser.add_argument("--run-ocr-demo", action="store_true", help="Run OCR CER/WER benchmark on generated scans")
    parser.add_argument("--make-samples", action="store_true", help="Only write sample PDFs/scans to data/sample_resumes")
    args = parser.parse_args()
    if args.make_samples:
        generate_sample_resumes()
        print(f"Sample resumes written to: {SAMPLE_DIR}")
    if args.run_ocr_demo:
        run_ocr_evaluation()
    if args.run_demo:
        run_demo_evaluation()
    if not (args.run_demo or args.run_ocr_demo or args.make_samples):
        print("Pass --run-demo, --run-ocr-demo, or --make-samples.")
