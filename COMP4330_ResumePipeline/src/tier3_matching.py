import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import canonicalize_skill, load_config
from src.tier2_extraction import extract_candidate_profile

_SBERT_CACHE: Dict[str, Any] = {}


def get_sbert_model(model_name: Optional[str] = None) -> Any:
    cfg = load_config()["matching"]
    target = model_name or cfg["sbert_model"]
    if target not in _SBERT_CACHE:
        from sentence_transformers import SentenceTransformer

        _SBERT_CACHE[target] = SentenceTransformer(target)
    return _SBERT_CACHE[target]


def extract_mandatory_skills_from_jd(jd_text: str, explicit_skills: Optional[List[str]] = None) -> List[str]:
    """Derives canonical mandatory skills from explicit recruiter tags or automatically from the JD text."""
    if explicit_skills:
        cleaned = []
        for s in explicit_skills:
            if s.strip():
                c = canonicalize_skill(s)
                if c not in cleaned:
                    cleaned.append(c)
        if cleaned:
            return cleaned
    jd_extracted = extract_candidate_profile(jd_text, filename="jd", mode="transformer")
    return jd_extracted.get("skills", [])


def compute_skill_gap(candidate_skills: List[str], mandatory_skills: List[str]) -> Dict[str, Any]:
    cand_canon = {canonicalize_skill(s).lower(): canonicalize_skill(s) for s in candidate_skills}
    req_canon = []
    for r in mandatory_skills:
        c = canonicalize_skill(r)
        if c not in req_canon:
            req_canon.append(c)

    if not req_canon:
        return {
            "skill_score": 1.0,
            "matched_skills": sorted(cand_canon.values()),
            "missing_skills": [],
        }

    matched = [r for r in req_canon if r.lower() in cand_canon]
    missing = [r for r in req_canon if r.lower() not in cand_canon]
    coverage = len(matched) / float(len(req_canon))
    return {
        "skill_score": round(coverage, 4),
        "matched_skills": matched,
        "missing_skills": missing,
    }


def _build_candidate_semantic_summary(profile: Dict[str, Any]) -> str:
    """Constructs structured representation combining skills, experience, education, and full text for SBERT."""
    parts: List[str] = []
    if profile.get("skills"):
        parts.append("Technical Skills: " + ", ".join(profile["skills"]))
    if profile.get("experience"):
        parts.append("Experience: " + " ; ".join(profile["experience"]))
    if profile.get("education"):
        parts.append("Education: " + " ; ".join(profile["education"]))
    if profile.get("raw_text"):
        parts.append(profile["raw_text"][:4000])
    return "\n".join(parts).strip()


def rank_candidates(
    profiles: List[Dict[str, Any]],
    jd_text: str,
    mandatory_skills: Optional[List[str]] = None,
    method: str = "hybrid",
    w_semantic: Optional[float] = None,
    w_skill: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Tier 3 Entry Point: Ranks candidate profiles against a Job Description.
    Supported methods:
      - 'keyword': Traditional Enterprise ATS TF-IDF lexical overlap baseline.
      - 'sbert': Pure Sentence-BERT (all-MiniLM-L6-v2) cosine similarity.
      - 'hybrid': Proposed Dual-Track Scorer (S_total = w_semantic * S_semantic + w_skill * S_skill).
    """
    if not profiles:
        return []

    t0 = time.perf_counter()
    cfg = load_config()["matching"]
    alpha = cfg["semantic_weight"] if w_semantic is None else float(w_semantic)
    beta = cfg["skill_weight"] if w_skill is None else float(w_skill)

    req_skills = extract_mandatory_skills_from_jd(jd_text, explicit_skills=mandatory_skills)
    cand_texts = [_build_candidate_semantic_summary(p) for p in profiles]

    # Compute similarity scores according to chosen method
    if method == "keyword":
        raw_texts = [p.get("raw_text", "") for p in profiles]
        vec = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b[A-Za-z0-9+#.-]{2,}\b")
        tfidf_matrix = vec.fit_transform([jd_text] + raw_texts)
        sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        semantic_scores = [float(np.clip(s, 0.0, 1.0)) for s in sims]
    else:
        model = get_sbert_model()
        embeddings = model.encode([jd_text] + cand_texts, convert_to_numpy=True, normalize_embeddings=True)
        jd_vec = embeddings[0]
        cand_vecs = embeddings[1:]
        sims = np.dot(cand_vecs, jd_vec)
        semantic_scores = [float(np.clip(s, 0.0, 1.0)) for s in sims]

    batch_latency = time.perf_counter() - t0
    per_doc_tier3_latency = batch_latency / float(len(profiles))

    ranked_results: List[Dict[str, Any]] = []
    for idx, prof in enumerate(profiles):
        sem_score = semantic_scores[idx]
        gap = compute_skill_gap(prof.get("skills", []), req_skills)
        sk_score = gap["skill_score"]

        if method == "keyword":
            total_score = sem_score
        elif method == "sbert":
            total_score = sem_score
        else:
            total_score = (alpha * sem_score) + (beta * sk_score)

        total_latency = (
            float(prof.get("tier1_latency_sec", 0.0))
            + float(prof.get("tier2_latency_sec", 0.0))
            + per_doc_tier3_latency
        )

        coverage_note = f"Matched {len(gap['matched_skills'])}/{len(req_skills)} required skills."
        if method == "keyword":
            explanation = f"Keyword (TF-IDF) Score {total_score * 100:.1f}%. {coverage_note}"
        elif method == "sbert":
            explanation = f"SBERT Cosine Score {total_score * 100:.1f}%. {coverage_note}"
        else:
            explanation = (
                f"Hybrid Score {total_score * 100:.1f}% = "
                f"{alpha:.2f} * Semantic ({sem_score * 100:.1f}%) + "
                f"{beta:.2f} * Hard-Skill Coverage ({sk_score * 100:.1f}%). "
                f"{coverage_note}"
            )

        ranked_results.append({
            "filename": prof.get("filename", f"candidate_{idx}"),
            "candidate_name": prof.get("name", "Unknown Candidate"),
            "email": prof.get("contact", {}).get("email"),
            "phone": prof.get("contact", {}).get("phone"),
            "ranking_method": method,
            "total_score": round(total_score, 4),
            "total_score_pct": round(total_score * 100.0, 2),
            "semantic_score": round(sem_score, 4),
            "semantic_score_pct": round(sem_score * 100.0, 2),
            "skill_score": round(sk_score, 4),
            "skill_score_pct": round(sk_score * 100.0, 2),
            "matched_skills": gap["matched_skills"],
            "missing_skills": gap["missing_skills"],
            "all_extracted_skills": prof.get("skills", []),
            "education": prof.get("education", []),
            "experience": prof.get("experience", []),
            "routed_mode": prof.get("routed_mode", "digital"),
            "ner_mode": prof.get("ner_mode", "transformer"),
            "end_to_end_latency_sec": round(total_latency, 4),
            "explanation": explanation,
        })

    ranked_results.sort(key=lambda r: r["total_score"], reverse=True)
    for rank_idx, item in enumerate(ranked_results, start=1):
        item["rank"] = rank_idx
    return ranked_results
