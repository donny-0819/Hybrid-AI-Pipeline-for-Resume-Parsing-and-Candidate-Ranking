import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.language import Language

from src.config import (
    IT_SKILLS_GAZETTEER,
    SKILL_SYNONYMS,
    canonicalize_skill,
    load_config,
)

_NLP_CACHE: Dict[str, Tuple[Language, str]] = {}

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+?[0-9]{1,3}[ .-]?)?(?:\(?[0-9]{2,4}\)?[ .-]?)?[0-9]{3,4}[ .-][0-9]{4}")

# Skill terms that are also everyday English words ("go", "the rest", "excel at") or short
# acronyms; these only count as skills when written with their technical capitalisation.
CASE_SENSITIVE_TERMS: Dict[str, str] = {
    "go": "Go",
    "rest": "REST",
    "js": "JS",
    "ts": "TS",
    "py": "Py",
    "ids": "IDS",
    "ips": "IPS",
    "iam": "IAM",
    "excel": "Excel",
    "kube": "Kube",
    "etl": "ETL",
    "nist": "NIST",
}


def _term_pattern(term: str) -> Tuple[str, int]:
    """Returns (regex, flags) for a skill term, honouring CASE_SENSITIVE_TERMS."""
    low = term.lower()
    if low in CASE_SENSITIVE_TERMS:
        return r"(?<![A-Za-z0-9])" + re.escape(CASE_SENSITIVE_TERMS[low]) + r"(?![A-Za-z0-9])", 0
    return r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE
DEGREE_KEYWORDS = [
    "Bachelor",
    "B.Sc",
    "BSc",
    "B.Eng",
    "BEng",
    "B.A",
    "Master",
    "M.Sc",
    "MSc",
    "M.Eng",
    "MBA",
    "Ph.D",
    "PhD",
    "Doctorate",
    "Associate Degree",
    "Higher Diploma",
    "Computer Science",
    "Software Engineering",
    "Information Security",
    "Data Science",
    "Information Technology",
]
EXPERIENCE_MARKERS = [
    "Engineer",
    "Developer",
    "Analyst",
    "Intern",
    "Architect",
    "Consultant",
    "Administrator",
    "Specialist",
    "Manager",
    "Researcher",
]


def _build_entity_ruler_patterns() -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    terms = [(skill, skill) for skill in IT_SKILLS_GAZETTEER] + list(SKILL_SYNONYMS.items())
    for term, canonical in terms:
        low = term.lower()
        if low in CASE_SENSITIVE_TERMS:
            token_pattern = [{"ORTH": CASE_SENSITIVE_TERMS[low]}]
        else:
            token_pattern = [{"LOWER": t.lower()} for t in term.split()]
        patterns.append({"label": "SKILL", "pattern": token_pattern, "id": canonical})
    return patterns


def get_nlp_pipeline(mode: str = "transformer") -> tuple:
    """
    Loads and caches the requested spaCy NER pipeline.
    mode='baseline' -> en_core_web_sm without synonym EntityRuler (replicates pyresparser rule/small-model baseline).
    mode='transformer' -> en_core_web_trf (falls back gracefully to en_core_web_sm if trf weights are not installed) + custom EntityRuler.
    """
    if mode in _NLP_CACHE:
        return _NLP_CACHE[mode]

    cfg = load_config()["ner"]
    loaded_model_name = ""

    if mode == "baseline":
        target = cfg["baseline_model"]
        try:
            nlp = spacy.load(target)
            loaded_model_name = target
        except OSError:
            nlp = spacy.blank("en")
            loaded_model_name = "spacy_blank_en (fallback)"
    else:
        target = cfg["upgraded_model"]
        try:
            nlp = spacy.load(target)
            loaded_model_name = target
        except OSError:
            try:
                fallback = cfg["baseline_model"]
                nlp = spacy.load(fallback)
                loaded_model_name = f"{fallback} (trf_not_installed_fallback)"
            except OSError:
                nlp = spacy.blank("en")
                loaded_model_name = "spacy_blank_en (fallback)"

        if "entity_ruler" not in nlp.pipe_names:
            before_pipe = "ner" if "ner" in nlp.pipe_names else None
            ruler = nlp.add_pipe("entity_ruler", before=before_pipe, config={"overwrite_ents": False})
            ruler.add_patterns(_build_entity_ruler_patterns())

    _NLP_CACHE[mode] = (nlp, loaded_model_name)
    return nlp, loaded_model_name


def _extract_contact(text: str) -> Dict[str, Optional[str]]:
    emails = EMAIL_PATTERN.findall(text)
    phones = PHONE_PATTERN.findall(text)
    return {
        "email": emails[0] if emails else None,
        "phone": phones[0].strip() if phones else None,
    }


def _extract_candidate_name(doc: Any, text: str) -> str:
    # Check top lines first for PERSON entities
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_chunk = "\n".join(lines[:6])
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text.strip() in header_chunk:
             candidate = " ".join(ent.text.split())
             if 2 <= len(candidate) <= 50 and "@" not in candidate:
                return candidate
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate = " ".join(ent.text.split())
            if 2 <= len(candidate) <= 50 and "@" not in candidate:
                return candidate
    if lines:
        first_line = lines[0]
        if len(first_line.split()) <= 5 and "@" not in first_line:
            return first_line
    return "Unknown Candidate"


def _extract_education(doc: Any, text: str) -> List[str]:
    edu_items: List[str] = []
    for line in text.splitlines():
        clean_ln = " ".join(line.split()).strip()
        if not clean_ln:
            continue
        if any(re.search(r"\b" + re.escape(kw) + r"\b", clean_ln, flags=re.IGNORECASE) for kw in DEGREE_KEYWORDS):
            if clean_ln not in edu_items:
                edu_items.append(clean_ln)
    for ent in doc.ents:
        if ent.label_ == "ORG" and any(w in ent.text.lower() for w in ("university", "college", "institute", "polytechnic")):
            org_clean = " ".join(ent.text.split())
            if org_clean not in edu_items:
                edu_items.append(org_clean)
    return edu_items[:8]


def _extract_experience(doc: Any, text: str) -> List[str]:
    exp_items: List[str] = []
    for line in text.splitlines():
        clean_ln = " ".join(line.split()).strip()
        if 6 <= len(clean_ln) <= 140:
            if any(re.search(r"\b" + re.escape(m) + r"\b", clean_ln, flags=re.IGNORECASE) for m in EXPERIENCE_MARKERS):
                if clean_ln not in exp_items:
                    exp_items.append(clean_ln)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_clean = " ".join(ent.text.split())
            if len(org_clean) > 2 and org_clean not in exp_items and "university" not in org_clean.lower():
                exp_items.append(f"Organization: {org_clean}")
    return exp_items[:10]


def _extract_skills_baseline(text: str) -> List[str]:
    """Static exact-dictionary lookup without synonym expansion (mirrors pyresparser baseline limitation)."""
    found: List[str] = []
    for skill in IT_SKILLS_GAZETTEER:
        pat, flags = _term_pattern(skill)
        if re.search(pat, text, flags=flags):
            if skill not in found:
                found.append(skill)
    return sorted(found)


def _extract_skills_upgraded(doc: Any, text: str) -> List[str]:
    """Combines spaCy EntityRuler SKILL spans with regex boundary matching for symbols like C++, C#, CI/CD, and synonyms."""
    found_set = set()
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            canonical = ent.ent_id_ if ent.ent_id_ else canonicalize_skill(ent.text)
            found_set.add(canonical)

    # Ensure punctuation-heavy tokens (e.g., C++, C#, Next.js, Node.js, K8s) are reliably captured
    all_terms = [(s, s) for s in IT_SKILLS_GAZETTEER] + [(k, v) for k, v in SKILL_SYNONYMS.items()]
    for raw_term, canonical in all_terms:
        pat, flags = _term_pattern(raw_term)
        if re.search(pat, text, flags=flags):
            found_set.add(canonical)

    return sorted(found_set)


def extract_candidate_profile(
    clean_text: str,
    filename: str = "resume",
    mode: str = "transformer",
) -> Dict[str, Any]:
    """
    Tier 2 Entry Point: Runs either 'baseline' or 'transformer' extraction track
    and returns the standardized Candidate Profile JSON schema.
    """
    t0 = time.perf_counter()
    nlp, model_used = get_nlp_pipeline(mode=mode)
    doc = nlp(clean_text[:25000])

    contact = _extract_contact(clean_text)
    name = _extract_candidate_name(doc, clean_text)
    education = _extract_education(doc, clean_text)
    experience = _extract_experience(doc, clean_text)
    if mode == "baseline":
        skills = _extract_skills_baseline(clean_text)
    else:
        skills = _extract_skills_upgraded(doc, clean_text)

    latency = round(time.perf_counter() - t0, 4)
    return {
        "filename": filename,
        "ner_mode": mode,
        "model_used": model_used,
        "name": name,
        "contact": contact,
        "education": education,
        "experience": experience,
        "skills": skills,
        "raw_text": clean_text,
        "tier2_latency_sec": latency,
    }


def save_profile_to_sqlite(profile: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Persists structured candidate JSON schema into a local SQLite store (Figure 4.1)."""
    cfg = load_config()
    target_db = Path(db_path or cfg["ner"]["sqlite_db_path"])
    target_db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_db))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                ner_mode TEXT,
                candidate_name TEXT,
                email TEXT,
                phone TEXT,
                skills_json TEXT,
                profile_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO candidate_profiles
            (filename, ner_mode, candidate_name, email, phone, skills_json, profile_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.get("filename"),
                profile.get("ner_mode"),
                profile.get("name"),
                profile.get("contact", {}).get("email"),
                profile.get("contact", {}).get("phone"),
                json.dumps(profile.get("skills", [])),
                json.dumps(profile),
            ),
        )
        conn.commit()
    finally:
        conn.close()
