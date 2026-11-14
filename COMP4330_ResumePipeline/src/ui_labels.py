"""
Plain-English labels and wording for the recruiter dashboard.

The pipeline itself works in technical terms (spaCy NER, SBERT, TF-IDF, OpenCV).
This module keeps every technical term in one place so the UI can speak plain
language to a recruiter while the underlying algorithm stays untouched.

Each label carries an English and a Traditional Chinese version. Use
`label_for(...)` to fetch the right one for the active language.

`src/ui_labels.py` contains no logic that affects ranking — it only maps
internal values to human-friendly text.
"""

from typing import Any, Dict, List

# --------------------------------------------------------------------------
# How resumes are read
# --------------------------------------------------------------------------

READ_METHOD_LABELS: Dict[str, Dict[str, str]] = {
    "PyMuPDF spatial layout (digital)": {"en": "Read straight from the PDF", "zh": "直接讀取 PDF 文字"},
    "python-docx (digital)": {"en": "Read straight from the Word file", "zh": "直接讀取 Word 檔案"},
    "plain-text (digital)": {"en": "Read straight from the text file", "zh": "直接讀取純文字檔"},
    "OpenCV + Tesseract OCR (scanned PDF)": {
        "en": "Read as a photo (scanned PDF)",
        "zh": "以照片方式辨讀（掃描 PDF）",
    },
    "Raw Tesseract OCR (scanned PDF)": {
        "en": "Read as a photo, no clean-up (scanned PDF)",
        "zh": "以照片方式辨讀，未做影像清理（掃描 PDF）",
    },
    "OpenCV + Tesseract OCR (image)": {"en": "Read from the photo", "zh": "從照片辨讀"},
    "Raw Tesseract OCR (image)": {
        "en": "Read from the photo, no clean-up",
        "zh": "從照片辨讀，未做影像清理",
    },
    "OpenCV + Tesseract OCR (forced)": {"en": "Read as a photo (forced)", "zh": "強制以照片方式辨讀"},
    "Raw Tesseract OCR (forced)": {
        "en": "Read as a photo, no clean-up (forced)",
        "zh": "強制以照片方式辨讀，未做影像清理",
    },
}

READ_METHOD_FALLBACK = {"en": "Read automatically", "zh": "自動辨讀"}


def label_for(entry: Dict[str, str], lang: str = "en") -> str:
    """Picks the right language out of a bilingual label dict."""
    return entry.get(lang) or entry.get("en") or ""


def read_method_label(routed_mode: str, lang: str = "en") -> str:
    """Turns an internal ingestion route into a recruiter-friendly phrase."""
    if not routed_mode:
        return label_for(READ_METHOD_FALLBACK, lang)
    if routed_mode in READ_METHOD_LABELS:
        return label_for(READ_METHOD_LABELS[routed_mode], lang)
    lowered = routed_mode.lower()
    if "ocr" in lowered and "opencv" in lowered:
        return {"en": "Read from a photo, with clean-up", "zh": "從照片辨讀，並已清理影像"}[lang]
    if "ocr" in lowered:
        return {"en": "Read from a photo, no clean-up", "zh": "從照片辨讀，未清理影像"}[lang]
    if "docx" in lowered:
        return {"en": "Read straight from the Word file", "zh": "直接讀取 Word 檔案"}[lang]
    if "text" in lowered:
        return {"en": "Read straight from the text file", "zh": "直接讀取純文字檔"}[lang]
    return label_for(READ_METHOD_FALLBACK, lang)


# --------------------------------------------------------------------------
# How candidates are scored
# --------------------------------------------------------------------------

SCORING_METHOD_OPTIONS = {
    "hybrid": {"en": "Balanced (recommended)", "zh": "平衡（建議）"},
    "sbert": {"en": "Overall fit only", "zh": "只看整體適合度"},
    "keyword": {"en": "Word matching only", "zh": "只看字詞比對"},
}

SCORING_METHOD_HELP = {
    "hybrid": {
        "en": (
            "Balanced mixes two things: how well the CV reads as a whole fit for the job, "
            "and how many of the must-have skills are actually there."
        ),
        "zh": (
            "平衡模式結合兩項因素：履歷整體而言與職位的適合程度，"
            "以及實際具備的必備技能數量。"
        ),
    },
    "sbert": {
        "en": (
            "Overall fit only compares the meaning of the CV with the job description. "
            "It ignores whether the exact skills are listed."
        ),
        "zh": (
            "只看整體適合度，會比較履歷與職位描述的語意相似程度，"
            "但不會考慮是否具備指定的個別技能。"
        ),
    },
    "keyword": {
        "en": (
            "Word matching only counts how many job words appear in the CV. "
            "This is how older applicant-tracking systems worked."
        ),
        "zh": (
            "只看字詞比對，僅計算職位描述中的字詞在履歷出現的次數，"
            "這是早期應徵者追蹤系統（ATS）運作的方式。"
        ),
    },
}

READING_ENGINE_OPTIONS = {
    "transformer": {"en": "Smarter reading (recommended)", "zh": "智慧型辨讀（建議）"},
    "baseline": {"en": "Simple reading (faster)", "zh": "簡易型辨讀（較快）"},
}

READING_ENGINE_HELP = {
    "transformer": {
        "en": (
            "Smarter reading uses the large language model plus a list of known IT terms, "
            "so it catches skills written as synonyms (for example 'k8s' or 'golang')."
        ),
        "zh": (
            "智慧型辨讀結合大型語言模型與已知 IT 技術名詞表，"
            "因此能辨識以同義詞書寫的技能（例如「k8s」或「golang」）。"
        ),
    },
    "baseline": {
        "en": (
            "Simple reading only matches exact words. It is faster, but it misses synonyms "
            "and unusual formats."
        ),
        "zh": (
            "簡易型辨讀只比對完全相同的字詞，速度較快，"
            "但會遺漏同義詞與特殊排版的內容。"
        ),
    },
}

# --------------------------------------------------------------------------
# Match strength
# --------------------------------------------------------------------------

_STRENGTH_LABELS = {
    "excellent": {"en": "Excellent match", "zh": "非常符合"},
    "strong": {"en": "Strong match", "zh": "高度符合"},
    "possible": {"en": "Possible match", "zh": "可能符合"},
    "weak": {"en": "Weak match", "zh": "符合度偏低"},
    "none": {"en": "Not a match", "zh": "不符合"},
}


def match_strength(score_pct: float, lang: str = "en") -> str:
    """A plain-language verdict for a match percentage."""
    if score_pct >= 85:
        key = "excellent"
    elif score_pct >= 70:
        key = "strong"
    elif score_pct >= 55:
        key = "possible"
    elif score_pct >= 40:
        key = "weak"
    else:
        key = "none"
    return label_for(_STRENGTH_LABELS[key], lang)


def match_strength_color(score_pct: float) -> str:
    """Colour token used by the UI for the verdict."""
    if score_pct >= 70:
        return "green"
    if score_pct >= 55:
        return "orange"
    return "red"


# --------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------


def plain_explanation(row: Dict[str, Any], skill_weight: float, lang: str = "en") -> str:
    """
    Builds a recruiter-friendly explanation for one candidate.

    The technical formula is kept in `src/tier3_matching.py`; this only
    re-describes the same numbers in everyday language.
    """
    total = float(row.get("total_score_pct", 0.0))
    semantic = float(row.get("semantic_score_pct", 0.0))
    skill = float(row.get("skill_score_pct", 0.0))

    matched: List[str] = list(row.get("matched_skills") or [])
    missing: List[str] = list(row.get("missing_skills") or [])
    required_total = len(matched) + len(missing)

    strength = match_strength(total, lang)

    if lang == "zh":
        parts = [f"**{strength}** — 整體匹配度 {total:.0f}%。"]
        if required_total:
            parts.append(
                f"已具備 {len(matched)}／{required_total} 項必備技能"
                f"（技能清單完成度 {skill:.0f}%）。"
            )
        else:
            parts.append("此職位尚未設定必備技能清單。")
        parts.append(f"整體而言，此份履歷與職位描述的語意相似度為 {semantic:.0f}%。")
        if missing:
            parts.append("缺少：" + "、".join(missing) + "。")
        if matched:
            parts.append("已具備：" + "、".join(matched) + "。")
        return " ".join(parts)

    parts = [f"**{strength}** — overall score {total:.0f}%."]
    if required_total:
        parts.append(
            f"They have {len(matched)} of the {required_total} must-have skills "
            f"({skill:.0f}% of the skill checklist)."
        )
    else:
        parts.append("No must-have skill checklist was set for this job.")
    parts.append(
        f"Taken as a whole, the CV reads as a {semantic:.0f}% match to the job description."
    )
    if missing:
        parts.append("Missing: " + ", ".join(missing) + ".")
    if matched:
        parts.append("Present: " + ", ".join(matched) + ".")
    return " ".join(parts)


def plain_metric_hint(method: str, lang: str = "en") -> str:
    """Short caption under the 'Overall match' metric, per scoring method."""
    hints = {
        "keyword": {
            "en": "Based on how many job words appear in the CV",
            "zh": "依據職位描述的字詞在履歷中出現的比例計算",
        },
        "sbert": {
            "en": "Based on the overall meaning of the CV vs. the job description",
            "zh": "依據履歷與職位描述的整體語意相似程度計算",
        },
        "hybrid": {
            "en": "Overall meaning of the CV, plus must-have skills",
            "zh": "結合履歷整體語意與必備技能覆蓋率計算",
        },
    }
    entry = hints.get(method, hints["hybrid"])
    return label_for(entry, lang)


def skills_or_none(items: List[str], lang: str = "en") -> str:
    """Joins a list of skills for display, or a friendly fallback."""
    if not items:
        return "None" if lang == "en" else "無"
    return ", ".join(items)
