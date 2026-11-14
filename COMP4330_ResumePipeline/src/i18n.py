"""
Bilingual (English / Traditional Chinese) strings for the recruiter dashboard.

Every user-visible phrase lives here so the whole interface can be switched
language at runtime without touching the pipeline code.

Each entry is a dict with an "en" key and a "zh" key. `t()` looks up a key and
falls back to English if a translation is missing.
"""

from typing import Any, Dict

LANGS = {
    "en": "English",
    "zh": "繁體中文",
}

DEFAULT_LANG = "en"

_STRINGS: Dict[str, Dict[str, str]] = {
    # ---- App shell ----
    "app_title": {
        "en": "Resume screening",
        "zh": "履歷篩選系統",
    },
    "app_subtitle": {
        "en": "Upload resumes, pick a job, and get a ranked shortlist. Everything runs locally on this machine.",
        "zh": "上載履歷、選擇職位，即可取得排名名單。所有運算均在本機進行。",
    },
    "footer_note": {
        "en": "COMP 4330SEF · P07 Group 3 · Runs entirely offline",
        "zh": "COMP 4330SEF · P07 第 3 組 · 完全離線運作",
    },
    # ---- Step 1: job ----
    "step1_title": {
        "en": "1. Choose the job",
        "zh": "一、選擇職位",
    },
    "step1_caption": {
        "en": "Pick a role, or write your own description.",
        "zh": "選擇預設職位，或自行填寫職位描述。",
    },
    "job_select_label": {
        "en": "Job to screen against",
        "zh": "要篩選的職位",
    },
    "job_description_label": {
        "en": "Job description",
        "zh": "職位描述",
    },
    "job_description_placeholder": {
        "en": "Paste the job description here...",
        "zh": "在此貼上職位描述……",
    },
    "must_have_label": {
        "en": "Must-have skills",
        "zh": "必備技能",
    },
    "must_have_placeholder": {
        "en": "Python, SQL, Docker (leave blank to detect automatically)",
        "zh": "Python、SQL、Docker（留空則自動偵測）",
    },
    "must_have_help": {
        "en": "Comma-separated. Leave blank and the tool will work it out from the description.",
        "zh": "以逗號分隔。留空時，系統會從職位描述自動判斷。",
    },
    # ---- Step 2: uploads ----
    "step2_title": {
        "en": "2. Add resumes",
        "zh": "二、加入履歷",
    },
    "step2_caption": {
        "en": "PDF, Word, plain text, or a photo of a printed CV.",
        "zh": "支援 PDF、Word、純文字，或印刷履歷的照片。",
    },
    "upload_label": {
        "en": "Upload resumes",
        "zh": "上載履歷",
    },
    "files_ready": {
        "en": "{n} file(s) ready: {names}",
        "zh": "已準備 {n} 個檔案：{names}",
    },
    "run_button": {
        "en": "Screen these resumes",
        "zh": "開始篩選履歷",
    },
    # ---- Errors ----
    "error_no_jd": {
        "en": "Please add a job description first.",
        "zh": "請先填寫職位描述。",
    },
    "error_no_files": {
        "en": "Please add at least one resume.",
        "zh": "請至少加入一份履歷。",
    },
    # ---- Progress ----
    "progress_reading": {
        "en": "Reading resumes...",
        "zh": "正在讀取履歷……",
    },
    "progress_item": {
        "en": "Read {i} of {n}: {name}",
        "zh": "已讀取第 {i}／{n} 份：{name}",
    },
    # ---- Results ----
    "results_header": {
        "en": "Results",
        "zh": "篩選結果",
    },
    "metric_screened": {
        "en": "Candidates screened",
        "zh": "已篩選人數",
    },
    "metric_best_match": {
        "en": "Best overall match",
        "zh": "最佳整體匹配",
    },
    "metric_time": {
        "en": "Time per resume",
        "zh": "每份履歷耗時",
    },
    "metric_time_help": {
        "en": "How long the whole process took per file",
        "zh": "每個檔案處理的平均時間",
    },
    "shortlist_header": {
        "en": "Shortlist",
        "zh": "入圍名單",
    },
    "shortlist_caption": {
        "en": "Ranked from best match to weakest.",
        "zh": "按匹配度由高至低排列。",
    },
    "download_csv": {
        "en": "Download shortlist (CSV)",
        "zh": "下載名單（CSV）",
    },
    "detail_header": {
        "en": "Candidate detail",
        "zh": "應徵者詳情",
    },
    "detail_caption": {
        "en": "Open a candidate to see why they scored the way they did.",
        "zh": "展開應徵者即可查看評分原因。",
    },
    # ---- Table columns ----
    "col_rank": {
        "en": "Rank",
        "zh": "排名",
    },
    "col_candidate": {
        "en": "Candidate",
        "zh": "應徵者",
    },
    "col_total": {
        "en": "Overall match",
        "zh": "整體匹配",
    },
    "col_semantic": {
        "en": "Overall fit",
        "zh": "整體適合度",
    },
    "col_skill": {
        "en": "Must-have skills",
        "zh": "必備技能",
    },
    "col_matched": {
        "en": "Skills they have",
        "zh": "已具備技能",
    },
    "col_missing": {
        "en": "Skills missing",
        "zh": "缺少技能",
    },
    "col_read_method": {
        "en": "How it was read",
        "zh": "讀取方式",
    },
    "col_file": {
        "en": "File",
        "zh": "檔案",
    },
    # ---- Candidate detail ----
    "skills_have_title": {
        "en": "Must-have skills they have",
        "zh": "已具備的必備技能",
    },
    "skills_none_found": {
        "en": "None of the must-have skills were found.",
        "zh": "找不到任何必備技能。",
    },
    "all_skills_title": {
        "en": "All skills found in the CV",
        "zh": "履歷中偵測到的所有技能",
    },
    "none_found": {
        "en": "None found",
        "zh": "沒有偵測到",
    },
    "skills_missing_title": {
        "en": "Must-have skills missing",
        "zh": "缺少的必備技能",
    },
    "all_present": {
        "en": "They have every must-have skill.",
        "zh": "已具備全部必備技能。",
    },
    "contact_title": {
        "en": "Contact & education",
        "zh": "聯絡方式與學歷",
    },
    "not_found_label": {
        "en": "not found",
        "zh": "找不到",
    },
    "technical_title": {
        "en": "Technical details for this candidate",
        "zh": "此應徵者的技術細節",
    },
    # ---- Sidebar ----
    "sidebar_header": {
        "en": "Advanced settings",
        "zh": "進階設定",
    },
    "sidebar_caption": {
        "en": "These control how precisely resumes are read. The default settings work well for most hiring.",
        "zh": "這些設定控制履歷的辨讀精確度。預設設定已適合大部分招聘情境。",
    },
    "appearance_header": {
        "en": "Appearance",
        "zh": "外觀",
    },
    "language_label": {
        "en": "Language",
        "zh": "語言",
    },
    "color_label": {
        "en": "Colour scheme",
        "zh": "配色方案",
    },
    "reading_header": {
        "en": "Reading accuracy",
        "zh": "辨讀精確度",
    },
    "scored_header": {
        "en": "Scoring method",
        "zh": "評分方式",
    },
    "storage_header": {
        "en": "Data storage",
        "zh": "資料儲存",
    },
}


def t(key: str, lang: str = DEFAULT_LANG, **kwargs: Any) -> str:
    """
    Looks up a UI string by key and translates it.

    Falls back to English when a key or a translation is missing, so the app
    never shows a blank label.
    """
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry.get(DEFAULT_LANG) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text
    return text


def lang_label(code: str, lang: str = DEFAULT_LANG) -> str:
    """Name of a language, shown in the language currently in use."""
    names = {
        "en": {"en": "English", "zh": "English"},
        "zh": {"en": "繁體中文", "zh": "繁體中文"},
    }
    entry = names.get(code)
    if entry is None:
        return code
    return entry.get(lang) or entry["en"]
