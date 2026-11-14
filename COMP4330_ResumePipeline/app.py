"""
Recruiter-facing dashboard for the Hybrid AI Resume Parsing & Ranking pipeline.

Design goal: a recruiter should be able to run the tool without knowing what
spaCy, SBERT, TF-IDF or OpenCV mean, and without reading English if they
prefer Chinese.

The interface supports:
  - English / Traditional Chinese, switched at runtime from the sidebar
  - Several colour schemes, switched at runtime from the sidebar
  - A three-step flow: choose a job -> add resumes -> read the results

Everything technical still exists and is still exposed, but it now lives inside
an "Advanced settings" panel and is labelled in plain language.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.i18n import DEFAULT_LANG, LANGS, lang_label, t
from src.tier1_ingestion import ingest_document
from src.tier2_extraction import extract_candidate_profile, save_profile_to_sqlite
from src.tier3_matching import rank_candidates
from src.ui_labels import (
    READING_ENGINE_HELP,
    READING_ENGINE_OPTIONS,
    SCORING_METHOD_HELP,
    SCORING_METHOD_OPTIONS,
    label_for,
    match_strength,
    match_strength_color,
    plain_explanation,
    plain_metric_hint,
    read_method_label,
)
from src.ui_theme import DEFAULT_THEME, build_theme_css, theme_label, theme_options

st.set_page_config(
    page_title="Resume screening",
    page_icon=":material/assignment_ind:",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)

JDS_PATH = Path(__file__).resolve().parent / "data" / "sample_jds.json"
RESULTS_KEY = "last_results"


def load_sample_jds() -> List[dict]:
    if JDS_PATH.exists():
        return json.loads(JDS_PATH.read_text(encoding="utf-8"))
    return []


# ---------------------------------------------------------------------------
# Session state bootstrap (one clear place, per Streamlit guidance)
# ---------------------------------------------------------------------------

sample_jds = load_sample_jds()
jd_options = [jd["title"] for jd in sample_jds] + ["Custom job description"]

DEFAULTS: Dict[str, Any] = {
    "lang": DEFAULT_LANG,
    "theme": DEFAULT_THEME,
    "jd_preset": jd_options[0],
    "jd_text": sample_jds[0]["description"] if sample_jds else "",
    "jd_skills": ", ".join(sample_jds[0]["mandatory_skills"]) if sample_jds else "",
}
for _key, _value in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

LANG = st.session_state["lang"]
THEME_KEY = st.session_state["theme"]


def _apply_preset() -> None:
    """Loads a preset job into the description and skill boxes."""
    chosen = next((jd for jd in sample_jds if jd["title"] == st.session_state["jd_preset"]), None)
    if chosen is None:
        return
    st.session_state["jd_text"] = chosen["description"]
    st.session_state["jd_skills"] = ", ".join(chosen["mandatory_skills"])


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
# Streamlit's own theme is fixed in config.toml, so the runtime colour choice
# is applied as a scoped CSS override on top of it.

st.html(build_theme_css(THEME_KEY), unsafe_allow_javascript=False)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    with st.expander(t("appearance_header", LANG), expanded=False):
        st.selectbox(
            t("language_label", LANG),
            options=list(LANGS.keys()),
            format_func=lambda code: lang_label(code, LANG),
            key="lang",
        )
        st.selectbox(
            t("color_label", LANG),
            options=theme_options(LANG),
            format_func=lambda key: theme_label(key, LANG),
            key="theme",
        )

    st.header(t("sidebar_header", LANG))
    st.caption(t("sidebar_caption", LANG))

    with st.expander(t("reading_header", LANG), expanded=False):
        engine_key = st.selectbox(
            "How resumes are read" if LANG == "en" else "履歷辨讀方式",
            options=list(READING_ENGINE_OPTIONS.keys()),
            format_func=lambda k: label_for(READING_ENGINE_OPTIONS[k], LANG),
            key="ner_mode",
            help=label_for(READING_ENGINE_HELP["transformer"], LANG),
        )
        st.caption(label_for(READING_ENGINE_HELP[engine_key], LANG))

        st.markdown("**Scanned photos**" if LANG == "en" else "**掃描影像**")
        use_opencv = st.toggle(
            "Clean up photos before reading" if LANG == "en" else "辨讀前先清理影像",
            value=True,
            key="use_opencv",
            help=(
                "Straightens a tilted page, removes speckles and increases contrast "
                "so text is easier to recognise."
                if LANG == "en"
                else "將傾斜的頁面校正為水平、去除雜點並提高對比，令文字更容易辨識。"
            ),
        )
        force_ocr = st.toggle(
            "Always treat PDFs as photos" if LANG == "en" else "一律將 PDF 當作影像",
            value=False,
            key="force_ocr",
            help=(
                "Test setting: ignores the selectable text and reads the page as an image."
                if LANG == "en"
                else "測試用設定：忽略可選取的文字，一律將頁面當作影像辨讀。"
            ),
        )

    with st.expander(t("scored_header", LANG), expanded=False):
        method_key = st.selectbox(
            "How candidates are scored" if LANG == "en" else "應徵者評分方式",
            options=list(SCORING_METHOD_OPTIONS.keys()),
            format_func=lambda k: label_for(SCORING_METHOD_OPTIONS[k], LANG),
            key="ranking_method",
            help=label_for(SCORING_METHOD_HELP["hybrid"], LANG),
        )
        st.caption(label_for(SCORING_METHOD_HELP[method_key], LANG))

        w_sem = st.slider(
            "Weight on overall fit" if LANG == "en" else "整體適合度的權重",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            key="w_sem",
            help=(
                "How much the final score depends on overall fit (the rest comes from skills)."
                if LANG == "en"
                else "最終分數有多少比重來自整體適合度（其餘來自必備技能）。"
            ),
        )
        w_sk = round(1.0 - w_sem, 2)
        st.caption(
            f"Final score = {w_sem:.0%} overall fit + {w_sk:.0%} must-have skills. "
            "Technical note: semantic similarity + hard-skill coverage."
            if LANG == "en"
            else f"最終分數 = {w_sem:.0%} 整體適合度 + {w_sk:.0%} 必備技能。"
            "技術說明：語意相似度 + 硬性技能覆蓋率。"
        )

    with st.expander(t("storage_header", LANG), expanded=False):
        persist_sqlite = st.toggle(
            "Save parsed CVs to the local database" if LANG == "en" else "將解析結果儲存至本機資料庫",
            value=True,
            key="persist_sqlite",
            help=(
                "Keeps a copy in data/candidates.db on this machine."
                if LANG == "en"
                else "在本機 data/candidates.db 保留一份副本。"
            ),
        )

    st.space("small")
    st.caption(t("footer_note", LANG))

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title(t("app_title", LANG))
st.caption(t("app_subtitle", LANG))

# ---------------------------------------------------------------------------
# Step 1 & 2: inputs
# ---------------------------------------------------------------------------

step1, step2 = st.columns([1.05, 1.0], gap="medium")

with step1:
    with st.container(border=True, key="card_job"):
        st.subheader(t("step1_title", LANG))
        st.caption(t("step1_caption", LANG))

        st.selectbox(
            t("job_select_label", LANG),
            options=jd_options,
            key="jd_preset",
            on_change=_apply_preset,
        )

        st.text_area(
            t("job_description_label", LANG),
            key="jd_text",
            height=170,
            label_visibility="collapsed",
            placeholder=t("job_description_placeholder", LANG),
        )

        st.text_input(
            t("must_have_label", LANG),
            key="jd_skills",
            placeholder=t("must_have_placeholder", LANG),
            help=t("must_have_help", LANG),
        )

with step2:
    with st.container(border=True, key="card_upload"):
        st.subheader(t("step2_title", LANG))
        st.caption(t("step2_caption", LANG))

        uploaded_files = st.file_uploader(
            t("upload_label", LANG),
            type=["pdf", "docx", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.caption(
                t(
                    "files_ready",
                    LANG,
                    n=len(uploaded_files),
                    names=", ".join(f.name for f in uploaded_files),
                )
            )

        st.space("small")
        run_clicked = st.button(
            t("run_button", LANG),
            type="primary",
            width="stretch",
            icon=":material/rate_review:",
        )

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

if run_clicked:
    jd_text = st.session_state["jd_text"]
    skills_input = st.session_state["jd_skills"]
    source_files = [(f.name, f.read()) for f in (uploaded_files or [])]

    if not jd_text.strip():
        st.error(t("error_no_jd", LANG), icon=":material/error:")
    elif not source_files:
        st.error(t("error_no_files", LANG), icon=":material/error:")
    else:
        explicit_skills = [s.strip() for s in skills_input.split(",") if s.strip()]
        profiles: List[Dict[str, Any]] = []
        progress = st.progress(0.0, text=t("progress_reading", LANG))

        for i, (fname, raw_bytes) in enumerate(source_files):
            t1_out = ingest_document(
                file_bytes=raw_bytes,
                filename=fname,
                force_ocr=force_ocr,
                use_opencv=use_opencv,
            )
            t2_prof = extract_candidate_profile(
                clean_text=t1_out["clean_text"],
                filename=fname,
                mode=engine_key,
            )
            t2_prof["routed_mode"] = t1_out["routed_mode"]
            t2_prof["tier1_latency_sec"] = t1_out["tier1_latency_sec"]
            t2_prof["preprocessing_metadata"] = t1_out["preprocessing_metadata"]
            if persist_sqlite:
                save_profile_to_sqlite(t2_prof)
            profiles.append(t2_prof)
            progress.progress(
                (i + 1) / len(source_files),
                text=t("progress_item", LANG, i=i + 1, n=len(source_files), name=fname),
            )

        ranked = rank_candidates(
            profiles=profiles,
            jd_text=jd_text,
            mandatory_skills=explicit_skills,
            method=method_key,
            w_semantic=w_sem,
            w_skill=w_sk,
        )
        progress.empty()

        st.session_state[RESULTS_KEY] = {
            "ranked": ranked,
            "method": method_key,
            "w_sk": w_sk,
        }

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

results = st.session_state.get(RESULTS_KEY)

if results:
    ranked = results["ranked"]
    method = results["method"]
    w_sk_run = results["w_sk"]

    st.space("medium")
    st.header(t("results_header", LANG))

    top = ranked[0]
    avg_lat = sum(r["end_to_end_latency_sec"] for r in ranked) / len(ranked)

    with st.container(horizontal=True, gap="medium"):
        st.metric(
            t("metric_screened", LANG),
            len(ranked),
            border=True,
        )
        st.metric(
            t("metric_best_match", LANG),
            f"{top['total_score_pct']:.0f}%",
            help=plain_metric_hint(method, LANG),
            border=True,
        )
        st.metric(
            t("metric_time", LANG),
            f"{avg_lat:.2f} s",
            help=t("metric_time_help", LANG),
            border=True,
        )

    # --- Shortlist ---
    with st.container(border=True, key="card_shortlist"):
        st.subheader(t("shortlist_header", LANG))
        st.caption(t("shortlist_caption", LANG))

        df_rows = []
        for r in ranked:
            df_rows.append(
                {
                    t("col_rank", LANG): r["rank"],
                    t("col_candidate", LANG): r["candidate_name"],
                    t("col_total", LANG): r["total_score_pct"],
                    t("col_semantic", LANG): r["semantic_score_pct"],
                    t("col_skill", LANG): r["skill_score_pct"],
                    t("col_matched", LANG): ", ".join(r["matched_skills"]) or "None",
                    t("col_missing", LANG): ", ".join(r["missing_skills"]) or "None",
                    t("col_read_method", LANG): read_method_label(r["routed_mode"], LANG),
                    t("col_file", LANG): r["filename"],
                }
            )
        df = pd.DataFrame(df_rows)

        st.dataframe(
            df,
            hide_index=True,
            column_config={
                t("col_rank", LANG): st.column_config.NumberColumn(
                    t("col_rank", LANG), width="small"
                ),
                t("col_candidate", LANG): st.column_config.TextColumn(
                    t("col_candidate", LANG), width="medium"
                ),
                t("col_total", LANG): st.column_config.ProgressColumn(
                    t("col_total", LANG),
                    format="%.0f%%",
                    min_value=0,
                    max_value=100,
                    width="medium",
                ),
                t("col_semantic", LANG): st.column_config.NumberColumn(
                    t("col_semantic", LANG), format="%.0f%%"
                ),
                t("col_skill", LANG): st.column_config.NumberColumn(
                    t("col_skill", LANG), format="%.0f%%"
                ),
                t("col_matched", LANG): st.column_config.TextColumn(t("col_matched", LANG)),
                t("col_missing", LANG): st.column_config.TextColumn(t("col_missing", LANG)),
                t("col_read_method", LANG): st.column_config.TextColumn(
                    t("col_read_method", LANG), width="medium"
                ),
                t("col_file", LANG): st.column_config.TextColumn(
                    t("col_file", LANG), width="small"
                ),
            },
            alt="Candidates ranked by overall match",
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            t("download_csv", LANG),
            data=csv_bytes,
            file_name="ranked_candidates.csv",
            mime="text/csv",
            icon=":material/download:",
        )

    # --- Per-candidate detail ---
    st.subheader(t("detail_header", LANG))
    st.caption(t("detail_caption", LANG))

    for r in ranked:
        strength = match_strength(r["total_score_pct"], LANG)
        color = match_strength_color(r["total_score_pct"])
        header = (
            f"#{r['rank']}  {r['candidate_name']}   ·   "
            f"{r['total_score_pct']:.0f}%  ·  {strength}"
        )
        with st.expander(header, expanded=(r["rank"] == 1)):
            st.markdown(
                f":{color}-badge[{strength} {r['total_score_pct']:.0f}%] "
                f":blue-badge[{t('col_semantic', LANG)} {r['semantic_score_pct']:.0f}%] "
                f":blue-badge[{t('col_skill', LANG)} {r['skill_score_pct']:.0f}%]"
            )
            st.write(plain_explanation(r, w_sk_run, LANG))

            c_left, c_right = st.columns(2, gap="medium")

            with c_left:
                st.markdown(f"**{t('skills_have_title', LANG)}**")
                if r["matched_skills"]:
                    st.success(", ".join(r["matched_skills"]))
                else:
                    st.caption(t("skills_none_found", LANG))

                st.markdown(f"**{t('all_skills_title', LANG)}**")
                st.write(
                    ", ".join(r["all_extracted_skills"])
                    if r["all_extracted_skills"]
                    else t("none_found", LANG)
                )

            with c_right:
                st.markdown(f"**{t('skills_missing_title', LANG)}**")
                if r["missing_skills"]:
                    st.warning(", ".join(r["missing_skills"]))
                else:
                    st.success(t("all_present", LANG))

                st.markdown(f"**{t('contact_title', LANG)}**")
                nf = t("not_found_label", LANG)
                st.write(f"Email: {r['email'] or nf}   ·   Phone: {r['phone'] or nf}")
                if r["education"]:
                    st.write(" ; ".join(r["education"][:3]))

            with st.expander(t("technical_title", LANG)):
                st.markdown(
                    f"- Score formula: `{r['explanation']}`\n"
                    f"- Raw reading route: `{r['routed_mode']}`\n"
                    f"- Reading engine: `{r['ner_mode']}`\n"
                    f"- End-to-end latency: `{r['end_to_end_latency_sec']} s`"
                )
