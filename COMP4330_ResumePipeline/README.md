# A Hybrid AI Pipeline for Resume Parsing and Candidate Ranking

**Course:** COMP 4330SEF — Advanced Programming and AI Algorithms (HKMU)  
**Group:** P07 Group 3  
- **FU Chuen Yin (14018512)** — Project Leader & Recruiter Interface Lead (`app.py`)
- **KAN Chung Wing (13897830)** — Semantic Matching & Ranking Algorithm Lead (`src/tier3_matching.py`)
- **SIN Wai Siu (12957821)** — Document Ingestion & Image OCR Lead (`src/tier1_ingestion.py`)
- **MOY Tsz Fung Donny (13901104)** — Information Extraction / NER Lead (`src/tier2_extraction.py`)
- **CHEN Weihua (14143648)** — Literature Review & Evaluation Benchmark Lead (`evaluate_benchmarks.py`, `scripts/degrade_scans.py`)

---

## 1. System Prerequisites
1. **Python 3.11**
2. **Tesseract OCR Binary** & **Poppler** (for `pdf2image`):
   - **Windows:** `winget install UB-Mannheim.TesseractOCR`
   - **macOS:** `brew install tesseract poppler`
   - **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr poppler-utils`

## 2. Installation
```bash
python -m venv .venv

# Windows:
py -m venv .venv
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

## 3. Running the Tools

### A. Launch the Local Recruiter Dashboard (Tier 4)
```bash
streamlit run app.py
```

On Windows you can also double-click `run_dashboard.bat`.

The dashboard is written for a recruiter, not an engineer. The screen flow is
**1. Choose the job → 2. Add resumes → 3. Results**. Every algorithmic setting
(spaCy NER track, SBERT vs. TF-IDF ranking, OpenCV preprocessing, SQLite
persistence) still works, but lives under **Advanced settings** in the sidebar
and is labelled in plain English. The raw values remain visible per candidate
under "Technical details for this candidate", so the dashboard can still be used
as evidence in the report.

### Appearance and language

Both are switched at runtime from the **Appearance** panel in the sidebar:

- **Language** — English or 繁體中文 (Traditional Chinese). Every label,
  table column, caption and per-candidate explanation is translated.
- **Colour scheme** — Blue (default), Teal, Purple, Forest, Sunset, or
  Midnight (dark). The choice is applied as a scoped CSS override on top of
  the base theme in `.streamlit/config.toml`, so no restart is needed.

Language strings live in `src/i18n.py`; colour palettes live in `src/ui_theme.py`;
the plain-language technical wording lives in `src/ui_labels.py`. None of these
touch the ranking logic in `src/tier3_matching.py`.

### B. Generate Controlled Smartphone-Degraded Scans (Aim B)
```bash
python scripts/degrade_scans.py --input_dir data/clean_resumes --output_dir data/scanned_resumes --skew 4.2
```

Ready-made samples (digital PDFs, degraded JPG scans, and ground-truth text) for trying the dashboard:
```bash
python evaluate_benchmarks.py --make-samples   # writes data/sample_resumes/
```

### C. Run Quantitative Benchmarks
```bash
python evaluate_benchmarks.py --run-ocr-demo   # CER/WER: digital vs raw scan vs OpenCV-cleaned scan (Hypothesis 3)
python evaluate_benchmarks.py --run-demo       # NER F1, Precision@K, nDCG@K, keyword vs SBERT vs hybrid (Hypotheses 1-2)
```

### D. Real Resume Test Corpus (24 public resumes)
```bash
python scripts/fetch_real_resumes.py    # downloads PDFs + phone-scan copies + grading_sheet.csv to data/real_resumes/
python scripts/check_real_resumes.py    # parses all, ranks against the 3 JDs, compares scans with their digital PDFs
```
Source: Kaggle Resume Dataset (LiveCareer), mirrored at https://huggingface.co/datasets/opensporks/resumes. Personal details are already anonymised.

Tesseract and Poppler are auto-detected in their default Windows install locations even if they are not on `PATH`.
