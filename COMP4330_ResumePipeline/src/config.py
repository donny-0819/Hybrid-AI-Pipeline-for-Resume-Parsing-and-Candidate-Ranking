from pathlib import Path
from typing import Dict, List
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "environment_config.yaml"

DEFAULT_CONFIG = {
    "ingestion": {
        "digital_char_density_threshold": 50,
        "ocr_dpi": 300,
        "tesseract_oem": 3,
        "tesseract_psm": 6,
        "tesseract_lang": "eng",
        "opencv_presets": {
            "grayscale": True,
            "denoise_h": 10,
            "deskew_enabled": True,
            "deskew_max_angle_deg": 15.0,
            "adaptive_binarization": True,
            "adaptive_block_size": 31,
            "adaptive_c": 10,
        },
    },
    "ner": {
        "baseline_model": "en_core_web_sm",
        "upgraded_model": "en_core_web_trf",
        "sqlite_db_path": "data/candidates.db",
    },
    "matching": {
        "sbert_model": "all-MiniLM-L6-v2",
        "semantic_weight": 0.70,
        "skill_weight": 0.30,
    },
    "evaluation": {
        "top_k": 5,
        "relevance_threshold_for_precision": 2,
    },
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        merged = DEFAULT_CONFIG.copy()
        merged.update(loaded)
        return merged
    return DEFAULT_CONFIG


# Canonical IT Skills Gazetteer spanning Cybersecurity, Software Engineering, and Data Analysis
IT_SKILLS_GAZETTEER: List[str] = [
    "Python",
    "Java",
    "C++",
    "C#",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Redis",
    "HTML",
    "CSS",
    "React",
    "Next.js",
    "Node.js",
    "FastAPI",
    "Flask",
    "Django",
    "Spring Boot",
    "REST API",
    "GraphQL",
    "Docker",
    "Kubernetes",
    "Git",
    "CI/CD",
    "GitHub Actions",
    "Linux",
    "Bash",
    "AWS",
    "Azure",
    "GCP",
    "Terraform",
    "Microservices",
    "PyTorch",
    "TensorFlow",
    "Scikit-learn",
    "Pandas",
    "NumPy",
    "Tableau",
    "Power BI",
    "Excel",
    "Apache Spark",
    "Airflow",
    "ETL",
    "A/B Testing",
    "Statistical Modeling",
    "Data Warehousing",
    "Snowflake",
    "SIEM",
    "Splunk",
    "Wireshark",
    "Nmap",
    "Metasploit",
    "Burp Suite",
    "Penetration Testing",
    "Incident Response",
    "Threat Hunting",
    "Vulnerability Assessment",
    "Network Security",
    "Firewalls",
    "IDS/IPS",
    "ISO 27001",
    "NIST",
    "OWASP",
    "Cryptography",
    "Identity and Access Management",
]

# Synonym & Acronym Mapping -> Canonical Skill Name
SKILL_SYNONYMS: Dict[str, str] = {
    "k8s": "Kubernetes",
    "kube": "Kubernetes",
    "js": "JavaScript",
    "ecmascript": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "python3": "Python",
    "golang": "Go",
    "postgres": "PostgreSQL",
    "psql": "PostgreSQL",
    "mongo": "MongoDB",
    "reactjs": "React",
    "react.js": "React",
    "nextjs": "Next.js",
    "nodejs": "Node.js",
    "restful api": "REST API",
    "restful apis": "REST API",
    "rest apis": "REST API",
    "rest": "REST API",
    "amazon web services": "AWS",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    "microsoft azure": "Azure",
    "cicd": "CI/CD",
    "continuous integration": "CI/CD",
    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "powerbi": "Power BI",
    "ms excel": "Excel",
    "microsoft excel": "Excel",
    "pentesting": "Penetration Testing",
    "pen testing": "Penetration Testing",
    "ethical hacking": "Penetration Testing",
    "owasp top 10": "OWASP",
    "iam": "Identity and Access Management",
    "ids": "IDS/IPS",
    "ips": "IDS/IPS",
    "infosec": "Network Security",
}


def canonicalize_skill(raw_skill: str) -> str:
    cleaned = raw_skill.strip()
    low = cleaned.lower()
    if low in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[low]
    for canonical in IT_SKILLS_GAZETTEER:
        if canonical.lower() == low:
            return canonical
    return cleaned
