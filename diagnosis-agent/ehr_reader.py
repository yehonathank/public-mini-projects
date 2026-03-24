"""
Synthetic EHR for CDS simulation: load JSON, host summary, and category slices for `check_chart`.
"""

from __future__ import annotations

import json
from pathlib import Path

EHR_DEFAULT_PATH = Path(__file__).resolve().parent / "ehr" / "ehr_patient_1.json"

# Top-level keys exposed to `check_chart(category=...)`.
CHART_CATEGORIES: list[str] = [
    "patient_id",
    "demographics",
    "past_medical_history",
    "medications",
    "allergies",
    "vitals",
    "recent_encounters",
]

# Node front matter `ehr_auto_when:` values — evaluated by the host before showing `patient:` text.
EHR_AUTO_WHEN_PREDICATES: tuple[str, ...] = (
    "appendectomy_in_pmh",
    "biological_sex_male",
)


def _pmh_blob(ehr: dict) -> str:
    return " ".join(str(x).lower() for x in (ehr.get("past_medical_history") or []))


def ehr_predicate_holds(when: str, ehr: dict) -> bool:
    """
    Deterministic chart checks for host pre-display redirects (`ehr_auto_when` in node front matter).
    Unknown predicate names return False (no redirect).
    """
    key = when.strip().lower().replace("-", "_")
    if key == "appendectomy_in_pmh":
        s = _pmh_blob(ehr)
        return "appendect" in s or "appendix" in s
    if key == "biological_sex_male":
        sex = str((ehr.get("demographics") or {}).get("biological_sex") or "").strip().lower()
        return sex == "male"
    return False


def load_ehr(path: Path | None = None) -> dict:
    p = path or EHR_DEFAULT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"EHR file not found: {p}")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def slice_category(data: dict, category: str) -> dict:
    """Return a single-key object for logging / tool payload."""
    if category not in CHART_CATEGORIES:
        return {
            "error": "unknown_category",
            "category": category,
            "allowed": CHART_CATEGORIES,
        }
    if category == "patient_id":
        return {"patient_id": data.get("patient_id")}
    val = data.get(category)
    if val is None:
        return {category: None, "note": "missing in record"}
    return {category: val}


def format_slice_json(data: dict, category: str) -> str:
    return json.dumps(slice_category(data, category), ensure_ascii=False, indent=2)


def host_ehr_summary(data: dict) -> str:
    """
    Deterministic 'chart-first' digest injected into every routing system prompt.
    Not from an LLM — reproducible for experiments.
    """
    lines: list[str] = []
    pid = data.get("patient_id")
    if pid:
        lines.append(f"- **patient_id:** `{pid}`")

    dem = data.get("demographics") or {}
    age = dem.get("age")
    sex = dem.get("biological_sex")
    lines.append(f"- **Demographics:** age {age}, biological sex {sex}")

    pmh = data.get("past_medical_history") or []
    if pmh:
        lines.append(f"- **PMH:** {', '.join(str(x) for x in pmh)}")
        low = " ".join(str(x).lower() for x in pmh)
        if "appendect" in low or "appendix" in low:
            lines.append(
                "- **Chart flag:** Prior appendectomy documented — acute *de novo* appendicitis is unlikely; still evaluate other LRQ pathology."
            )

    meds = data.get("medications") or []
    if meds:
        lines.append(f"- **Medications:** {', '.join(str(x) for x in meds)}")

    enc = data.get("recent_encounters") or []
    if enc:
        last = enc[-1]
        lines.append(
            f"- **Recent encounter:** {last.get('date')} — {last.get('reason')} ({last.get('diagnosis')})"
        )

    vit = data.get("vitals") or {}
    if vit:
        lines.append(f"- **Last vitals (synthetic):** {vit}")

    return "\n".join(lines) if lines else "_Empty EHR._"


def format_prefetch_block(data: dict, categories: list[str]) -> str:
    """Format multiple EHR slices (helper for tooling / experiments; `chat.py` does not inject this into prompts)."""
    if not categories:
        return "_No categories requested._"
    parts: list[str] = []
    for cat in categories:
        parts.append(f"##### `{cat}`\n```json\n{format_slice_json(data, cat)}\n```")
    return "\n\n".join(parts)
