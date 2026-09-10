"""
skill_extractor.py
MVP version: simple case-insensitive substring matching against a skill
taxonomy. This is intentionally simple for the fast-launch MVP.

In a later stage this gets upgraded to use spaCy NER + lemmatization for
more robust extraction (handling plurals, abbreviations, synonyms, etc.)
per the full project spec's NLP pipeline requirement — the function
signature here (extract_skills) stays the same so nothing else needs to
change when that upgrade happens.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Set


def load_taxonomy(taxonomy_path: str) -> dict:
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_taxonomy(taxonomy: dict) -> List[str]:
    skills = []
    for category_skills in taxonomy.values():
        skills.extend(category_skills)
    return skills


def extract_skills(text: str, taxonomy: dict) -> Set[str]:
    """Extract skills mentioned in `text` by matching against the taxonomy.

    Uses word-boundary-aware matching so e.g. 'java' doesn't match inside
    'javascript'. Matching is case-insensitive. Returns the canonical
    (lowercase) skill names as they appear in the taxonomy.
    """
    text_lower = text.lower()
    found = set()

    for skill in _flatten_taxonomy(taxonomy):
        # Escape regex special chars (e.g. "c++") and match on word boundaries
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)

    return found


def skill_gap(resume_skills: Set[str], job_text: str, taxonomy: dict) -> dict:
    """Compare resume skills against a job's required skills.

    Returns a dict with 'matched', 'missing', and 'coverage_pct'.
    """
    job_skills = extract_skills(job_text, taxonomy)

    if not job_skills:
        return {"matched": [], "missing": [], "coverage_pct": 0.0}

    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    coverage_pct = round(100 * len(matched) / len(job_skills), 1)

    return {"matched": matched, "missing": missing, "coverage_pct": coverage_pct}
