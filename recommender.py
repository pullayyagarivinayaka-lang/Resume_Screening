"""
recommender.py
Combines skill-overlap scoring and TF-IDF text-similarity scoring into a
single, explainable match score per job, then ranks jobs for a candidate.

MVP weighting (simplified from the full spec's 5-component weighting,
which gets restored once experience/education extraction is built):
    final_score = 0.6 * skills_score + 0.4 * tfidf_score

This keeps the interface stable: recommend_jobs() is what the Streamlit
app calls, and its return shape won't change when we add the semantic
Sentence-Transformer score and the experience/education components later.
"""

from __future__ import annotations
import pandas as pd

from src.resume_parser import extract_text, UnsupportedFileTypeError
from src.recommender import recommend_jobs

SKILLS_WEIGHT = 0.6
TFIDF_WEIGHT = 0.4


def _skill_score(resume_skills: set, job_skills: set) -> float:
    """Fraction of the job's required skills that the resume covers (0-1)."""
    if not job_skills:
        return 0.0
    return len(resume_skills & job_skills) / len(job_skills)


def recommend_jobs(resume_text: str, jobs_df: pd.DataFrame, taxonomy: dict, top_k: int = 10) -> pd.DataFrame:
    """Rank all jobs in jobs_df for the given resume text.

    Returns a DataFrame with columns:
        job_id, job_title, company, location, match_score,
        skills_score, semantic_score (placeholder, filled in a later stage),
        matched_skills, missing_skills, explanation
    sorted by match_score descending, limited to top_k rows.
    """
    resume_skills = extract_skills(resume_text, taxonomy)
    tfidf_scores = compute_tfidf_match_scores(resume_text, jobs_df)

    rows = []
    for idx, job in jobs_df.iterrows():
        job_skills = extract_skills(job.get("description", "") + " " + str(job.get("required_skills", "")), taxonomy)
        s_score = _skill_score(resume_skills, job_skills)
        t_score = tfidf_scores.loc[idx]

        final_score = SKILLS_WEIGHT * s_score + TFIDF_WEIGHT * t_score

        gap = skill_gap(resume_skills, job.get("description", "") + " " + str(job.get("required_skills", "")), taxonomy)

        if gap["matched"]:
            strongest = ", ".join(gap["matched"][:4])
            explanation = f"Your resume matches {len(gap['matched'])} of {len(gap['matched']) + len(gap['missing'])} required skills. Strongest matches: {strongest}."
            if gap["missing"]:
                explanation += f" Main gap: {', '.join(gap['missing'][:3])}."
        else:
            explanation = "No direct skill overlap detected; ranking is based on overall text similarity only."

        rows.append({
            "job_id": job.get("job_id"),
            "job_title": job.get("job_title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "match_score": round(final_score * 100, 1),
            "skills_score": round(s_score * 100, 1),
            "semantic_score": None,  # filled in once Sentence-Transformers are added
            "matched_skills": gap["matched"],
            "missing_skills": gap["missing"],
            "explanation": explanation,
        })

    result = pd.DataFrame(rows).sort_values("match_score", ascending=False).head(top_k)
    return result.reset_index(drop=True)
