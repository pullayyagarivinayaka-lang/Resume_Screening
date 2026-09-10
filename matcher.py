"""
matcher.py
Baseline matching engine: TF-IDF vectorization + Cosine Similarity between
a candidate's resume text and a set of job descriptions.

This is intentionally the "Method 1" baseline from the project spec.
Method 2 (Sentence-Transformer semantic similarity) gets added in a later
stage once the MVP is live, without changing this module's interface.
"""

from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_match_scores(resume_text: str, jobs_df: pd.DataFrame) -> pd.Series:
    """Compute TF-IDF cosine similarity between a resume and every job.

    Args:
        resume_text: cleaned/raw resume text (single string).
        jobs_df: DataFrame with at least a 'description' column.

    Returns:
        A pandas Series of similarity scores (0-1), indexed the same as
        jobs_df, so it can be assigned directly as a new column.
    """
    if not resume_text or not resume_text.strip():
        # No usable text extracted from the resume — return all zeros
        # rather than crashing the vectorizer on empty input.
        return pd.Series([0.0] * len(jobs_df), index=jobs_df.index)

    documents = [resume_text] + jobs_df["description"].fillna("").tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    resume_vector = tfidf_matrix[0]
    job_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(resume_vector, job_vectors).flatten()
    return pd.Series(similarities, index=jobs_df.index)


def rank_jobs_for_resume(resume_text: str, jobs_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Return the top_k jobs ranked by TF-IDF match score for this resume.

    Adds a 'match_score' column (0-100, rounded) to the result.
    """
    scores = compute_tfidf_match_scores(resume_text, jobs_df)
    result = jobs_df.copy()
    result["match_score"] = (scores * 100).round(1)
    result = result.sort_values("match_score", ascending=False).head(top_k)
    return result.reset_index(drop=True)
