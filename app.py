"""
AI-Powered Resume Screening and Personalized Job Recommendation System
MVP: Upload a resume -> extract skills -> match against a synthetic
job dataset using TF-IDF + skill overlap -> show ranked recommendations
with explainable skill-gap analysis.

No database yet (added in a later stage). This is intentionally the
fastest path to a live, demoable link.
"""

import json
import pandas as pd
import streamlit as st

from src.resume_parser import extract_text, UnsupportedFileTypeError
from src.recommender import recommend_jobs

st.set_page_config(
    page_title="AI Resume Screening & Job Recommendation",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def load_jobs():
    return pd.read_csv("data/raw/jobs.csv")


@st.cache_data
def load_taxonomy():
    with open("data/taxonomy/skills.json", "r", encoding="utf-8") as f:
        return json.load(f)


jobs_df = load_jobs()
taxonomy = load_taxonomy()

st.title("🧠 AI-Powered Resume Screening & Job Recommendation System")
st.caption(
    "MVP version — matches your resume against a sample job dataset using "
    "TF-IDF text similarity + skill-taxonomy overlap. Semantic matching, a "
    "real database, and a recruiter console are being added next."
)

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    max_size_mb = 5
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        st.error(f"File is too large. Please upload a file under {max_size_mb} MB.")
        st.stop()

    try:
        with st.spinner("Extracting text from your resume..."):
            resume_text = extract_text(uploaded_file.name, uploaded_file)
    except UnsupportedFileTypeError as e:
        st.error(str(e))
        st.stop()

    if not resume_text.strip():
        st.warning(
            "No readable text could be extracted from this file. "
            "If it's a scanned PDF (image-only), please upload a text-based version."
        )
        st.stop()

    with st.expander("Show extracted resume text"):
        st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

    with st.spinner("Matching your resume against available jobs..."):
        results = recommend_jobs(resume_text, jobs_df, taxonomy, top_k=10)

    st.subheader("Top Job Recommendations")

    for _, row in results.iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {row['job_title']} — {row['company']}")
                st.caption(f"📍 {row['location']}")
            with col2:
                st.metric("Match Score", f"{row['match_score']}%")

            st.progress(min(int(row["match_score"]), 100) / 100)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ Matched Skills**")
                if row["matched_skills"]:
                    st.write(", ".join(row["matched_skills"]))
                else:
                    st.write("_None detected_")
            with c2:
                st.markdown("**❌ Missing Skills**")
                if row["missing_skills"]:
                    st.write(", ".join(row["missing_skills"]))
                else:
                    st.write("_None — full skill coverage_")

            st.info(row["explanation"])

    st.caption(
        "Match Score is a recommendation metric based on skill overlap and text "
        "similarity — it is not a guaranteed hiring decision."
    )

else:
    st.info("👆 Upload a resume to see job recommendations.")

    with st.expander("What does this MVP do vs. what's coming next?"):
        st.markdown(
            """
            **Working now:**
            - PDF/DOCX resume text extraction
            - Skill extraction against a taxonomy
            - TF-IDF + skill-overlap based matching against sample jobs
            - Explainable match scores with matched/missing skills

            **Coming in later stages:**
            - Real MySQL database (candidates, jobs, recommendations persisted)
            - Semantic matching with Sentence-Transformers
            - Recruiter console (post jobs, rank candidates)
            - Full analytics dashboard (Plotly, KPIs)
            - Larger synthetic dataset (500+ jobs/candidates)
            - Formal evaluation metrics (Precision@K, NDCG, MRR)
            """
        )
