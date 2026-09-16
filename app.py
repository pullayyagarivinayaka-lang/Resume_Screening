"""
AI-Powered Resume Screening & Job Recommendation API

Vercel-compatible FastAPI backend.

Features:
- Resume upload: PDF/DOCX
- Resume text extraction
- Skill extraction
- TF-IDF + skill-overlap job matching
- Top 10 job recommendations
- Matched and missing skills
- Explainable recommendations
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.resume_parser import extract_text, UnsupportedFileTypeError
from src.recommender import recommend_jobs


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

JOBS_FILE = BASE_DIR / "data" / "raw" / "jobs.csv"
TAXONOMY_FILE = BASE_DIR / "data" / "taxonomy" / "skills.json"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Resume Screening & Job Recommendation API",
    description=(
        "API for extracting resume information and recommending "
        "jobs using TF-IDF similarity and skill overlap."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DATA
# ============================================================

def load_jobs():
    """
    Load jobs dataset.
    """
    if not JOBS_FILE.exists():
        raise FileNotFoundError(
            f"Jobs dataset not found: {JOBS_FILE}"
        )

    return pd.read_csv(JOBS_FILE)


def load_taxonomy():
    """
    Load skills taxonomy.
    """
    if not TAXONOMY_FILE.exists():
        raise FileNotFoundError(
            f"Skills taxonomy not found: {TAXONOMY_FILE}"
        )

    with open(TAXONOMY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "success",
        "message": "AI Resume Screening & Job Recommendation API is running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "resume-screening-api",
    }


# ============================================================
# JOB DATASET ENDPOINT
# ============================================================

@app.get("/api/jobs")
async def get_jobs():
    """
    Return available jobs from the dataset.
    """

    try:
        jobs_df = load_jobs()

        jobs = jobs_df.fillna("").to_dict(orient="records")

        return {
            "status": "success",
            "total_jobs": len(jobs),
            "jobs": jobs,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load jobs: {str(e)}",
        )


# ============================================================
# RESUME RECOMMENDATION ENDPOINT
# ============================================================

@app.post("/api/recommend")
async def recommend(
    file: UploadFile = File(...)
):
    """
    Upload a resume and receive job recommendations.
    """

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    filename = file.filename.lower()

    allowed_extensions = [".pdf", ".docx"]

    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX resume files are supported.",
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    try:
        file_content = await file.read()

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read uploaded file: {str(e)}",
        )

    # --------------------------------------------------------
    # File size validation
    # --------------------------------------------------------

    max_size_mb = 5
    max_size_bytes = max_size_mb * 1024 * 1024

    if len(file_content) > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Maximum size is {max_size_mb} MB.",
        )

    if len(file_content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Create file adapter
    #
    # This makes FastAPI's uploaded file compatible with
    # parsers that expect a Streamlit-style uploaded file.
    # --------------------------------------------------------

    class UploadedFileAdapter:

        def __init__(self, name, content):
            self.name = name
            self._content = content
            self.size = len(content)
            self._position = 0

        def read(self, size=-1):
            if size == -1:
                result = self._content[self._position:]
                self._position = len(self._content)
                return result

            result = self._content[
                self._position:self._position + size
            ]

            self._position += len(result)

            return result

        def getvalue(self):
            return self._content

        def seek(self, position):
            self._position = position

        def tell(self):
            return self._position

    uploaded_file = UploadedFileAdapter(
        file.filename,
        file_content,
    )

    # --------------------------------------------------------
    # Extract resume text
    # --------------------------------------------------------

    try:

        resume_text = extract_text(
            file.filename,
            uploaded_file,
        )

    except UnsupportedFileTypeError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Resume text extraction failed: {str(e)}",
        )

    # --------------------------------------------------------
    # Validate extracted text
    # --------------------------------------------------------

    if not resume_text or not resume_text.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text could be extracted from the resume. "
                "If the PDF is scanned/image-only, upload a text-based PDF."
            ),
        )

    # --------------------------------------------------------
    # Load jobs and taxonomy
    # --------------------------------------------------------

    try:

        jobs_df = load_jobs()
        taxonomy = load_taxonomy()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to load recommendation data: {str(e)}",
        )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    try:

        results = recommend_jobs(
            resume_text,
            jobs_df,
            taxonomy,
            top_k=10,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Job recommendation failed: {str(e)}",
        )

    # --------------------------------------------------------
    # Convert results to JSON-safe format
    # --------------------------------------------------------

    recommendations = []

    for _, row in results.iterrows():

        matched_skills = row.get(
            "matched_skills",
            [],
        )

        missing_skills = row.get(
            "missing_skills",
            [],
        )

        # Convert possible strings/NaN values safely
        if pd.isna(matched_skills):
            matched_skills = []

        if pd.isna(missing_skills):
            missing_skills = []

        if isinstance(matched_skills, str):
            matched_skills = [matched_skills]

        if isinstance(missing_skills, str):
            missing_skills = [missing_skills]

        recommendation = {
            "job_title": str(
                row.get("job_title", "")
            ),

            "company": str(
                row.get("company", "")
            ),

            "location": str(
                row.get("location", "")
            ),

            "match_score": float(
                row.get("match_score", 0)
            ),

            "matched_skills": matched_skills,

            "missing_skills": missing_skills,

            "explanation": str(
                row.get("explanation", "")
            ),
        }

        recommendations.append(
            recommendation
        )

    # --------------------------------------------------------
    # Return API response
    # --------------------------------------------------------

    return {
        "status": "success",

        "message": (
            "Resume analyzed and job recommendations generated."
        ),

        "resume": {
            "filename": file.filename,
            "file_size_bytes": len(file_content),
            "extracted_text_length": len(resume_text),
        },

        "total_recommendations": len(
            recommendations
        ),

        "recommendations": recommendations,

        "disclaimer": (
            "Match scores are recommendation metrics based on "
            "skill overlap and text similarity. They are not "
            "guaranteed hiring decisions."
        ),
    }


# ============================================================
# RESUME TEXT ENDPOINT
# ============================================================

@app.post("/api/extract")
async def extract_resume_text(
    file: UploadFile = File(...)
):
    """
    Extract text from a PDF or DOCX resume.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    filename = file.filename.lower()

    if not (
        filename.endswith(".pdf")
        or filename.endswith(".docx")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported.",
        )

    try:

        content = await file.read()

        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File must be smaller than 5 MB.",
            )

        class UploadedFileAdapter:

            def __init__(self, name, content):
                self.name = name
                self._content = content
                self.size = len(content)
                self._position = 0

            def read(self, size=-1):
                if size == -1:
                    result = self._content[self._position:]
                    self._position = len(self._content)
                    return result

                result = self._content[
                    self._position:self._position + size
                ]

                self._position += len(result)

                return result

            def getvalue(self):
                return self._content

            def seek(self, position):
                self._position = position

        uploaded_file = UploadedFileAdapter(
            file.filename,
            content,
        )

        text = extract_text(
            file.filename,
            uploaded_file,
        )

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text found in the resume.",
            )

        return {
            "status": "success",
            "filename": file.filename,
            "text_length": len(text),
            "text": text,
        }

    except UnsupportedFileTypeError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Text extraction failed: {str(e)}",
        )
