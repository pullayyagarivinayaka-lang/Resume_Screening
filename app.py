"""
AI-Powered Resume Screening & Job Recommendation API

Vercel + FastAPI backend.

Project structure:

Resume_Screening/
├── api.py
├── app.py
├── jobs.csv
├── skills.json
├── resume_parser.py
├── skill_extractor.py
├── recommender.py
├── matcher.py
└── requirements.txt
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))

from resume_parser import extract_text, UnsupportedFileTypeError
from recommender import recommend_jobs


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

JOBS_FILE = BASE_DIR / "jobs.csv"
SKILLS_FILE = BASE_DIR / "skills.json"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Resume Screening & Job Recommendation API",
    description=(
        "AI-powered resume screening and personalized job "
        "recommendation system."
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
# LOAD JOB DATA
# ============================================================

def load_jobs():
    """Load jobs.csv."""

    if not JOBS_FILE.exists():
        raise FileNotFoundError(
            f"jobs.csv was not found at: {JOBS_FILE}"
        )

    return pd.read_csv(JOBS_FILE)


# ============================================================
# LOAD SKILLS TAXONOMY
# ============================================================

def load_taxonomy():
    """Load skills.json."""

    if not SKILLS_FILE.exists():
        raise FileNotFoundError(
            f"skills.json was not found at: {SKILLS_FILE}"
        )

    with open(SKILLS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "AI Resume Screening API is running",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "resume-screening-api",
    }


# ============================================================
# GET AVAILABLE JOBS
# ============================================================

@app.get("/api/jobs")
def get_jobs():

    try:
        jobs_df = load_jobs()

        jobs_df = jobs_df.fillna("")

        jobs = jobs_df.to_dict(
            orient="records"
        )

        return {
            "status": "success",
            "total_jobs": len(jobs),
            "jobs": jobs,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to load jobs: {str(error)}",
        )


# ============================================================
# RESUME FILE ADAPTER
# ============================================================

class UploadedFileAdapter:
    """
    Adapter that makes FastAPI UploadFile compatible with
    parsers expecting a Streamlit-style uploaded file.
    """

    def __init__(
        self,
        name: str,
        content: bytes,
    ):
        self.name = name
        self._content = content
        self.size = len(content)
        self._position = 0

    def read(self, size=-1):

        if size == -1:

            result = self._content[
                self._position:
            ]

            self._position = len(
                self._content
            )

            return result

        result = self._content[
            self._position:
            self._position + size
        ]

        self._position += len(result)

        return result

    def getvalue(self):
        return self._content

    def seek(self, position):
        self._position = position

    def tell(self):
        return self._position


# ============================================================
# EXTRACT RESUME TEXT
# ============================================================

@app.post("/api/extract")
async def extract_resume(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    filename = file.filename.lower()

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    if not (
        filename.endswith(".pdf")
        or filename.endswith(".docx")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Please upload a PDF or DOCX file."
            ),
        )

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    try:

        content = await file.read()

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read file: {str(error)}",
        )

    # --------------------------------------------------------
    # Validate file size
    # --------------------------------------------------------

    max_size = 5 * 1024 * 1024

    if len(content) > max_size:

        raise HTTPException(
            status_code=400,
            detail="File is too large. Maximum size is 5 MB.",
        )

    if len(content) == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Create adapter
    # --------------------------------------------------------

    uploaded_file = UploadedFileAdapter(
        file.filename,
        content,
    )

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    try:

        resume_text = extract_text(
            file.filename,
            uploaded_file,
        )

    except UnsupportedFileTypeError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Resume text extraction failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # Validate extracted text
    # --------------------------------------------------------

    if not resume_text or not resume_text.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text could be extracted. "
                "If this is a scanned PDF, upload a "
                "text-based PDF."
            ),
        )

    return {
        "status": "success",
        "filename": file.filename,
        "text_length": len(resume_text),
        "text": resume_text,
    }


# ============================================================
# MAIN RECOMMENDATION ENDPOINT
# ============================================================

@app.post("/api/recommend")
async def recommend_jobs_api(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

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
            detail=(
                "Only PDF and DOCX files are supported."
            ),
        )

    # --------------------------------------------------------
    # Read uploaded resume
    # --------------------------------------------------------

    try:

        content = await file.read()

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read file: {str(error)}",
        )

    # --------------------------------------------------------
    # Validate file size
    # --------------------------------------------------------

    max_size = 5 * 1024 * 1024

    if len(content) > max_size:

        raise HTTPException(
            status_code=400,
            detail="File is too large. Maximum size is 5 MB.",
        )

    if len(content) == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Convert to parser-compatible object
    # --------------------------------------------------------

    uploaded_file = UploadedFileAdapter(
        file.filename,
        content,
    )

    # --------------------------------------------------------
    # Extract resume text
    # --------------------------------------------------------

    try:

        resume_text = extract_text(
            file.filename,
            uploaded_file,
        )

    except UnsupportedFileTypeError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Resume extraction failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # Check extracted text
    # --------------------------------------------------------

    if not resume_text or not resume_text.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text could be extracted "
                "from the resume."
            ),
        )

    # --------------------------------------------------------
    # Load jobs
    # --------------------------------------------------------

    try:

        jobs_df = load_jobs()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to load jobs.csv: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # Load skills taxonomy
    # --------------------------------------------------------

    try:

        taxonomy = load_taxonomy()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to load skills.json: "
                f"{str(error)}"
            ),
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

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Job recommendation failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # Convert results to JSON
    # --------------------------------------------------------

    recommendations = []

    for _, row in results.iterrows():

        # ----------------------------------------------
        # Job title
        # ----------------------------------------------

        job_title = row.get(
            "job_title",
            "",
        )

        # ----------------------------------------------
        # Company
        # ----------------------------------------------

        company = row.get(
            "company",
            "",
        )

        # ----------------------------------------------
        # Location
        # ----------------------------------------------

        location = row.get(
            "location",
            "",
        )

        # ----------------------------------------------
        # Match score
        # ----------------------------------------------

        match_score = row.get(
            "match_score",
            0,
        )

        # ----------------------------------------------
        # Matched skills
        # ----------------------------------------------

        matched_skills = row.get(
            "matched_skills",
            [],
        )

        # ----------------------------------------------
        # Missing skills
        # ----------------------------------------------

        missing_skills = row.get(
            "missing_skills",
            [],
        )

        # ----------------------------------------------
        # Explanation
        # ----------------------------------------------

        explanation = row.get(
            "explanation",
            "",
        )

        # ----------------------------------------------
        # Handle NaN
        # ----------------------------------------------

        if pd.isna(match_score):
            match_score = 0

        # ----------------------------------------------
        # Convert strings into lists
        # ----------------------------------------------

        if isinstance(
            matched_skills,
            str,
        ):

            matched_skills = [
                matched_skills
            ]

        elif not isinstance(
            matched_skills,
            list,
        ):

            matched_skills = list(
                matched_skills
            ) if matched_skills else []

        if isinstance(
            missing_skills,
            str,
        ):

            missing_skills = [
                missing_skills
            ]

        elif not isinstance(
            missing_skills,
            list,
        ):

            missing_skills = list(
                missing_skills
            ) if missing_skills else []

        # ----------------------------------------------
        # Create recommendation
        # ----------------------------------------------

        recommendations.append(
            {
                "job_title": str(
                    job_title
                ),

                "company": str(
                    company
                ),

                "location": str(
                    location
                ),

                "match_score": round(
                    float(match_score),
                    2,
                ),

                "matched_skills": [
                    str(skill)
                    for skill in matched_skills
                ],

                "missing_skills": [
                    str(skill)
                    for skill in missing_skills
                ],

                "explanation": str(
                    explanation
                ),
            }
        )

    # --------------------------------------------------------
    # Final API response
    # --------------------------------------------------------

    return {
        "status": "success",

        "message": (
            "Resume analyzed successfully."
        ),

        "resume": {
            "filename": file.filename,
            "file_size_bytes": len(content),
            "extracted_text_length": len(
                resume_text
            ),
        },

        "total_recommendations": len(
            recommendations
        ),

        "recommendations": recommendations,

        "disclaimer": (
            "Match scores are recommendation metrics "
            "based on skill overlap and text similarity. "
            "They are not guaranteed hiring decisions."
        ),
    }
