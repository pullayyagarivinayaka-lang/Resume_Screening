# AI-Powered Resume Screening and Personalized Job Recommendation System

An end-to-end Data Science project that parses resumes and job descriptions,
matches candidates to jobs using TF-IDF and semantic (Sentence-Transformer)
similarity, produces an explainable weighted match score, performs skill-gap
analysis, and surfaces everything through a Streamlit app with a recruiter
console and an analytics dashboard.

> **Status:** Under active development, stage-by-stage. See `docs/roadmap.md`
> (added later) for progress.

## Tech Stack

- **Language:** Python 3.11+
- **Data Science:** pandas, NumPy, scikit-learn, matplotlib, seaborn
- **NLP:** spaCy, NLTK, TF-IDF, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Database:** MySQL + SQLAlchemy
- **Web App:** Streamlit
- **Visualization:** Plotly

## Prerequisites

- Python 3.11 or newer
- MySQL Server 8.x running locally (or a hosted instance)
- Git

## Setup Instructions

1. **Clone the repository and enter the project folder**
   ```bash
   git clone <your-repo-url>
   cd resume-job-matcher
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Set up MySQL**
   ```sql
   CREATE DATABASE resume_matcher_db;
   ```

5. **Configure environment variables**
   ```bash
   cp .env.example .env
   # then edit .env with your real MySQL credentials
   ```

6. **Run the app**
   ```bash
   streamlit run app.py
   ```

   The app opens at `http://localhost:8501`.

## Project Structure

See `docs/architecture.md` (added in a later stage) for the full breakdown.
The short version:

```
resume-job-matcher/
├── app.py              # Streamlit entry point
├── config.yaml         # central configuration (scoring weights, paths, DB)
├── src/                # core logic: parsing, NLP, matching, recommending
├── pages/              # Streamlit multi-page UI (Candidate/Recruiter/Analytics)
├── data/                # synthetic dataset + skill taxonomy
├── notebooks/          # EDA and model evaluation notebooks
├── docs/               # final-year project documentation
└── tests/              # unit tests
```

## Data & Privacy Note

Uploaded resumes are stored under `uploads/` (gitignored, not publicly served)
and are referenced by file path only — no resume text is stored in the
database beyond the structured fields the candidate consents to extract.
See `docs/security.md` (added later) for the full data-handling policy.

## License

Add your chosen license here (e.g., MIT) before publishing.
