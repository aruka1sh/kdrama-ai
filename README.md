# K-DRAMA AI - Discover Your Next Favorite Story

AI Fundamentals final exam project (QAIRU), **Variant 21 - Movie Preference**, adapted to Korean dramas.

**Live site:** <https://kdrama-ai.vercel.app> &nbsp;|&nbsp; **Team:** Shakenova Aruzhan (lead), Qabir Madina, Shahap Aizere, Zholdaspek Dana

| Part | What it is | Main file |
|---|---|---|
| Project 1 - Supervised | Will a viewer like a K-Drama? Logistic Regression vs Random Forest | `notebooks/Project1_Supervised_KDrama.ipynb` |
| Project 2 - Unsupervised | Groups, low-dimensional structure and unusual dramas | `notebooks/Project2_Unsupervised_KDrama.ipynb` |
| Reports | Two academic reports built from the real results | `reports/*.docx` |
| Presentation | One combined deck with speaker notes | `presentation/KDrama_AI_Presentation.pptx` |
| Website (extra) | Next.js site that calls the real model through FastAPI | `frontend/`, `backend/` |

## Data (real, nothing simulated)

Two public Kaggle datasets scraped from MyDramaList, downloaded automatically (no Kaggle account needed):

* **Source A** - C. Charuchinda (2023), *Korean Drama from 2015-2023 with Actors & Reviews*, CC0.
  <https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist>
* **Source B** - N. Rizki (2023), *Top Korean Drama List (~1500)*, licence not stated (classroom use only).
  <https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500>

`ml/kdrama_data.py` joins them into a **drama table** (1,094 dramas) and a **review table** (9,840 real viewer reviews).
Target of Project 1: `liked = 1` if the viewer's own score is 8 or more. Project 2 never loads the target.

## Project structure

```
ml/                  kdrama_data.py (data), train_supervised.py, analyze_unsupervised.py
notebooks/           the two Colab notebooks + build_notebooks.py (generator)
artifacts/           preference_model.joblib, *_results.json, figures/   (created by ml/)
backend/             FastAPI app (main.py = ML API, catalog.py + tvmaze.py + tmdb.py = drama catalogue) and API tests
frontend/            Next.js + TypeScript + Tailwind + Motion + Plotly website
reports/             Report1/Report2 .docx + build_reports.py
presentation/        .pptx + build_presentation.py
docs/                requirement checklists, team roles, speaking script, defence Q&A
qa/                  check_consistency.py
```

## Option A - only the notebooks (Google Colab)

1. Open <https://colab.research.google.com> -> File -> Upload notebook.
2. Upload `notebooks/Project1_Supervised_KDrama.ipynb` (or Project 2).
3. Runtime -> Run all. The data is downloaded by the notebook itself. Project 2 takes about one minute.

The notebooks are self-contained: they do not need any other file from this folder.

## Option B - everything on your computer

Requirements: Python 3.11+ and Node.js 20+.

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

### 1. Create the model and the analysis results

```bash
python ml/train_supervised.py
```
```bash
python ml/analyze_unsupervised.py
```

This downloads the data to `data/raw/` and writes `artifacts/` (model, JSON results, figures).

### 2. Rebuild and run the notebooks, reports and slides (optional)

```bash
python notebooks/build_notebooks.py
```
```bash
cd notebooks && jupyter nbconvert --to notebook --execute --inplace Project1_Supervised_KDrama.ipynb Project2_Unsupervised_KDrama.ipynb && cd ..
```
```bash
python reports/build_reports.py
```
```bash
python presentation/build_presentation.py
```

### 3. Start the website

Terminal 1 - Python backend (port 8000):

```bash
python ml/kdrama_data.py
```
```bash
uvicorn app.main:app --app-dir backend --port 8000
```

Terminal 2 - website (port 3000):

```bash
cd frontend && npm install && npm run dev
```

Open <http://localhost:3000>. The site calls `/api/...`, which Next.js forwards to the backend
(`BACKEND_URL`, default `http://127.0.0.1:8000`). If the backend is not running, the pages show an error
message instead of numbers. API documentation: <http://127.0.0.1:8000/docs>.

### Drama catalogue (real posters, cast, images)

The website shows real Korean dramas through the backend. Two providers return the same data shape:

| Provider | Key needed | Used when | Notes |
|---|---|---|---|
| **TVmaze** | no | default | 2,862 Korean dramas in a local index (`backend/data/tvmaze_kdramas.json`), details fetched live. No trailers or vote counts. Licence CC BY-SA 4.0. |
| **TMDB** | yes (free) | a credential is in `backend/.env` | Adds trailers, vote counts, more images. Copy `backend/.env.example` to `backend/.env`. Not tested with a real key yet. |

Rebuild the TVmaze index (about 4 minutes, no key):

```bash
python backend/scripts/build_tvmaze_index.py
```

The key (if any) is read only by the Python backend and never reaches the browser. Without any provider the
catalogue pages show a "not connected" message, never placeholder dramas. The AI Predictor and AI Insights pages
do not depend on the catalogue.

Pages: `/` (featured dramas and collections), `/explore` (search, genre and year filters, sorting, infinite
scroll), `/drama/[id]` (details, cast, gallery with lightbox, trailer when available, "Will You Love This Drama?"
predictor), `/watchlist` (saved in the browser's local storage), `/predictor`, `/insights`, `/about`.

### 4. Checks

```bash
pytest backend/tests -q
```
```bash
python qa/check_consistency.py
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | which artifacts are loaded |
| `GET /api/model` | input schema, training ranges, real CV and test metrics |
| `POST /api/predict` | runs the saved pipeline on one record: probability, class, threshold, what-if explanation |
| `GET /api/dramas?q=` | search the real drama catalogue |
| `GET /api/insights` | all Project 2 results (clusters, projections, anomalies) |
| `GET /api/summary` | headline numbers for the Home and About pages |
| `GET /api/catalog/status` | which catalogue provider is active (TMDB or TVmaze) |
| `GET /api/catalog/home` | hero dramas and homepage collections (each with its real criterion) |
| `GET /api/catalog/discover` | catalogue: `q`, `genre`, `year`, `sort`, `page` |
| `GET /api/catalog/drama/{id}` | details, cast, trailer, images, related dramas, match with the training dataset |

## Deployment (Vercel)

The site is one Vercel project with two services defined in `vercel.json`: the Next.js frontend (`frontend/`)
and the FastAPI backend (`backend/`). Requests to `/api/*` go to the backend, everything else to the frontend.
The backend folder is self-contained: `backend/data/ml/` holds a copy of the trained model, the result files and
the drama table. After retraining, refresh that copy and redeploy:

```bash
python backend/scripts/bundle_artifacts.py
```
```bash
npx vercel@latest deploy --prod
```

The GitHub repository is connected to the Vercel project, so a push to `main` also triggers a deployment.

## Reproducibility

`random_state = 42` everywhere. Re-running the scripts gives the same results (Project 1 byte-identical;
Project 2 identical labels and coordinates). Package versions are pinned in `requirements.txt`.
Colab uses its own library versions, so tiny numerical differences are possible there.

## Before you submit

* Ask the instructor to approve the adaptation "movie -> K-Drama" (the worksheet requires approval for adapted tasks).
* Read `docs/REQUIREMENTS_CHECKLIST.md`, `docs/TEAM_ROLES.md`, `docs/PRESENTATION_SCRIPT.md`, `docs/DEFENSE_QA.md`.
