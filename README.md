# ORCA — Marine EcoSystem Reasoning with Collaborative Agents

A multi-agent marine intelligence platform: ask one question about the sea, in your own language, and get a single evidence-backed answer with a map, a safety verdict and a live agent trace.

[![CI](https://github.com/Varunkumar-07/orca-sih2026/actions/workflows/ci.yml/badge.svg)](https://github.com/Varunkumar-07/orca-sih2026/actions/workflows/ci.yml)

Built for Smart India Hackathon 2026

Problem Statement **SIH26176** · ISRO · Theme: Disaster Management · Category: Software

---

## Problem

People who work at sea have to answer one question every day: is it safe and productive to go out at this location right now? The answer is scattered across separate sources: fishing-zone advisories, weather and cyclone warnings, marine-protected-area boundaries and satellite ocean data. None of them is combined into one plain-language answer. SIH26176 asks for an agentic, multi-agent AI system that correlates these sources into one answer.

## What ORCA does

- **Conversational query pipeline.** Accepts a natural-language question and returns one merged answer with a risk verdict, map pins and overlays, and a step-by-step trace of every agent's input and output.
- **Multilingual.** Detects the query's script locally (Hindi, Bengali, Punjabi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Urdu). It translates the query to English and the answer back using Bhashini.
- **Session memory.** Follow-up questions like "what about Friday?" reuse the last location for 30 minutes.
- **Potential Fishing Zones (PFZ) from live satellite data.** Detects SST and chlorophyll fronts in Copernicus Marine grids around 11 coastal anchor cities, returning up to 3 zones per anchor.
- **Restricted-zone geofencing.** Checks locations against real Marine Protected Area polygons from WDPA (Protected Planet), cached locally. A location inside a protected area always gets a PROHIBITED verdict.
- **Hazard-avoiding routes.** Plans a route with A* search over a grid, treating restricted zones as impassable.
- **Weather and 7-day forecast.** Shows current wind, waves and SST from Open-Meteo. Its own RandomForest models forecast wave height and wind for the next 7 days.
- **Alerts.** Raises cyclone and thunderstorm advisories for every tracked fishing zone.
- **Historical analytics and export.** Provides multi-variable historical time series, downloadable as CSV, JSON, PDF or DOCX.
- **History.** Logs every lookup across the platform to PostgreSQL, browsable on its own page.
- **Degrades gracefully.** Every external key is optional. A missing key or failed feed gives a partial answer that says so, not a crash. `POST /query/demo` runs fully offline from built-in fixtures.

The frontend has 9 pages: Home, Chat + Map, Zones Explorer, Weather, Route Planner, Alerts, Analytics, Download and History.

## Architecture

```mermaid
flowchart LR
    subgraph FE["Frontend — React + Vite (9 pages)"]
        UI["Chat + Map · Zones · Weather · Route<br/>Alerts · Analytics · Download · History"]
    end

    subgraph BE["Backend — FastAPI"]
        MW["Middleware<br/>history logging · rate limit · optional API key"]

        subgraph CHAT["Chat pipeline (/query*)"]
            LANG1["Language Agent<br/>detect + translate in"]
            PLAN["Planning Agent<br/>intent + location"]
            UIA["User Interaction Agent<br/>session memory"]
            MARINE["Marine Data Agent"]
            WEATHER["Weather Agent"]
            RISK["Risk Assessment Agent"]
            DET["Analytics · Geospatial<br/>Navigation (A*)"]
            REP["Reporting · Visualization"]
            LANG2["Language Agent<br/>translate out"]
            LANG1 --> PLAN --> MARINE --> WEATHER --> RISK --> DET --> REP --> LANG2
            PLAN <--> UIA
        end

        subgraph SVC["Page services"]
            PFZ["pfz_service<br/>front detection + 3h cache"]
            WS["weather · forecast (ML)"]
            AL["alerts · analytics · export"]
        end
    end

    subgraph EXT["Data sources"]
        GROQ["Groq LLM"]
        CMEMS["Copernicus Marine<br/>SST + chlorophyll"]
        OM["Open-Meteo<br/>forecast · marine · archive"]
        WDPA["WDPA MPA polygons<br/>(local GeoJSON cache)"]
        BH["Bhashini"]
        PG[("PostgreSQL<br/>history")]
    end

    UI -- "/api/*" --> MW
    MW --> CHAT
    MW --> SVC
    MW --> PG
    PLAN & MARINE & WEATHER & RISK -.-> GROQ
    LANG1 & LANG2 -.-> BH
    MARINE --> PFZ
    PFZ -.-> CMEMS
    WEATHER & WS & AL -.-> OM
    DET -.-> WDPA
```

The LLM (Groq, `openai/gpt-oss-20b`) handles only judgment calls: intent, location and risk reasoning. Geofencing, thresholds, PFZ scoring, A* routing and the forecast models are deterministic code.

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic |
| AI / LLM | Groq (`openai/gpt-oss-20b`) |
| Data and geo | NumPy, pandas, xarray, Shapely, `copernicusmarine`, httpx |
| ML | scikit-learn (RandomForestRegressor), joblib |
| Persistence | PostgreSQL, SQLAlchemy (async) + asyncpg, Alembic |
| Reports | reportlab (PDF), python-docx (DOCX) |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, React Router, Leaflet / react-leaflet, Recharts |
| Testing | pytest, Vitest + React Testing Library, oxlint |
| CI / deploy | GitHub Actions, Render (`render.yaml`) |

## Local setup

**Prerequisites:** Python 3.12, Node.js 22, PostgreSQL. PostgreSQL is only needed for the History page and the backend tests.

### Backend

The backend imports itself as the `backend` package, so **run the server from the repo root**, not from inside `backend/`.

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `backend/.env`:
- Set `DATABASE_URL` to your local Postgres user and database.
- Delete every line you don't have a real value for. All keys are optional, and a leftover placeholder for the Copernicus credentials will be sent as a real login attempt.

Then create the database, apply migrations and start the server:

```bash
createdb orca_dev
alembic upgrade head
cd ..
uvicorn backend.main:app --reload --port 8000
```

Check it's running:

```bash
curl http://127.0.0.1:8000/health
```

To try the pipeline with no API keys:

```bash
curl -X POST http://127.0.0.1:8000/query/demo -H "Content-Type: application/json" -d '{"query": "can I fish near Gulf of Mannar"}'
```

### Frontend

Run this in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

The app runs at http://localhost:5173. The Vite dev server proxies `/api/*` to the backend on port 8000.

### Pre-commit secret scanner (recommended)

Run this once per clone to enable the hook in `.githooks/`, which blocks commits containing credential-like files or content:

```bash
git config core.hooksPath .githooks
```

## Environment variables

Full descriptions are in [`backend/.env.example`](backend/.env.example) and [`frontend/.env.example`](frontend/.env.example). Every variable is optional.

**Backend** (`backend/.env`)

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | LLM for the Planning, Marine, Weather and Risk agents. Without it, `/query/full` falls back to the offline demo fixtures. |
| `COPERNICUSMARINE_USERNAME`, `COPERNICUSMARINE_PASSWORD` | Live SST and chlorophyll grids for PFZ detection and chlorophyll readings. Without them, no PFZ zones are returned. |
| `BHASHINI_USER_ID`, `BHASHINI_ULCA_API_KEY` | Translation to and from Indian languages. Without them, language is still detected but the pipeline runs in English. |
| `PROTECTED_PLANET_API_KEY` | Only for `python -m backend.scripts.fetch_mpa_boundaries`, which refreshes the cached MPA polygons. |
| `DATABASE_URL` | PostgreSQL connection string (`postgresql+asyncpg://…`) for History. Without it, history logging is a no-op. |
| `ORCA_API_KEY` | When set, every request except `/health` needs a matching `X-API-Key` header. |
| `RATE_LIMIT_ENABLED`, `RATE_LIMIT_PER_MINUTE` | Per-IP rate limiting, which is on by default. |
| `MAX_REQUEST_BODY_BYTES` | Maximum request body size (default 2 MiB). |

**Frontend** (`frontend/.env`)

| Variable | Purpose |
|---|---|
| `VITE_ORCA_API_KEY` | Sent as `X-API-Key`. Must match the backend's `ORCA_API_KEY`, and is only needed when that is set. |

## Running tests

**Backend.** The tests use a separate `orca_test` database (see `backend/tests/conftest.py`). Create and migrate it once:

```bash
createdb orca_test
cd backend
DATABASE_URL=postgresql+asyncpg://$USER@localhost:5432/orca_test alembic upgrade head
cd ..
```

Then run the suite from the repo root, with the venv active:

```bash
PYTHONPATH=. python -m pytest backend/tests -q
```

The suite runs offline, and external APIs are mocked. `test_live_smoke.py` makes real Groq and Copernicus calls and skips itself unless a real `GROQ_API_KEY` is set.

**Frontend**

```bash
cd frontend
npm run lint
npm test
npm run build
```

**CI.** [`ci.yml`](.github/workflows/ci.yml) runs both suites on every push and pull request. [`live-smoke.yml`](.github/workflows/live-smoke.yml) runs the live smoke test daily, and [`models.yml`](.github/workflows/models.yml) retrains and validates the forecast models weekly.

## Project structure

```
.
├── backend/            FastAPI app
│   ├── agents/         reasoning agents (LLM) + deterministic modules
│   ├── services/       PFZ detection, weather, forecast, alerts, analytics, export
│   ├── routers/        HTTP endpoints (chat, pages, history)
│   ├── history/        PostgreSQL history logging
│   ├── alembic/        database migrations
│   ├── models/         trained forecast models (.pkl)
│   ├── scripts/        MPA boundary fetch, model training/validation
│   └── tests/          pytest suite
├── frontend/           React + Vite app (9 pages)
├── .github/workflows/  CI, live smoke test, model retraining
├── .githooks/          pre-commit secret scanner
└── render.yaml         Render deployment blueprint
```

Endpoint reference and implementation details: [`backend/README.md`](backend/README.md).

## Team

**Team NEXORA** · Team ID: 159223

| Name | Role | GitHub |
|---|---|---|
| Kinshue Priya M | Frontend | [@Kinshuepriya](https://github.com/Kinshuepriya) |
| Varun Kumar U | Backend & ML | [@Varunkumar-07](https://github.com/Varunkumar-07) |
| Mohammed Bilal Sharief | Backend & ML | [@MOHAMMEDBILAL-007](https://github.com/MOHAMMEDBILAL-007) |
| Vedha V | Frontend | [@ved-24-2006](https://github.com/ved-24-2006) |
| Mk Reddy Venkata Santhosh | Research | [@SANTHOSHMK07](https://github.com/SANTHOSHMK07) |
| Rakesh Patil | Design | [@Raku770](https://github.com/Raku770) |
