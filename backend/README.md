# ORCA Backend — Smart India Hackathon 2026 (ISRO, SIH26176)

Multi-agent marine intelligence platform: conversational query → planning/orchestrator → marine/weather/risk (Groq `openai/gpt-oss-20b`) → ocean analytics/geospatial (deterministic `pandas`/`numpy`/`shapely`) → reporting/visualization → `FinalResponse` with live reasoning trace + `MapPayload`. Backs 9 frontend pages: Home, Chat + Map, Zones Explorer, Weather, Route Planner, Alerts, Analytics, Download, History.

## Quick Start (Demo-Ready — No Live API Required)

The backend runs fully in **demo mode** with cached golden fixtures when `GROQ_API_KEY` is unset. This is the judged hackathon path — never depends on live external APIs during presentation.

**One-time setup:** run `git config core.hooksPath .githooks` once per clone to activate the `.githooks/pre-commit` secret scanner.

Every file under `backend/` imports with the absolute prefix `backend.xxx` (e.g. `from backend.schemas.contracts import ...`) — that's the convention used consistently across the codebase. That means **uvicorn must be started from the project root** (the directory *containing* `backend/`, not from inside it) — `backend` needs to be importable as a top-level package.

```bash
# 1. Set up the venv — this part happens inside backend/
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # fastapi, uvicorn, pydantic, python-dotenv, groq, httpx,
                                   # copernicusmarine, xarray, numpy, pandas, shapely, pytest, reportlab,
                                   # python-docx, scikit-learn, joblib, sqlalchemy, asyncpg, alembic

# 2. Go back to the project root before starting the server — required, see above
cd ..
# no .env needed for demo — /zones, /weather, /route, /alerts, /analytics/historical, /export
# and /query/demo all work with zero configuration (without Copernicus credentials, /zones
# returns restricted areas only — no PFZ zones; see "PFZ zone detection" below). History logging also degrades to a no-op
# without DATABASE_URL (see Persistence Layer below) — it just won't record anything.
uvicorn backend.main:app --reload --port 8000
```

**⚠️ Common mistake:** running `cd backend && uvicorn main:app --reload` (i.e. staying inside `backend/` and dropping the `backend.` prefix) fails with `ModuleNotFoundError: No module named 'backend'` — `main.py`'s own imports (`from backend.agents.deterministic...`) need `backend` on the path as a package, which only happens when the working directory is the project root. Always run `uvicorn backend.main:app`, from one level up.

Frontend (separate terminal):

```bash
cd frontend
npm ci
npm run dev   # http://localhost:5173, proxies /api → localhost:8000
npm run build # production check — must pass for demo
```

## Environment

Copy `.env.example` to `.env` and fill in whichever of these you want live (every one of them degrades gracefully when unset — see the comments in `.env.example` for exactly what each does and doesn't affect). Values left as the `your-...-here` placeholders are treated as unset.

| Variable | Used by | If unset |
|---|---|---|
| `GROQ_API_KEY` | 4 reasoning agents (Planning, Marine, Weather, Risk) | `/query/full` falls back to fixture demo; `/query/demo` is unaffected either way |
| `PROTECTED_PLANET_API_KEY` | one-time `backend/scripts/fetch_mpa_boundaries.py` refresh | geospatial.py reads its local GeoJSON cache instead, or hardcoded fallback boundaries |
| `BHASHINI_USER_ID` / `BHASHINI_ULCA_API_KEY` | language_agent.py translation | language is still detected locally; pipeline runs in English end-to-end |
| `COPERNICUSMARINE_USERNAME` / `COPERNICUSMARINE_PASSWORD` | marine_data_agent.py live chlorophyll lookup, and the shared live PFZ front-detection cache (`pfz_service.py`) read by **chat, Zones Explorer, `/route`, `/alerts`, `/weather/forecast`, and `/export`** | chlorophyll is `None` (chat's marine result degrades to `partial`); no PFZ zones anywhere — `/zones` returns restricted areas only, and chat's marine agent returns `status="error"`. No mock/sample fallback |
| `DATABASE_URL` | history persistence layer (`backend/history/`) | history logging is a silent no-op; the other 8 pages are unaffected |
| `ORCA_API_KEY` | shared-secret auth (`auth.py`) — every request except `GET /health` must send a matching `X-API-Key` | every route stays open |
| `RATE_LIMIT_ENABLED` / `RATE_LIMIT_PER_MINUTE` | per-client-IP rate limiting (`rate_limit.py`) | enabled by default with a generous limit |
| `MAX_REQUEST_BODY_BYTES` | request body size cap (`main.py`) | 2 MiB |

`POST /query/demo` always uses fixtures (no key needed). `POST /query/full` uses live planning when `GROQ_API_KEY` is set, otherwise falls back to fixture demo.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Raw `EvidenceBundle` (planning → marine → weather → risk) |
| `POST` | `/query/full` | Full pipeline `FinalResponse` (deterministic + reporting + visualization) |
| `POST` | `/query/demo` | **Demo endpoint** — fixture-based `FinalResponse`, no LLM call |
| `GET` | `/zones` | Zones Explorer — real satellite-front-detected PFZ catalog (an anchor with no live data contributes no zones — no mock fallback) + real restricted-area boundaries; the same shared cache chat, `/route`, `/alerts`, `/weather/forecast`, and `/export` all read |
| `GET` | `/weather` | Live current weather/marine snapshot for a lat/lon |
| `GET` | `/weather/forecast` | 7-day ML-predicted wave height/wind speed for a zone |
| `POST` | `/route` | A* route between a start point and a destination/zone |
| `GET` | `/alerts` | Active cyclone/lightning advisories across tracked PFZ zones |
| `GET` | `/analytics/historical` | Historical multi-variable time series for a lat/lon + date range |
| `GET` | `/export` | Historical data as `csv` / `json` / `pdf` / `docx` |
| `GET` | `/history` | Paginated history list — filter by `page_source` + date range |
| `GET` | `/history/{id}` | Single history record detail (raw request/response payload) |
| `GET` | `/health` | Health check |
| `GET` | `/fixtures` | All 4 golden fixtures |

```bash
curl -X POST http://127.0.0.1:8000/query/demo \
  -H "Content-Type: application/json" \
  -d '{"query": "is it safe to go out tomorrow near Chennai"}'

curl -X POST http://127.0.0.1:8000/query/demo \
  -H "Content-Type: application/json" \
  -d '{"query": "can I fish near Gulf of Mannar"}'
```

## Multi-turn sessions (`session_id`)

`/query` and `/query/full` accept an optional `session_id` in the request body. Omit it (or pass `null`) on the first turn — the response carries the active session in the **`X-Session-Id` response header** (the JSON body shape is unchanged, so this is additive). Pass that same value back as `session_id` on the next turn to continue the conversation.

With a live session, the Planning Agent resolves ambiguous follow-ups (no place name in the query — e.g. "what about Friday?") against the session's most recently extracted location, instead of failing to find one. Sessions live in-memory per backend process and expire after 30 minutes of inactivity.

```bash
# Turn 1 — no session_id, capture X-Session-Id from the response headers
curl -i -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "is it safe to go out tomorrow near Chennai"}'

# Turn 2 — reuse that session_id; "what about Friday?" has no location of its own,
# so it reuses Chennai from turn 1
curl -i -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what about Friday?", "session_id": "<value from X-Session-Id above>"}'
```

## Persistence layer — history (Phase 8)

Every request to the 7 page-facing data endpoints (`/query*` → `chat`, `/zones`, `/weather`, `/weather/forecast` → `weather`, `/route`, `/alerts`, `/analytics/historical` → `analytics`, `/export` → `download`) is logged into a Postgres `history` table by `HistoryLoggingMiddleware` — one ASGI middleware, no per-endpoint code, no changes to the 9 core agents.

```bash
# One-time setup, from inside backend/ (after activating .venv and installing requirements.txt):
createdb orca_dev
cp .env.example .env   # then set DATABASE_URL=postgresql+asyncpg://<you>@localhost:5432/orca_dev
alembic upgrade head
```

- Schema / migration: `backend/history/models.py` + `backend/alembic/versions/`. Additive and reversible (`alembic downgrade -1` cleanly drops the table).
- Read/write logic: `backend/history/service.py`. Binary export responses (pdf/docx) are logged as `{content_type, size_bytes}` metadata, not raw bytes.
- Without `DATABASE_URL` set, all of the above is a silent no-op — `GET /history` returns an empty list and none of the other 8 pages are affected.
- Tests: `backend/tests/test_history.py`, run against a separate `orca_test` database (see `backend/tests/conftest.py`).

## PFZ zone detection — live satellite fronts (shared: chat + Zones Explorer + `/route` + `/alerts` + `/weather/forecast` + `/export`)

Chat and `GET /zones` both serve Potential Fishing Zones computed from real satellite data instead of a fixed heuristic — the same class of remote-sensing method (SST + chlorophyll "front" detection) real PFZ advisory systems are built on, since INCOIS itself has no queryable API to consume directly (see `marine_data_agent.py`'s module docstring). They read the **same shared live cache**, not two separate implementations — `/route`, `/alerts`, `/weather/forecast`, and `/export` all resolve zone ids through `GET /zones`'s own catalog, so they inherit the same live data transitively, with no separate fetch of their own.

**Method** (`backend/services/pfz_service.py`):
1. Fetch a real SST grid (Copernicus OSTIA L4, `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2`) and chlorophyll grid (Copernicus L4 gapfree, same dataset the chat flow's live chlorophyll uses) over a box around each of the 11 anchor cities.
2. Align both to one grid (they have different native resolutions), compute the gradient magnitude of each — a "front" is wherever a value changes sharply, indicating a nutrient-rich boundary.
3. Score every cell by `(SST front strength) × (chlorophyll front strength) × (SST favorability) × (chlorophyll favorability)`, reusing `analytics.py`'s existing optimal-range thresholds — a cell only ranks highly if a real front *and* biologically favorable absolute conditions co-occur there.
4. Pick up to 3 distinct zones per anchor via greedy non-max suppression (≥15km apart, so one strong front doesn't produce 3 near-duplicate picks).

**Per-anchor resilience:** concurrency is capped at 4 simultaneous anchors (`_MAX_CONCURRENT_PFZ_ANCHORS`) — Copernicus's service was measured to fail *every* request once concurrency reached 6 anchors (12 connections) at once. If live data still isn't available for a specific anchor (credentials unset, a fetch failure, or Copernicus under heavier load than usual), that anchor contributes **zero zones** — there is no mock/sample fallback, so missing data shows up as genuinely missing rather than fabricated. With no Copernicus credentials at all, every anchor is empty and the catalog holds restricted areas only. Cached for 3 hours (`_ZONES_CACHE_TTL_SECONDS`, in `pfz_service.py` — see `get_cached_zones()`) since the satellite data itself only refreshes about once a day, and pre-warmed in the background at app startup (`main.py`'s lifespan, `asyncio.create_task`, not awaited) so the first real request — chat or `/zones` — never pays the cold-cache cost. If more than 25% of anchors come back empty, the result is cached for only 5 minutes instead (`_ZONES_CACHE_DEGRADED_TTL_SECONDS`), so a transient outage recovers quickly. If pre-warm itself fails (no credentials, a network issue), that's logged only; the next request recomputes the cache itself. Single-flight: an `asyncio.Lock` (`_zones_cache_lock`) ensures that if the TTL does expire while several requests land at once, only the first actually recomputes — everyone else waiting on the lock gets that same fresh result rather than each independently re-triggering the full 11-anchor Copernicus fetch.

**Chat's own lookup — nearest-anchor, not a fresh computation at the query point:** chat (`marine_data_agent.run_marine_data_agent` → `_live_pfz_candidate_zones` → `pfz_service.get_nearest_anchor_zones`) resolves the query location to whichever of the 11 anchor regions above is closest by haversine distance, and reads *that anchor's* cached zones — recomputing `distance_km` from the real query point, but the zone *locations* themselves are still that anchor's cached regional result, not a fresh front-detection run at the exact coordinates asked about. Because the anchor grid is sparse (anchors sit ~277–1166km apart from their nearest neighbor), a query far from every anchor still resolves to the nearest one rather than erroring — but when that anchor is more than 75km away (`_NEARBY_ANCHOR_THRESHOLD_KM`), chat says so plainly instead of presenting a distant regional match as an ordinary nearby result:

> Nearest known fishing-zone data: &lt;ANCHOR&gt;-PFZ-001 (&lt;distance&gt; km away) — this is the closest live data available, but it's well outside typical local range; treat it as a regional reference, not a nearby recommendation

(vs. the normal-case phrasing, `Nearest fishing zone: <ANCHOR>-PFZ-001 (<distance> km away)`, when the resolved anchor is within threshold — see `reporting.py`'s `_format_geospatial`/`_nearest_zone_is_distant`.) On any live-lookup failure, timeout (`_LIVE_PFZ_LOOKUP_TIMEOUT`, 10s), or empty live result, chat gets no PFZ zones: the Marine Data agent returns `status="error"` ("no live PFZ zones available for this location") and the response is built from the remaining evidence — it never substitutes mock zones, and never crashes the request.

**Honest caveats, inherent to this method (not bugs):**
- **~1 day lag** — "live" means the most recent satellite pass, not real-time-this-second.
- **Cloud/land gaps are real** — some grid cells legitimately have no data on a given day; land is correctly masked as NaN and never selected.
- **A proxy, not a fish detector** — SST/chlorophyll fronts indicate favorable *conditions* for fish aggregation; no satellite system (INCOIS included) can see actual fish.
- **Algorithmically-chosen locations, not an official INCOIS bulletin** — the zone centers are real front-detection output, not a reproduction of INCOIS's own published advisory zones (which aren't available as an API to compare against).
- **Chat's result is the nearest live-detected *region*, not a fresh computation at the exact query point** — see "Chat's own lookup" above; `is_distant` is how the response stays honest about it.

## Real MPA boundary data (fetch-and-cache)

`geospatial.py`'s restricted-zone check runs against a local cache of real MPA boundary
polygons — `backend/agents/deterministic/data/mpa_boundaries.geojson` — fetched from the
[Protected Planet (WDPA) API](https://api.protectedplanet.net/). The cache is read once per
process at query time; the API is **never** called live in the request path.

Refresh it with:

```bash
# requires PROTECTED_PLANET_API_KEY in backend/.env — free key at
# https://api.protectedplanet.net/request
python -m backend.scripts.fetch_mpa_boundaries
```

If the cache file is missing, empty, or fails to parse, `geospatial.py` falls back to a small
set of hardcoded boundaries (Gulf of Mannar / Palk Bay) and logs a warning — restricted-zone
checks never go dark just because the cache wasn't refreshed. Real cached polygons are buffered
outward by ~15km at load time (`MPA_BUFFER_KM` in `geospatial.py`) since WDPA digitizes some
Indian MPAs as tight island/reef clusters rather than the full park extent, and others as a
bare representative point with no boundary at all.

## Known limitations (accepted, not defects)

- **Chat's PFZ zones are the cached nearest-anchor region's result, not a fresh computation at the exact query point** — see "Chat's own lookup" in "PFZ zone detection" above. `distance_km` is always recomputed from the real query point, but the zone locations themselves are that anchor's cached result; the `is_distant` flag (75km threshold) is what keeps the response honest when the nearest anchor is genuinely far. INCOIS's own advisory zones remain unavailable as an API either way, so no path reproduces INCOIS's exact published zones.
- **Tide and UV index are not modeled** — dropped from scope; `weather_service.py`/`analytics_service.py` do not surface them.
- **7-day wave/wind forecast models** (`backend/models/forecast_day1-7.pkl`) are trained artifacts, committed to the repo — regenerate with `python -m backend.scripts.train_forecast_models` if you need to retrain them.

## Project Structure

```
backend/
  main.py                          # composition root only: app/CORS/history-middleware setup,
                                    # startup PFZ cache pre-warm, GET /health — endpoints themselves
                                    # live in routers/ (P4 codebase-audit split, no behavior change)
  routers/
    chat.py     # POST /query, /query/full, /query/demo + GET /fixtures
    pages.py    # GET /zones, /weather, /weather/forecast, /alerts, /analytics/historical,
                # /export, POST /route — the 6-non-chat-page backbone
    history.py  # GET /history, /history/{id}
  history/                         # Phase 8 — persistence layer (models, db session, middleware, service, schemas)
  alembic/                         # migrations for the history table (additive + reversible)
  schemas/contracts.py             # Frozen Pydantic contracts (single source of truth)
  schemas/test_fixtures.py         # 4 golden EvidenceBundles
  schemas/demo_snapshot.py         # live-mode fallback system (Tier 2 demo-reliability snapshots)
  agents/reasoning/                # Groq-backed: planning, marine_data, weather, risk_assessment,
                                    # navigation, user_interaction (session memory), language
  agents/deterministic/
    analytics.py   # SST/chlorophyll thresholds (pandas/numpy) + TraceStep
    geospatial.py  # haversine, nearest PFZ, real MPA geofence (shapely) + TraceStep
    reporting.py   # merges all evidence → FinalResponse (graceful fallback) + TraceStep
    visualization.py # pins/overlays, (lat,lon)→GeoJSON (lon,lat) + TraceStep
    data/mpa_boundaries.geojson # cached real WDPA MPA polygons (see below)
  services/                        # alerts, analytics, weather, forecast, export-render — shared
                                    # data-fetch/rendering backbone for the 6 non-chat pages
                                    # pfz_service.py — live SST/chlorophyll front detection + the
                                    # shared cache chat, Zones Explorer, /route, /alerts,
                                    # /weather/forecast, /export all read (see dedicated section above)
  auth.py / rate_limit.py          # optional X-API-Key auth + per-IP rate limiting middlewares
  models/                          # trained forecast .pkl artifacts (committed, regenerable)
  scripts/
    fetch_mpa_boundaries.py        # one-time/periodic refresh of the MPA cache above
    train_forecast_models.py       # trains the 7-day wave/wind forecast models
    validate_forecast_models.py    # checks the trained .pkl files load and match the training features
frontend/
  src/App.tsx            # react-router routes for all 9 pages
  src/components/
    Layout.tsx           # nav bar + <Outlet/> — shared shell for every page
    ChatPanel.tsx         # conversation view, example queries
    MapView.tsx           # Leaflet: PFZ pins + restricted-area polygons
    TracePanel.tsx        # judged live trace — agent, input, output, timestamp
  src/pages/              # Home, ChatMap, ZonesExplorer, Weather, RoutePlanner,
                           # Alerts, Analytics, Download, History
```

Trace requirement: every deterministic function appends a `TraceStep` to `bundle.trace` before returning.

## Testing

Run these from the **project root** too, for the same reason as above. The backend suite needs a migrated `orca_test` database (see `backend/tests/conftest.py`) — one-time: `createdb orca_test`, then from `backend/`: `DATABASE_URL=postgresql+asyncpg://$USER@localhost:5432/orca_test alembic upgrade head`.

```bash
# backend — full suite: 9 core agents + history layer + PFZ front
# detection/cache + chat's live wiring + startup pre-warm. External APIs are
# mocked (PFZ endpoint tests use a synthetic satellite grid — see
# conftest.py's offline_pfz_data). test_live_smoke.py is the exception: a
# gated live end-to-end test that only runs with a real GROQ_API_KEY,
# otherwise skips instantly
source backend/.venv/bin/activate
PYTHONPATH="$(pwd)" pytest backend/tests/ -q
# via HTTP
curl -s http://127.0.0.1:8000/health | jq
# frontend — component/page tests (Vitest + React Testing Library),
# separate from the build check below
cd frontend && npm test
npm run build
```
