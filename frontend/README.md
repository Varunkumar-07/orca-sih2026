# ORCA Frontend

The web app for ORCA, built with React, TypeScript and Vite. Maps use Leaflet, charts use Recharts, and styling is Tailwind CSS. It talks to the FastAPI backend in `../backend`.

For full project setup (backend, database, environment variables), see the [root README](../README.md).

## Pages

| Route | Page |
|---|---|
| `/` | Home |
| `/chat` | Chat + Map |
| `/zones` | Zones Explorer |
| `/weather` | Weather |
| `/route` | Route Planner |
| `/alerts` | Alerts |
| `/analytics` | Analytics |
| `/download` | Download |
| `/history` | History |

## Running it

Requires Node.js 22 (the version CI uses).

```bash
npm ci
npm run dev
```

The dev server runs at http://localhost:5173. Requests to `/api/*` are proxied to the backend at `http://localhost:8000`, with the `/api` prefix removed (see `vite.config.ts`). Start the backend first.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Start the Vite dev server |
| `npm run lint` | Lint with oxlint |
| `npm test` | Run the Vitest + React Testing Library suite once |
| `npm run test:watch` | Run tests in watch mode |
| `npm run test:coverage` | Run tests with a coverage report |
| `npm run build` | Type-check (`tsc -b`) and build to `dist/` |
| `npm run preview` | Serve the production build locally |

## Environment

| Variable | Purpose |
|---|---|
| `VITE_ORCA_API_KEY` | Optional. Sent as the `X-API-Key` header on `/api/*` requests. Only needed when the backend has `ORCA_API_KEY` set, and it must match that value exactly. Leave it unset for normal local development. |

Copy `.env.example` to `.env` to set it. Vite bakes the value into the build, so rebuild after changing it.
