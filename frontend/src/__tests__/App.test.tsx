import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'

// Every route but Home is React.lazy() (see App.tsx) — this is the
// regression test for that split (added P4, this session): confirms
// every one of the 9 routes actually resolves to its real page component
// through BrowserRouter + Suspense, not just that the lazy() call
// compiles. MapView (Leaflet) is stubbed for the same reason
// ChatMapPage.test.tsx stubs it — jsdom can't run a real Leaflet map, and
// that's not what this test is checking.
vi.mock('../components/MapView', () => ({
  MapView: () => <div data-testid="map-view-stub" />,
}))

// Generic response every page's on-mount fetch(es) can consume without
// crashing — most pages populate a zone dropdown from GET /api/zones on
// load; an empty catalog is a valid, real response shape (e.g. right
// after a fresh deploy before the cache has ever populated).
function stubFetchForEveryPage() {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => ({ zones: [], generated_at: '2026-01-01T00:00:00Z', alerts: [], checked_zones: 0, items: [], total: 0, limit: 10, offset: 0 }),
      text: async () => '{}',
    }),
  )
}

function navigateTo(path: string) {
  window.history.pushState({}, '', path)
}

beforeEach(() => {
  stubFetchForEveryPage()
})

afterEach(() => {
  vi.unstubAllGlobals()
  window.history.pushState({}, '', '/')
})

describe('App routing — every route resolves to its real page', () => {
  const routeCases: [path: string, heading: string, level?: number][] = [
    ['/', 'ORCA'],
    ['/chat', 'ORCA Chat'],
    ['/zones', 'Zones Explorer · PFZ & Restricted Areas'],
    ['/weather', 'Weather', 1], // level 1: the WeatherStatsCard also defaults its own <h3> title to "Weather" before a zone is picked
    ['/route', 'Route Planner · Hazard-Avoiding A*'],
    ['/alerts', 'Alerts & Advisories'],
    ['/analytics', 'Analytics Dashboard'],
    ['/download', 'Download'],
    ['/history', 'History'],
  ]

  it.each(routeCases)('%s renders the real (lazy-loaded) page, not a loading state stuck forever', async (path, heading, level) => {
    navigateTo(path)
    render(<App />)

    // getByRole('heading', ...), not getByText: "ORCA" (and similar short
    // titles) also appear in the nav bar's own brand link, which is not a
    // heading — this targets the page's actual <h1>/<h2> unambiguously.
    await waitFor(
      () => expect(screen.getByRole('heading', { name: heading, ...(level ? { level } : {}) })).toBeInTheDocument(),
      { timeout: 3000 },
    )
  })

  it('an unknown path falls back to Home instead of a blank/broken page', async () => {
    navigateTo('/this-route-does-not-exist')
    render(<App />)

    await waitFor(() => expect(screen.getByRole('heading', { name: 'ORCA' })).toBeInTheDocument())
  })

  it('the nav bar stays visible and interactive while a lazy page chunk is still resolving', async () => {
    navigateTo('/analytics')
    render(<App />)

    // The nav itself is NOT lazy (only page content is — see Layout.tsx's
    // Suspense boundary wrapping just <Outlet/>) — it must be in the DOM
    // immediately, not wait for the page's own chunk/fetch to resolve.
    expect(screen.getByRole('link', { name: /Zones Explorer/ })).toBeInTheDocument()

    await waitFor(() => expect(screen.getByText('Analytics Dashboard')).toBeInTheDocument())
  })
})
