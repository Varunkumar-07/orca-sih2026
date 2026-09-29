import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { RoutePlannerPage } from '../RoutePlannerPage'
import type { RouteResponse, ZoneRecord } from '../../types'

// RoutePlannerPage renders a real Leaflet map via MapView — same convention
// as ChatMapPage.test.tsx: stub it out so this file exercises the page's own
// state/fetch wiring, not Leaflet internals (which have no tests to
// duplicate here, and no ResizeObserver polyfill is set up in this file).
vi.mock('../../components/MapView', () => ({
  MapView: () => <div data-testid="map-view-stub" />,
}))

const ZONES: ZoneRecord[] = [
  { id: 'KOCHI-PFZ-001', name: 'PFZ-001', type: 'pfz', near: 'Kochi', coordinates: { lat: 9.93, lon: 76.27 } },
  { id: 'MPA-001', name: 'MPA Reserve', type: 'restricted', geometry: { type: 'Polygon', coordinates: [[[80, 13], [81, 13], [81, 14], [80, 13]]] } },
]

function jsonResponse(body: unknown, ok = true) {
  return { ok, status: ok ? 200 : 500, statusText: ok ? 'OK' : 'Error', json: async () => body }
}

function routeResponse(overrides: Partial<RouteResponse> = {}): RouteResponse {
  return { route: [{ lat: 13.08, lon: 80.27 }, { lat: 9.93, lon: 76.27 }], distance_km: 42.5, waypoint_count: 2, reason: null, ...overrides }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

async function waitForZonesLoaded() {
  await waitFor(() => expect(screen.getByRole('option', { name: /PFZ-001 — near Kochi/ })).toBeInTheDocument())
}

describe('RoutePlannerPage — zone loading', () => {
  it('shows an error banner when the zones fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false)))
    render(<RoutePlannerPage />)
    await waitFor(() => expect(screen.getByText(/Couldn't load zones/)).toBeInTheDocument())
  })

  it('counts restricted zones in the header and offers only PFZ zones as destinations', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ zones: ZONES })))
    render(<RoutePlannerPage />)
    await waitForZonesLoaded()

    expect(screen.getByText('1 restricted areas avoided')).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: /MPA Reserve/ })).not.toBeInTheDocument()
  })
})

describe('RoutePlannerPage — planning a route', () => {
  it('Plan Route is disabled until a destination zone is chosen', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ zones: ZONES })))
    const user = userEvent.setup()
    render(<RoutePlannerPage />)
    await waitForZonesLoaded()

    expect(screen.getByRole('button', { name: /Plan Route/ })).toBeDisabled()
    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    expect(screen.getByRole('button', { name: /Plan Route/ })).not.toBeDisabled()
  })

  it('posts start/destination and renders a found route', async () => {
    const fetchMock = vi.fn(async (input: string) => {
      const url = String(input)
      if (url.startsWith('/api/zones')) return jsonResponse({ zones: ZONES })
      if (url.startsWith('/api/route')) return jsonResponse(routeResponse())
      throw new Error(`unhandled: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<RoutePlannerPage />)
    await waitForZonesLoaded()

    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    await user.click(screen.getByRole('button', { name: /Plan Route/ }))

    await waitFor(() => expect(screen.getByText('Route found')).toBeInTheDocument())
    expect(screen.getByText('42.5 km')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/route',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ start: { lat: 13.08, lon: 80.27 }, destination_zone_id: 'KOCHI-PFZ-001' }),
      }),
    )
  })

  it('says when a start on land was moved to the nearest open water', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: string) => {
      const url = String(input)
      if (url.startsWith('/api/zones')) return jsonResponse({ zones: ZONES })
      return jsonResponse(routeResponse({ start_offset_km: 3.53, end_offset_km: 0 }))
    }))
    const user = userEvent.setup()
    render(<RoutePlannerPage />)
    await waitForZonesLoaded()

    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    await user.click(screen.getByRole('button', { name: /Plan Route/ }))

    await waitFor(() => expect(screen.getByText('Route found')).toBeInTheDocument())
    expect(screen.getByText(/sea route begins at the nearest open water, 3.53 km away/)).toBeInTheDocument()
    expect(screen.queryByText(/route ends at the nearest reachable open water/)).not.toBeInTheDocument()
  })

  it('shows the "no route found" reason when the backend reports none', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: string) => {
      const url = String(input)
      if (url.startsWith('/api/zones')) return jsonResponse({ zones: ZONES })
      return jsonResponse(routeResponse({ route: null, distance_km: null, waypoint_count: 0, reason: 'Destination unreachable without crossing a restricted area.' }))
    }))
    const user = userEvent.setup()
    render(<RoutePlannerPage />)
    await waitForZonesLoaded()

    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    await user.click(screen.getByRole('button', { name: /Plan Route/ }))

    await waitFor(() => expect(screen.getByText('No route found')).toBeInTheDocument())
    expect(screen.getByText('Destination unreachable without crossing a restricted area.')).toBeInTheDocument()
  })

  it('shows an error banner when the route request itself fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: string) => {
      const url = String(input)
      if (url.startsWith('/api/zones')) return jsonResponse({ zones: ZONES })
      return jsonResponse(null, false)
    }))
    const user = userEvent.setup()
    render(<RoutePlannerPage />)
    await waitForZonesLoaded()

    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    await user.click(screen.getByRole('button', { name: /Plan Route/ }))

    await waitFor(() => expect(screen.getByText(/Something went wrong on the server/)).toBeInTheDocument())
    expect(screen.queryByText(/500 Error/)).not.toBeInTheDocument()
  })
})

describe('RoutePlannerPage — use my location', () => {
  it('shows an error when geolocation is unavailable in this environment', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ zones: ZONES })))
    const user = userEvent.setup()
    render(<RoutePlannerPage />)
    await waitForZonesLoaded()

    // jsdom has no navigator.geolocation by default — getUserLocation()'s own
    // `!navigator.geolocation` check resolves null gracefully for that case.
    await user.click(screen.getByRole('button', { name: /Use my location/ }))

    await waitFor(() => expect(screen.getByText(/Could not get your location/)).toBeInTheDocument())
  })
})
