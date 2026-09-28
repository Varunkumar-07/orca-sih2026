import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { WeatherPage } from '../WeatherPage'
import type { WeatherSnapshot, ZoneRecord } from '../../types'

// ResponsiveContainer (recharts) needs a real element size to lay out a
// chart — jsdom reports 0x0 for everything, so recharts renders nothing
// visible rather than erroring; that's fine here since the trend chart
// itself isn't what's under test — the data-fetching/wiring is.

const ZONES: ZoneRecord[] = [
  { id: 'KOCHI-PFZ-001', name: 'PFZ-001', type: 'pfz', near: 'Kochi', coordinates: { lat: 9.93, lon: 76.27 } },
  { id: 'CHENNAI-PFZ-001', name: 'PFZ-001', type: 'pfz', near: 'Chennai', coordinates: { lat: 13.08, lon: 80.27 } },
]

function weatherSnapshot(overrides: Partial<WeatherSnapshot> = {}): WeatherSnapshot {
  return {
    lat: 9.93,
    lon: 76.27,
    status: 'ok',
    air_temperature_c: 29.5,
    sst_celsius: 28.7,
    wind_kmh: 18.5,
    wave_height_m: 1.2,
    chlorophyll_mg_m3: 0.635,
    precipitation_mm: 0.2,
    wind_max_kmh: 22.0,
    sunrise_hour_ist: 6.15,
    sunset_hour_ist: 18.27,
    cyclone_alert: false,
    lightning_alert: false,
    source_timestamp: '2026-01-01T00:00:00Z',
    ...overrides,
  }
}

function jsonResponse(body: unknown, ok = true) {
  return { ok, status: ok ? 200 : 500, statusText: ok ? 'OK' : 'Error', json: async () => body }
}

function installFetchRouter(handlers: Record<string, () => unknown>) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string) => {
      const url = typeof input === 'string' ? input : String(input)
      for (const [prefix, handler] of Object.entries(handlers)) {
        if (url.startsWith(prefix)) return jsonResponse(handler())
      }
      throw new Error(`unhandled fetch in test: ${url}`)
    }),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

// "PFZ-001 — near Kochi" is a valid <option> in all three <select>s on the
// page (single-zone + both compare selects show the same catalog) — so
// asserting zones have loaded means "at least one", not "the one", match.
async function waitForZonesLoaded() {
  await waitFor(() => expect(screen.getAllByText(/PFZ-001 — near Kochi/).length).toBeGreaterThan(0))
}

describe('WeatherPage — zone list loads on mount', () => {
  it('populates the zone dropdown from GET /api/zones', async () => {
    installFetchRouter({ '/api/zones': () => ({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' }) })
    render(<WeatherPage />)

    await waitForZonesLoaded()
    expect(screen.getAllByText(/PFZ-001 — near Chennai/).length).toBeGreaterThan(0)
  })

  it('shows a clear error banner (not a silent blank page) when the zones fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false)))
    render(<WeatherPage />)

    await waitFor(() => expect(screen.getByText(/Couldn't load zones/)).toBeInTheDocument())
  })
})

describe('WeatherPage — fetching a single zone\'s conditions', () => {
  beforeEach(() => {
    installFetchRouter({
      '/api/zones': () => ({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' }),
      '/api/weather?': () => weatherSnapshot(),
      '/api/analytics/historical': () => ({ series: { wave_height_m: { dates: ['2026-01-01'], values: [1.2], unit: 'm', status: 'ok' } } }),
    })
  })

  it('is disabled until a zone is picked, then fetches and renders live conditions', async () => {
    const user = userEvent.setup()
    render(<WeatherPage />)

    const getWeatherButton = screen.getByRole('button', { name: /Get Weather/ })
    expect(getWeatherButton).toBeDisabled()

    await waitForZonesLoaded()
    const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement
    await user.selectOptions(select, 'KOCHI-PFZ-001')

    expect(getWeatherButton).not.toBeDisabled()
    await user.click(getWeatherButton)

    await waitFor(() => expect(screen.getByText('18.5 km/h')).toBeInTheDocument())
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/weather?lat=9.93&lon=76.27'))
  })

  it('running the ML forecast only becomes available after a snapshot has loaded', async () => {
    const user = userEvent.setup()
    render(<WeatherPage />)

    expect(screen.queryByRole('button', { name: /Run ML Forecast/ })).not.toBeInTheDocument()

    await waitForZonesLoaded()
    await user.selectOptions(screen.getAllByRole('combobox')[0], 'KOCHI-PFZ-001')
    await user.click(screen.getByRole('button', { name: /Get Weather/ }))

    await waitFor(() => expect(screen.getByRole('button', { name: /Run ML Forecast/ })).toBeInTheDocument())
  })

  it('shows a clear error banner when the weather fetch itself fails, without crashing the page', async () => {
    const user = userEvent.setup()
    installFetchRouter({
      '/api/zones': () => ({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' }),
    })
    vi.stubGlobal('fetch', vi.fn(async (input: string) => {
      const url = String(input)
      if (url.startsWith('/api/zones')) return jsonResponse({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' })
      if (url.startsWith('/api/weather?')) return jsonResponse(null, false)
      return jsonResponse({})
    }))
    render(<WeatherPage />)

    await waitForZonesLoaded()
    await user.selectOptions(screen.getAllByRole('combobox')[0], 'KOCHI-PFZ-001')
    await user.click(screen.getByRole('button', { name: /Get Weather/ }))

    await waitFor(() => expect(screen.getByText(/Couldn't fetch weather/)).toBeInTheDocument())
  })
})

describe('WeatherPage — compare two zones', () => {
  it('fetches both zones concurrently and renders each result independently', async () => {
    const user = userEvent.setup()
    installFetchRouter({
      '/api/zones': () => ({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' }),
      '/api/weather?lat=9.93': () => weatherSnapshot({ wind_kmh: 18.5 }),
      '/api/weather?lat=13.08': () => weatherSnapshot({ wind_kmh: 40.0 }),
    })
    render(<WeatherPage />)

    await waitFor(() => expect(screen.getAllByText(/PFZ-001 — near Kochi/).length).toBeGreaterThan(0))
    const selects = screen.getAllByRole('combobox')
    // The first two comboboxes are the "Compare Two Zones" A/B selects —
    // see WeatherPage's own layout (single-zone select is a third,
    // earlier one in DOM order).
    const compareSelects = selects.slice(1)
    await user.selectOptions(compareSelects[0], 'KOCHI-PFZ-001')
    await user.selectOptions(compareSelects[1], 'CHENNAI-PFZ-001')
    await user.click(screen.getByRole('button', { name: /Compare/ }))

    await waitFor(() => expect(screen.getByText('18.5 km/h')).toBeInTheDocument())
    expect(screen.getByText('40 km/h')).toBeInTheDocument()
  })
})
