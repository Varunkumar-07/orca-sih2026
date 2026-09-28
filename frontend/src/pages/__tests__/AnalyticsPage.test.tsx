import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AnalyticsPage } from '../AnalyticsPage'
import type { AnalyticsResponse, ZoneRecord } from '../../types'

const ZONES: ZoneRecord[] = [
  { id: 'KOCHI-PFZ-001', name: 'PFZ-001', type: 'pfz', near: 'Kochi', coordinates: { lat: 9.93, lon: 76.27 } },
]

function jsonResponse(body: unknown, ok = true) {
  return { ok, status: ok ? 200 : 500, statusText: ok ? 'OK' : 'Error', json: async () => body }
}

function analyticsResponse(overrides: Partial<AnalyticsResponse> = {}): AnalyticsResponse {
  return {
    lat: 9.93,
    lon: 76.27,
    start_date: '2025-12-25',
    end_date: '2026-01-01',
    series: {
      sst_celsius: { unit: '°C', dates: ['2026-01-01'], values: [28.5], status: 'ok' },
      wave_direction_deg: { unit: 'deg', dates: ['2026-01-01'], values: [90], status: 'ok' },
      moon_phase: { unit: '', dates: ['2026-01-01'], values: [0.5], status: 'ok', names: ['Full Moon'] },
    },
    ...overrides,
  }
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

async function waitForZonesLoaded() {
  await waitFor(() => expect(screen.getByRole('option', { name: /PFZ-001 — near Kochi/ })).toBeInTheDocument())
}

describe('AnalyticsPage — zone loading', () => {
  it('shows an error banner when the zones fetch fails', async () => {
    installFetchRouter({ '/api/zones': () => { throw new Error('unused') } })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false)))
    render(<AnalyticsPage />)
    await waitFor(() => expect(screen.getByText(/Couldn't load zones/)).toBeInTheDocument())
  })

  it('prompts to pick a zone before any historical fetch happens', async () => {
    installFetchRouter({ '/api/zones': () => ({ zones: ZONES }) })
    render(<AnalyticsPage />)
    await waitForZonesLoaded()
    expect(screen.getByText('Pick a zone to see its historical trends.')).toBeInTheDocument()
  })
})

describe('AnalyticsPage — historical data', () => {
  it('fetches and renders series once a zone is selected: chart card, direction table, moon-phase table', async () => {
    installFetchRouter({
      '/api/zones': () => ({ zones: ZONES }),
      '/api/analytics/historical': () => analyticsResponse(),
    })
    const user = userEvent.setup()
    render(<AnalyticsPage />)
    await waitForZonesLoaded()

    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')

    await waitFor(() => expect(screen.getByText('Sea Surface Temperature')).toBeInTheDocument())
    expect(screen.getByText('Wave Direction')).toBeInTheDocument()
    expect(screen.getByText('compass bearing — shown as a table, not a line chart')).toBeInTheDocument()
    expect(screen.getByText('Moon Phase')).toBeInTheDocument()
    expect(screen.getByText('Full Moon')).toBeInTheDocument()
  })

  it('shows an error banner when the historical fetch fails, without crashing', async () => {
    installFetchRouter({
      '/api/zones': () => ({ zones: ZONES }),
      '/api/analytics/historical': () => { throw new Error('unused') },
    })
    vi.stubGlobal('fetch', vi.fn(async (input: string) => {
      const url = String(input)
      if (url.startsWith('/api/zones')) return jsonResponse({ zones: ZONES })
      if (url.startsWith('/api/analytics/historical')) return jsonResponse(null, false)
      throw new Error(`unhandled: ${url}`)
    }))
    const user = userEvent.setup()
    render(<AnalyticsPage />)
    await waitForZonesLoaded()

    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    await waitFor(() => expect(screen.getByText(/Couldn't load historical data/)).toBeInTheDocument())
  })

  it('switching the date range re-fetches with a wider window', async () => {
    const fetchMock = vi.fn(async (input: string) => {
      const url = String(input)
      if (url.startsWith('/api/zones')) return jsonResponse({ zones: ZONES })
      if (url.startsWith('/api/analytics/historical')) return jsonResponse(analyticsResponse())
      throw new Error(`unhandled: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<AnalyticsPage />)
    await waitForZonesLoaded()
    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    await waitFor(() => expect(screen.getByText('Sea Surface Temperature')).toBeInTheDocument())

    fetchMock.mockClear()
    await user.click(screen.getByRole('button', { name: '30 Days' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/analytics/historical')))
  })
})
