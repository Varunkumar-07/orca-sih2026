import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AlertsPage } from '../AlertsPage'
import type { AlertRecord, AlertsResponse } from '../../types'

function jsonResponse(body: unknown, ok = true) {
  return { ok, status: ok ? 200 : 500, statusText: ok ? 'OK' : 'Error', json: async () => body }
}

function alert(overrides: Partial<AlertRecord> = {}): AlertRecord {
  return {
    zone_id: 'CHENNAI-PFZ-001',
    zone_name: 'PFZ-001',
    near: 'Chennai',
    alert_type: 'cyclone',
    severity: 'high',
    detail: 'Cyclone warning issued for this zone.',
    ...overrides,
  }
}

function alertsResponse(overrides: Partial<AlertsResponse> = {}): AlertsResponse {
  return { alerts: [], checked_zones: 11, generated_at: '2026-01-01T00:00:00Z', ...overrides }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('AlertsPage — loading and empty state', () => {
  it('shows an all-clear message when there are no active alerts', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(alertsResponse())))
    render(<AlertsPage />)

    await waitFor(() => expect(screen.getByText(/No active cyclone or lightning alerts/)).toBeInTheDocument())
    expect(screen.getByText(/All 11 tracked fishing zones/)).toBeInTheDocument()
  })

  it('does not claim all zones are calm when live weather was unavailable for every zone', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(alertsResponse({ unavailable_zones: 11 }))))
    render(<AlertsPage />)

    await waitFor(() => expect(screen.getByText(/alert status is unknown/)).toBeInTheDocument())
    expect(screen.queryByText(/No active cyclone or lightning alerts/)).not.toBeInTheDocument()
    expect(screen.queryByText(/report calm conditions/)).not.toBeInTheDocument()
  })

  it('qualifies the all-clear when only some zones could be checked', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(alertsResponse({ unavailable_zones: 3 }))))
    render(<AlertsPage />)

    await waitFor(() => expect(screen.getByText(/couldn't be checked for 3 of 11 zones/)).toBeInTheDocument())
    expect(screen.getByText(/The 8 zones that could be checked currently report calm conditions/)).toBeInTheDocument()
    expect(screen.queryByText(/All 11 tracked fishing zones/)).not.toBeInTheDocument()
  })

  it('shows an error banner when the fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false)))
    render(<AlertsPage />)

    await waitFor(() => expect(screen.getByText(/Couldn't load alerts/)).toBeInTheDocument())
  })
})

describe('AlertsPage — active alerts', () => {
  it('renders one card per alert plus per-type summary chips', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse(
          alertsResponse({
            alerts: [
              alert({ zone_id: 'A1', alert_type: 'cyclone', severity: 'high' }),
              alert({ zone_id: 'A2', alert_type: 'lightning', severity: 'moderate', zone_name: 'PFZ-002', detail: 'Lightning risk this evening.' }),
            ],
          }),
        ),
      ),
    )
    render(<AlertsPage />)

    await waitFor(() => expect(screen.getByText(/1 cyclone alert/)).toBeInTheDocument())
    expect(screen.getByText(/1 lightning alert/)).toBeInTheDocument()
    expect(screen.getByText('Cyclone warning issued for this zone.')).toBeInTheDocument()
    expect(screen.getByText('Lightning risk this evening.')).toBeInTheDocument()
  })

  it('Refresh re-fetches and can clear the list back to all-clear', async () => {
    const user = userEvent.setup()
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(alertsResponse({ alerts: [alert()] })))
      .mockResolvedValueOnce(jsonResponse(alertsResponse({ alerts: [] })))
    vi.stubGlobal('fetch', fetchMock)
    render(<AlertsPage />)

    await waitFor(() => expect(screen.getByText('Cyclone warning issued for this zone.')).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: /Refresh/ }))

    await waitFor(() => expect(screen.getByText(/No active cyclone or lightning alerts/)).toBeInTheDocument())
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})
