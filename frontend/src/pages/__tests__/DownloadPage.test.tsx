import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DownloadPage } from '../DownloadPage'
import type { ZoneRecord } from '../../types'

const ZONES: ZoneRecord[] = [
  { id: 'KOCHI-PFZ-001', name: 'PFZ-001', type: 'pfz', near: 'Kochi', coordinates: { lat: 9.93, lon: 76.27 } },
]

function jsonResponse(body: unknown, ok = true) {
  return { ok, status: ok ? 200 : 500, statusText: ok ? 'OK' : 'Error', json: async () => body }
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' })))
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function waitForZonesLoaded() {
  await waitFor(() => expect(screen.getByRole('option', { name: /PFZ-001 — near Kochi/ })).toBeInTheDocument())
}

describe('DownloadPage — zone loading', () => {
  it('shows a clear error banner when the zones fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false)))
    render(<DownloadPage />)
    await waitFor(() => expect(screen.getByText(/Couldn't load zones/)).toBeInTheDocument())
  })
})

describe('DownloadPage — export gating', () => {
  it('Export is disabled until a zone is picked', async () => {
    render(<DownloadPage />)
    await waitForZonesLoaded()
    expect(screen.getByRole('button', { name: /Export CSV/ })).toBeDisabled()

    const user = userEvent.setup()
    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    expect(screen.getByRole('button', { name: /Export CSV/ })).not.toBeDisabled()
  })

  it('Export is disabled once every variable is cleared', async () => {
    const user = userEvent.setup()
    render(<DownloadPage />)
    await waitForZonesLoaded()
    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')

    await user.click(screen.getByRole('button', { name: 'Clear' }))
    expect(screen.getByText('Select at least one variable.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Export/ })).toBeDisabled()
  })

  it('switching format updates the Export button label', async () => {
    const user = userEvent.setup()
    render(<DownloadPage />)
    await waitForZonesLoaded()

    await user.click(screen.getByRole('button', { name: /json/i }))
    expect(screen.getByRole('button', { name: /Export json/i })).toBeInTheDocument()
  })

  it('clicking Export fetches /api/export and triggers a file download, not a raw navigation', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn((url: unknown) => {
      if (typeof url === 'string' && url.startsWith('/api/export')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          blob: async () => new Blob(['csv,data'], { type: 'text/csv' }),
          headers: { get: (name: string) => (name === 'Content-Disposition' ? 'attachment; filename="orca_export.csv"' : null) },
        })
      }
      return Promise.resolve(jsonResponse({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' }))
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('URL', { ...URL, createObjectURL: vi.fn().mockReturnValue('blob:mock-url'), revokeObjectURL: vi.fn() })

    const clickSpy = vi.fn()
    const originalCreateElement = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = originalCreateElement(tag)
      if (tag === 'a') el.click = clickSpy
      return el
    })

    render(<DownloadPage />)
    await waitForZonesLoaded()
    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    await user.click(screen.getByRole('button', { name: /Export CSV/ }))

    await waitFor(() => expect(clickSpy).toHaveBeenCalled())
    const exportCall = fetchMock.mock.calls.find(([url]) => typeof url === 'string' && url.startsWith('/api/export'))
    expect(exportCall?.[0]).toContain('zone=KOCHI-PFZ-001')
    expect(exportCall?.[0]).toContain('format=csv')

    vi.restoreAllMocks()
  })

  it('shows an error banner instead of a raw navigation when the export request fails', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn((url: unknown) => {
      if (typeof url === 'string' && url.startsWith('/api/export')) {
        return Promise.resolve({ ok: false, status: 500, text: async () => 'export failed' })
      }
      return Promise.resolve(jsonResponse({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' }))
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<DownloadPage />)
    await waitForZonesLoaded()
    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')
    await user.click(screen.getByRole('button', { name: /Export CSV/ }))

    await waitFor(() => expect(screen.getByText(/Export failed/)).toBeInTheDocument())
  })
})
