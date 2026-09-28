import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ZonesExplorerPage } from '../ZonesExplorerPage'
import type { ZoneRecord } from '../../types'

// Same convention as ChatMapPage.test.tsx / RoutePlannerPage.test.tsx: stub
// out MapView (real Leaflet, needs DOM APIs jsdom doesn't provide) so this
// file exercises ZonesExplorerPage's own fetch/selection wiring, while still
// exposing onSelectZone so the list<->map selection sync stays covered.
vi.mock('../../components/MapView', () => ({
  MapView: ({ selectedZone, onSelectZone }: { selectedZone?: string | null; onSelectZone?: (id: string | null) => void }) => (
    <div data-testid="map-view-stub">
      <span data-testid="map-selected-zone">{selectedZone ?? ''}</span>
      <button onClick={() => onSelectZone?.('KOCHI-PFZ-001')}>select-from-map</button>
    </div>
  ),
}))

const ZONES: ZoneRecord[] = [
  { id: 'KOCHI-PFZ-001', name: 'PFZ-001', type: 'pfz', near: 'Kochi', coordinates: { lat: 9.93, lon: 76.27 }, distance_km: 12, sst_celsius: 28.4, chlorophyll_mg_m3: 0.5, advisory: 'Good conditions' },
  { id: 'MPA-001', name: 'Gulf of Mannar', type: 'restricted', geometry: { type: 'Polygon', coordinates: [[[80, 13], [81, 13], [81, 14], [80, 13]]] } },
]

function jsonResponse(body: unknown, ok = true) {
  return { ok, status: ok ? 200 : 500, statusText: ok ? 'OK' : 'Error', json: async () => body }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

async function waitForZonesLoaded() {
  await waitFor(() => expect(screen.getByText('PFZ-001')).toBeInTheDocument())
}

describe('ZonesExplorerPage — loading', () => {
  it('shows an error banner when the zones fetch fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false)))
    render(<ZonesExplorerPage />)
    await waitFor(() => expect(screen.getByText(/Couldn't load zones/)).toBeInTheDocument())
  })

  it('lists PFZ and restricted zones in their own sections with counts', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' })))
    render(<ZonesExplorerPage />)
    await waitForZonesLoaded()

    expect(screen.getByText('Potential Fishing Zones (1)')).toBeInTheDocument()
    expect(screen.getByText('Restricted Areas (1)')).toBeInTheDocument()
    expect(screen.getByText('Gulf of Mannar')).toBeInTheDocument()
    expect(screen.getByText('1 PFZ · 1 restricted')).toBeInTheDocument()
  })
})

describe('ZonesExplorerPage — selecting a zone', () => {
  it('clicking a zone row shows its detail panel, clicking again closes it', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' })))
    const user = userEvent.setup()
    render(<ZonesExplorerPage />)
    await waitForZonesLoaded()

    await user.click(screen.getByText('PFZ-001'))
    expect(screen.getByText('Potential Fishing Zone')).toBeInTheDocument()
    expect(screen.getByText('Good conditions')).toBeInTheDocument()

    // Close via the detail panel's own ✕ button.
    await user.click(screen.getByText('✕'))
    expect(screen.queryByText('Potential Fishing Zone')).not.toBeInTheDocument()
  })

  it('selecting a zone from the map (via onSelectZone) opens the same detail panel', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ zones: ZONES, generated_at: '2026-01-01T00:00:00Z' })))
    const user = userEvent.setup()
    render(<ZonesExplorerPage />)
    await waitForZonesLoaded()

    await user.click(screen.getByText('select-from-map'))
    expect(screen.getByTestId('map-selected-zone')).toHaveTextContent('KOCHI-PFZ-001')
    expect(screen.getByText('Potential Fishing Zone')).toBeInTheDocument()
  })
})
