import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MapView } from '../MapView'
import type { MapPayload } from '../../types'

// FitBounds (inside MapView) calls `new ResizeObserver(...)` to re-fit the
// map whenever its container resizes — jsdom doesn't implement
// ResizeObserver, so without a stub, mounting MapView with any payload
// throws "ResizeObserver is not defined" before anything can render.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('MapView — empty payload', () => {
  it('shows the placeholder when payload is null', () => {
    render(<MapView payload={null} />)
    expect(screen.getByText('Map will appear here')).toBeInTheDocument()
  })

  it('shows the placeholder when payload has no pins and no overlays', () => {
    render(<MapView payload={{ pins: [], overlays: [] }} />)
    expect(screen.getByText('Map will appear here')).toBeInTheDocument()
  })
})

describe('MapView — with pins', () => {
  const payload: MapPayload = {
    pins: [{ lat: 13.08, lon: 80.27, label: 'PFZ-001', type: 'pfz', zone_id: 'PFZ-001' }],
    overlays: [],
  }

  it('renders a real Leaflet map container instead of the placeholder', () => {
    const { container } = render(<MapView payload={payload} />)
    expect(screen.queryByText('Map will appear here')).not.toBeInTheDocument()
    expect(container.querySelector('.leaflet-container')).not.toBeNull()
  })

  it('renders one marker per pin', () => {
    const { container } = render(<MapView payload={payload} />)
    expect(container.querySelectorAll('.leaflet-marker-icon')).toHaveLength(1)
  })
})

describe('MapView — restricted-area overlay', () => {
  it('renders a polygon for a restricted_area overlay without crashing', () => {
    const payload: MapPayload = {
      pins: [],
      overlays: [
        {
          type: 'restricted_area',
          name: 'Gulf of Mannar',
          geojson: { type: 'Polygon', coordinates: [[[80, 13], [81, 13], [81, 14], [80, 13]]] },
          highlighted: false,
        },
      ],
    }
    const { container } = render(<MapView payload={payload} />)
    expect(container.querySelector('.leaflet-container')).not.toBeNull()
    expect(container.querySelector('path')).not.toBeNull()
  })
})

describe('MapView — route', () => {
  it('renders a route polyline plus start/destination circle markers', () => {
    const payload: MapPayload = {
      // RoutePlannerPage always pairs a route with at least a "Start" pin —
      // an empty pins/overlays payload falls into MapView's own placeholder
      // branch regardless of `route`, so this mirrors real usage rather than
      // exercising a combination the app never actually produces.
      pins: [{ lat: 13.08, lon: 80.27, label: 'Start', type: 'query' }],
      overlays: [],
      route: [{ lat: 13.08, lon: 80.27 }, { lat: 12.5, lon: 79.9 }, { lat: 9.93, lon: 76.27 }],
    }
    const { container } = render(<MapView payload={payload} />)
    // The Polyline itself plus the two CircleMarkers (start/destination) are
    // all SVG <path> elements in Leaflet's default (non-canvas) renderer.
    expect(container.querySelectorAll('path').length).toBeGreaterThanOrEqual(3)
  })
})
