import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ChatMapPage from '../ChatMapPage'
import type { FinalResponse } from '../../types'

// MapView renders a real Leaflet map, which needs DOM APIs jsdom doesn't
// fully provide (ResizeObserver, real layout/canvas). ChatMapPage's own
// logic — sendQuery, the selectedZone reset, the trace reveal — is what's
// under test here, not Leaflet's internals (Leaflet has no tests of its
// own to duplicate). Stubbed with a minimal component that still exposes
// onSelectZone, so the selectedZone-reset behavior (the exact bug fixed
// earlier this session — see Layout.tsx/ChatMapPage's derived-state
// pattern) stays genuinely exercised, not just assumed.
vi.mock('../../components/MapView', () => ({
  MapView: ({ payload, onSelectZone }: { payload: { pins: { label: string }[] } | null; onSelectZone?: (id: string | null) => void }) => (
    <div data-testid="map-view-stub">
      {payload?.pins.map((p) => (
        <button key={p.label} onClick={() => onSelectZone?.(p.label)}>
          select {p.label}
        </button>
      ))}
    </div>
  ),
}))

function finalResponse(overrides: Partial<FinalResponse> = {}): FinalResponse {
  return {
    answer_text: 'Query: is it safe?\nLocation: 13.0800, 80.2700\n✅ Safe to go (confidence 88%) — Calm seas.',
    reasoning_trace: [
      { agent_name: 'planning_agent', input_summary: 'q', output_summary: 'ok', timestamp: '2026-01-01T00:00:00Z' },
      { agent_name: 'marine_data_agent', input_summary: 'q', output_summary: 'ok', timestamp: '2026-01-01T00:00:00Z' },
    ],
    map_payload: {
      pins: [{ lat: 13.1, lon: 80.3, label: 'PFZ-001', type: 'pfz', zone_id: 'PFZ-001' }],
      overlays: [],
    },
    ...overrides,
  }
}

function mockFetchOnce(response: FinalResponse, opts: { ok?: boolean; status?: number } = {}) {
  const ok = opts.ok ?? true
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValueOnce({
      ok,
      status: opts.status ?? (ok ? 200 : 500),
      headers: { get: () => null },
      json: async () => response,
      text: async () => JSON.stringify(response),
    }),
  )
}

beforeEach(() => {
  // jsdom has no navigator.geolocation by default — getUserLocation()'s
  // own `!navigator.geolocation` check already handles that gracefully,
  // so no polyfill is needed; asserting that here would just be
  // re-testing jsdom itself.
  vi.stubGlobal('fetch', vi.fn())
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ChatMapPage — initial render', () => {
  it('shows the welcome message and example queries, no map/trace panels yet', () => {
    render(<ChatMapPage />)

    expect(screen.getByText('Welcome to ORCA')).toBeInTheDocument()
    expect(screen.getByText(/is it safe to go out tomorrow near Chennai/)).toBeInTheDocument()
    expect(screen.queryByTestId('map-view-stub')).not.toBeInTheDocument()
    expect(screen.queryByText(/Live Reasoning Trace/)).not.toBeInTheDocument()
  })
})

describe('ChatMapPage — sending a query', () => {
  it('sends the demo-mode request, shows the loading indicator, then renders the answer + map + trace', async () => {
    const user = userEvent.setup()
    // A manually-controlled promise, not an instantly-resolving mock —
    // otherwise the loading state can flip back to false before this test
    // ever gets to assert it was true, since everything else here is
    // synchronous/microtask-fast.
    let resolveFetch!: (v: unknown) => void
    const pending = new Promise((resolve) => {
      resolveFetch = resolve
    })
    vi.stubGlobal('fetch', vi.fn().mockReturnValueOnce(pending))
    render(<ChatMapPage />)

    await user.click(screen.getByRole('button', { name: /is it safe to go out tomorrow near Chennai/ }))

    expect(screen.getByText(/ORCA is reasoning/)).toBeInTheDocument()

    resolveFetch({
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => finalResponse(),
    })

    await waitFor(() => expect(screen.getByText(/Safe to go \(confidence 88%\)/)).toBeInTheDocument())

    expect(fetch).toHaveBeenCalledWith(
      '/api/query/demo',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(screen.getByTestId('map-view-stub')).toBeInTheDocument()
    // Trace reveal is a real 350ms-per-step setInterval — wait for it to
    // finish counting up to the full step count rather than mocking
    // timers, keeping this test close to real runtime behavior.
    await waitFor(() => expect(screen.getByText('2/2 steps')).toBeInTheDocument(), { timeout: 3000 })
  })

  it('calls /api/query/full when Live mode is toggled on', async () => {
    const user = userEvent.setup()
    mockFetchOnce(finalResponse())
    render(<ChatMapPage />)

    await user.click(screen.getByRole('button', { name: /toggle demo\/live/i }))
    await user.click(screen.getByRole('button', { name: /is it safe to go out tomorrow near Chennai/ }))

    await waitFor(() => expect(fetch).toHaveBeenCalled())
    expect(fetch).toHaveBeenCalledWith('/api/query/full', expect.anything())
  })

  it('shows an error message in chat and the error banner when the request fails', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 500,
        headers: { get: () => null },
        text: async () => 'internal error',
      }),
    )
    render(<ChatMapPage />)

    await user.click(screen.getByRole('button', { name: /is it safe to go out tomorrow near Chennai/ }))

    await waitFor(() => expect(screen.getByText(/request failed/)).toBeInTheDocument())
    // Appears twice by design — once in the chat bubble, once in the
    // dismissable error banner above it — so assert there's at least one
    // rather than picking an arbitrarily "the" element.
    expect(screen.getAllByText(/500 internal error/).length).toBeGreaterThan(0)
  })

  it('the re-entrancy guard blocks a second send while the first request is still in flight', async () => {
    const user = userEvent.setup()
    let resolveFetch!: (v: unknown) => void
    const pending = new Promise((resolve) => {
      resolveFetch = resolve
    })
    vi.stubGlobal('fetch', vi.fn().mockReturnValueOnce(pending))
    render(<ChatMapPage />)

    const input = screen.getByPlaceholderText(/Ask e.g\./)
    await user.type(input, 'is it safe?')
    const sendButton = screen.getByRole('button', { name: 'Send' })
    await user.click(sendButton)
    // Second click while still loading — the Send button is disabled
    // during loading, so clicking it again is a no-op; this confirms the
    // UI itself prevents the double-send the component's own guard also
    // protects against.
    await user.click(sendButton)

    expect(fetch).toHaveBeenCalledTimes(1)
    // The input is disabled purely by `loading` (unlike the Send button,
    // which is ALSO disabled once sendQuery clears the input text) — the
    // more direct signal that the component is still mid-request.
    expect(input).toBeDisabled()

    resolveFetch({
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => finalResponse(),
    })
    await waitFor(() => expect(input).not.toBeDisabled())
  })
})

describe('ChatMapPage — selectedZone resets on new map data', () => {
  it('clears the selected-zone panel when a second query brings a new mapPayload', async () => {
    const user = userEvent.setup()
    mockFetchOnce(
      finalResponse({
        map_payload: {
          pins: [{ lat: 13.1, lon: 80.3, label: 'PFZ-001', type: 'pfz', zone_id: 'PFZ-001' }],
          overlays: [],
        },
      }),
    )
    render(<ChatMapPage />)

    await user.click(screen.getByRole('button', { name: /is it safe to go out tomorrow near Chennai/ }))
    await waitFor(() => expect(screen.getByTestId('map-view-stub')).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: 'select PFZ-001' }))
    expect(screen.getByText(/Selected PFZ:/)).toBeInTheDocument()
    expect(within(screen.getByText(/Selected PFZ:/).closest('span')!).getByText('PFZ-001')).toBeInTheDocument()

    // Second query, a genuinely different mapPayload (different pin) —
    // must clear the selection made against the FIRST response's zones,
    // since those zone ids may not even exist in the new response.
    mockFetchOnce(
      finalResponse({
        answer_text: 'Query: second query\nLocation: 9.9300, 76.2700\n✅ Safe to go (confidence 90%) — Calm.',
        map_payload: {
          pins: [{ lat: 9.9, lon: 76.3, label: 'PFZ-002', type: 'pfz', zone_id: 'PFZ-002' }],
          overlays: [],
        },
      }),
    )
    const input = screen.getByPlaceholderText(/Ask e.g\./)
    await user.type(input, 'second query')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    // "second query" itself matches both the user's own message bubble
    // AND the echoed "Query: second query" line in ORCA's reply — assert
    // on something unique to the reply having actually landed instead.
    await waitFor(() => expect(screen.getByText(/confidence 90%/)).toBeInTheDocument())
    expect(screen.queryByText(/Selected PFZ:/)).not.toBeInTheDocument()
  })
})
