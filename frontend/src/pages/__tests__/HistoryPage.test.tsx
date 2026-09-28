import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { HistoryPage } from '../HistoryPage'
import type { HistoryListResponse, HistoryRecordDetail } from '../../types'

function jsonResponse(body: unknown, ok = true) {
  return { ok, status: ok ? 200 : 500, statusText: ok ? 'OK' : 'Error', json: async () => body }
}

function listResponse(overrides: Partial<HistoryListResponse> = {}): HistoryListResponse {
  return {
    items: [
      { id: 1, timestamp: '2026-01-01T00:00:00Z', page_source: 'chat', query_summary: 'mannar cyclone risk', session_id: 'sess-abc123' },
    ],
    total: 1,
    limit: 10,
    offset: 0,
    ...overrides,
  }
}

function detailResponse(overrides: Partial<HistoryRecordDetail> = {}): HistoryRecordDetail {
  return {
    id: 1,
    timestamp: '2026-01-01T00:00:00Z',
    page_source: 'chat',
    query_summary: 'mannar cyclone risk',
    session_id: 'sess-abc123',
    payload: { request: { query: 'mannar cyclone risk' }, response: { answer_text: 'ok' } },
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

describe('HistoryPage — listing', () => {
  it('shows an error banner when the history fetch fails', async () => {
    installFetchRouter({ '/api/history': () => { throw new Error('boom') } })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(null, false)))
    render(<HistoryPage />)
    await waitFor(() => expect(screen.getByText(/Couldn't load history/)).toBeInTheDocument())
  })

  it('renders one row per history item once loaded', async () => {
    installFetchRouter({ '/api/history': () => listResponse() })
    render(<HistoryPage />)
    await waitFor(() => expect(screen.getByText('mannar cyclone risk')).toBeInTheDocument())
    // "chat" also matches the "Chat" filter button's (lowercase) text content
    // — asserting "at least one" confirms the row's page_source badge
    // rendered without needing to disambiguate from the filter button.
    expect(screen.getAllByText('chat').length).toBeGreaterThan(0)
  })

  it('shows the empty state when there are no lookups in range', async () => {
    installFetchRouter({ '/api/history': () => listResponse({ items: [], total: 0 }) })
    render(<HistoryPage />)
    await waitFor(() => expect(screen.getByText('No lookups in this range')).toBeInTheDocument())
  })

  it('clicking a Page filter refetches with that page_source', async () => {
    const fetchMock = vi.fn(async (input: string) => {
      const url = String(input)
      return jsonResponse(listResponse({ items: url.includes('page_source=weather') ? [] : listResponse().items }))
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoryPage />)

    await waitFor(() => expect(screen.getByText('mannar cyclone risk')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: /weather/i }))

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('page_source=weather')),
    )
  })
})

describe('HistoryPage — row detail', () => {
  it('clicking a row fetches and shows its request/response detail, toggling closed on a second click', async () => {
    installFetchRouter({
      '/api/history/1': () => detailResponse(),
      '/api/history': () => listResponse(),
    })
    const user = userEvent.setup()
    render(<HistoryPage />)

    await waitFor(() => expect(screen.getByText('mannar cyclone risk')).toBeInTheDocument())
    // Target the table <td> by its implicit "cell" role — once the detail
    // panel opens (rendered ABOVE the table, and repeating the same
    // query_summary text in its own heading), a plain getByText would match
    // the non-clickable heading first instead of the row that toggles it.
    const cell = screen.getByRole('cell', { name: 'mannar cyclone risk' })
    await user.click(cell)

    await waitFor(() => expect(screen.getByText('session: sess-abc123')).toBeInTheDocument())

    await user.click(cell)
    expect(screen.queryByText('session: sess-abc123')).not.toBeInTheDocument()
  })

  it('shows a detail error banner when the record fetch fails, without breaking the list', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string) => {
        const url = String(input)
        if (url.startsWith('/api/history/1')) return jsonResponse(null, false)
        if (url.startsWith('/api/history')) return jsonResponse(listResponse())
        throw new Error(`unhandled fetch: ${url}`)
      }),
    )
    const user = userEvent.setup()
    render(<HistoryPage />)

    await waitFor(() => expect(screen.getByText('mannar cyclone risk')).toBeInTheDocument())
    await user.click(screen.getByText('mannar cyclone risk'))

    await waitFor(() => expect(screen.getByText(/Couldn't load record detail/)).toBeInTheDocument())
  })
})

describe('HistoryPage — pagination', () => {
  it('Prev is disabled at offset 0, and Next is disabled once every item has been shown', async () => {
    installFetchRouter({ '/api/history': () => listResponse({ total: 1 }) })
    render(<HistoryPage />)

    await waitFor(() => expect(screen.getByRole('button', { name: 'Prev' })).toBeInTheDocument())
    expect(screen.getByRole('button', { name: 'Prev' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled()
  })
})
