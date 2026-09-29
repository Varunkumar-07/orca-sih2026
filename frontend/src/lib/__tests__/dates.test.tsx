// Default date ranges between 00:00 and 05:30 IST — India is already on
// the next calendar day, UTC is not. Every page used to build its range
// from toISOString() (UTC), so "today" was still yesterday in that window
// and History hid the night's rows. Runs with the process timezone set to
// Asia/Kolkata and the clock frozen at 00:30 IST on 30 Sep 2026.
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { localIsoDate, localIsoDateDaysAgo, localUtcOffsetMinutes } from '../dates'
import { HistoryPage } from '../../pages/HistoryPage'
import { DownloadPage } from '../../pages/DownloadPage'
import { AnalyticsPage } from '../../pages/AnalyticsPage'
import type { ZoneRecord } from '../../types'

const JUST_AFTER_MIDNIGHT_IST = new Date('2026-09-29T19:00:00Z') // 00:30 IST, 30 Sep
const ZONES: ZoneRecord[] = [
  { id: 'KOCHI-PFZ-001', name: 'PFZ-001', type: 'pfz', near: 'Kochi', coordinates: { lat: 9.93, lon: 76.27 } },
]

beforeAll(() => {
  vi.stubEnv('TZ', 'Asia/Kolkata')
})

afterAll(() => {
  vi.unstubAllEnvs()
})

beforeEach(() => {
  // Only Date is faked — React, userEvent and waitFor keep real timers.
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(JUST_AFTER_MIDNIGHT_IST)
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

function jsonResponse(body: unknown) {
  return { ok: true, status: 200, statusText: 'OK', json: async () => body }
}

function stubFetch() {
  const fetchMock = vi.fn(async (input: string) => {
    const url = String(input)
    if (url.startsWith('/api/zones')) return jsonResponse({ zones: ZONES })
    if (url.startsWith('/api/history')) return jsonResponse({ items: [], total: 0, limit: 10, offset: 0 })
    if (url.startsWith('/api/analytics/historical')) {
      return jsonResponse({ lat: 9.93, lon: 76.27, start_date: '', end_date: '', series: {} })
    }
    throw new Error(`unhandled fetch in test: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function dateInputs(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll<HTMLInputElement>('input[type="date"]')).map((el) => el.value)
}

describe('lib/dates at 00:30 IST', () => {
  it('the UTC date is still yesterday — the bug', () => {
    expect(new Date().toISOString().slice(0, 10)).toBe('2026-09-29')
  })

  it('localIsoDate is the Indian calendar day', () => {
    expect(localIsoDate()).toBe('2026-09-30')
  })

  it('localIsoDateDaysAgo counts back from the local day', () => {
    expect(localIsoDateDaysAgo(7)).toBe('2026-09-23')
    expect(localIsoDateDaysAgo(30)).toBe('2026-08-31')
  })

  it('localUtcOffsetMinutes is IST', () => {
    expect(localUtcOffsetMinutes()).toBe(330)
  })
})

describe('page default ranges at 00:30 IST', () => {
  it('History asks for the last 30 local days in IST, ending today', async () => {
    const fetchMock = stubFetch()
    const { container } = render(<HistoryPage />)

    expect(dateInputs(container)).toEqual(['2026-08-31', '2026-09-30'])
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const url = new URL(String(fetchMock.mock.calls[0][0]), 'http://x')
    expect(url.searchParams.get('start_date')).toBe('2026-08-31')
    expect(url.searchParams.get('end_date')).toBe('2026-09-30')
    expect(url.searchParams.get('tz_offset_minutes')).toBe('330')
  })

  it('History quick ranges end today too', async () => {
    stubFetch()
    const user = userEvent.setup()
    const { container } = render(<HistoryPage />)
    await user.click(screen.getByRole('button', { name: '7 Days' }))
    expect(dateInputs(container)).toEqual(['2026-09-23', '2026-09-30'])
  })

  it('Download defaults to the last 7 local days, ending today', async () => {
    stubFetch()
    const user = userEvent.setup()
    const { container } = render(<DownloadPage />)
    expect(dateInputs(container)).toEqual(['2026-09-23', '2026-09-30'])
    await user.click(screen.getByRole('button', { name: '30 Days' }))
    expect(dateInputs(container)).toEqual(['2026-08-31', '2026-09-30'])
  })

  it('Analytics requests the last 7 local days, ending today', async () => {
    const fetchMock = stubFetch()
    const user = userEvent.setup()
    render(<AnalyticsPage />)
    await waitFor(() => expect(screen.getByRole('option', { name: /PFZ-001 — near Kochi/ })).toBeInTheDocument())
    await user.selectOptions(screen.getByRole('combobox'), 'KOCHI-PFZ-001')

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('start_date=2026-09-23&end_date=2026-09-30')),
    )
  })
})
