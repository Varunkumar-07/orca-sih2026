import { useEffect, useRef, useState } from 'react'
import type { HistoryListResponse, HistoryRecordDetail, PageSource } from '../types'
import { ensureOk, friendlyError, devBackendHint } from '../lib/httpError'
import { localIsoDate, localIsoDateDaysAgo, localUtcOffsetMinutes } from '../lib/dates'
const PAGE_SOURCES: PageSource[] = ['chat', 'zones', 'weather', 'route', 'alerts', 'analytics', 'download']

const PAGE_SOURCE_STYLE: Record<PageSource, string> = {
  chat: 'bg-sky-100 text-sky-700',
  zones: 'bg-emerald-100 text-emerald-700',
  weather: 'bg-amber-100 text-amber-700',
  route: 'bg-indigo-100 text-indigo-700',
  alerts: 'bg-rose-100 text-rose-700',
  analytics: 'bg-violet-100 text-violet-700',
  download: 'bg-teal-100 text-teal-700',
}

const QUICK_RANGES = [
  { label: '7 Days', days: 7 },
  { label: '30 Days', days: 30 },
  { label: '90 Days', days: 90 },
]

const PAGE_SIZE = 10

function HistoryDetailPanel({ record, onClose }: { record: HistoryRecordDetail; onClose: () => void }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <div>
          <span className={`text-[9px] px-1.5 py-0.5 rounded-full uppercase tracking-wide ${PAGE_SOURCE_STYLE[record.page_source]}`}>
            {record.page_source}
          </span>
          <h3 className="text-sm font-bold text-slate-800 mt-1.5">{record.query_summary}</h3>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-sm leading-none">
          ✕
        </button>
      </div>

      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-500">
        <span>{new Date(record.timestamp).toLocaleString()}</span>
        <span className="font-mono">session: {record.session_id}</span>
        <span className="font-mono">ID: {record.id}</span>
      </div>

      <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-500 mb-1">Request</div>
          <pre className="text-[10px] font-mono bg-slate-50 border border-slate-200 rounded-lg p-2 max-h-64 overflow-auto whitespace-pre-wrap break-all">
            {JSON.stringify(record.payload?.request ?? null, null, 2)}
          </pre>
        </div>
        <div>
          <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-500 mb-1">Response</div>
          <pre className="text-[10px] font-mono bg-slate-50 border border-slate-200 rounded-lg p-2 max-h-64 overflow-auto whitespace-pre-wrap break-all">
            {JSON.stringify(record.payload?.response ?? null, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  )
}

export function HistoryPage() {
  const [pageSource, setPageSource] = useState<PageSource | ''>('')
  const [startDate, setStartDate] = useState(() => localIsoDateDaysAgo(30))
  const [endDate, setEndDate] = useState(() => localIsoDate())

  const [offset, setOffset] = useState(0)
  const [data, setData] = useState<HistoryListResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [detail, setDetail] = useState<HistoryRecordDetail | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)

  // Token-based request guard: fetchHistory can fire again (a new filter,
  // date range, or page) before an in-flight request resolves. Without
  // this, a slower/stale request's response can overwrite the result of a
  // newer one — the data shown then doesn't match the currently-selected
  // filter, with nothing indicating the mismatch.
  const fetchTokenRef = useRef(0)

  function fetchHistory() {
    const token = ++fetchTokenRef.current
    setLoading(true)
    setError(null)
    const params = new URLSearchParams({
      start_date: startDate,
      end_date: endDate,
      limit: String(PAGE_SIZE),
      offset: String(offset),
      // The dates above are the user's local calendar days; this tells the
      // backend which timezone they are days in.
      tz_offset_minutes: String(localUtcOffsetMinutes()),
    })
    if (pageSource) params.set('page_source', pageSource)

    fetch(`/api/history?${params.toString()}`)
      .then(ensureOk)
      .then((res) => res.json() as Promise<HistoryListResponse>)
      .then((d) => {
        if (fetchTokenRef.current === token) setData(d)
      })
      .catch((e: unknown) => {
        if (fetchTokenRef.current === token) setError(friendlyError(e))
      })
      .finally(() => {
        if (fetchTokenRef.current === token) setLoading(false)
      })
  }

  useEffect(() => {
    fetchHistory()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageSource, startDate, endDate, offset])

  function selectPageSource(next: PageSource | '') {
    setPageSource(next)
    setOffset(0)
  }

  function applyQuickRange(days: number) {
    setStartDate(localIsoDateDaysAgo(days))
    setEndDate(localIsoDate())
    setOffset(0)
  }

  function selectRecord(id: number) {
    if (selectedId === id) {
      setSelectedId(null)
      setDetail(null)
      return
    }
    setSelectedId(id)
    setDetail(null)
    setDetailError(null)
    fetch(`/api/history/${id}`)
      .then(ensureOk)
      .then((res) => res.json() as Promise<HistoryRecordDetail>)
      .then(setDetail)
      .catch((e: unknown) => setDetailError(friendlyError(e)))
  }

  const total = data?.total ?? 0
  const from = total === 0 ? 0 : offset + 1
  const to = Math.min(offset + PAGE_SIZE, total)

  return (
    <div className="h-full overflow-y-auto">
      {/* Intro block — ported from Mock's .page-hero (badge + H1 + subtitle) */}
      <div className="shrink-0 px-4 sm:px-6 pt-6 pb-2 text-center">
        <span className="inline-flex items-center gap-1.5 text-[11px] tracking-[.06em] uppercase px-2.5 py-1.5 rounded-full bg-teal-500/[.14] border border-teal-500/[.28] text-teal-700">
          History · Smart India Hackathon 2026 · ISRO
        </span>
        <h1 className="mt-2.5 text-[clamp(26px,3.6vw,38px)] leading-none tracking-[-.03em] text-[#0d0c0b] font-normal">
          History
        </h1>
        <p className="mt-2.5 text-sm leading-relaxed text-[#0d0c0b]/64 max-w-[52ch] mx-auto">
          Revisit past queries and their results.
        </p>
      </div>

      <div className="max-w-5xl mx-auto px-4 pb-8">
      <div className="rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg p-4 space-y-4">
        <div className="flex items-center justify-end gap-1 bg-slate-100 rounded-full p-0.5 w-fit ml-auto">
          {QUICK_RANGES.map((r) => (
            <button
              key={r.days}
              onClick={() => applyQuickRange(r.days)}
              className="text-[11px] px-3 py-1 rounded-full transition text-slate-500 hover:text-slate-700"
            >
              {r.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
            Couldn't load history: {error}{devBackendHint()}
          </div>
        )}

        <div className="rounded-xl border border-slate-200 bg-white p-3 space-y-3">
          <div>
            <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-500 mb-1.5">Page</div>
            <div className="flex flex-wrap gap-1.5">
              <button
                onClick={() => selectPageSource('')}
                className={`text-[11px] px-3 py-1.5 rounded-full border transition ${
                  pageSource === ''
                    ? 'bg-slate-900 text-white border-slate-900 font-semibold'
                    : 'bg-white text-slate-600 border-slate-200 hover:border-slate-400'
                }`}
              >
                All
              </button>
              {PAGE_SOURCES.map((p) => (
                <button
                  key={p}
                  onClick={() => selectPageSource(p)}
                  className={`text-[11px] px-3 py-1.5 rounded-full border transition capitalize ${
                    pageSource === p
                      ? 'bg-slate-900 text-white border-slate-900 font-semibold'
                      : 'bg-white text-slate-600 border-slate-200 hover:border-slate-400'
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="date"
              value={startDate}
              onChange={(e) => {
                setStartDate(e.target.value)
                setOffset(0)
              }}
              max={endDate}
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            />
            <span className="text-slate-400 text-xs">to</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => {
                setEndDate(e.target.value)
                setOffset(0)
              }}
              min={startDate}
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            />
          </div>
        </div>

        {selectedId !== null && (
          <>
            {detailError && (
              <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
                Couldn't load record detail: {detailError}
              </div>
            )}
            {detail && detail.id === selectedId && <HistoryDetailPanel record={detail} onClose={() => setSelectedId(null)} />}
            {!detail && !detailError && <div className="text-xs text-slate-400 px-1">Loading detail…</div>}
          </>
        )}

        {loading && !data && <div className="text-xs text-slate-400 px-1">Fetching history…</div>}

        {data && data.items.length === 0 && (
          <div className="rounded-xl border border-slate-200 bg-white p-6 text-center">
            <div className="text-2xl mb-2">🗒️</div>
            <p className="text-sm font-semibold text-slate-700">No lookups in this range</p>
            <p className="text-xs text-slate-500 mt-1">Try a wider date range or a different page filter.</p>
          </div>
        )}

        {data && data.items.length > 0 && (
          <div className="rounded-xl border border-slate-200 bg-white overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-[10px] uppercase tracking-wide text-slate-500">
                  <th className="text-left font-semibold px-3 py-2 whitespace-nowrap">Time</th>
                  <th className="text-left font-semibold px-3 py-2">Page</th>
                  <th className="text-left font-semibold px-3 py-2">Query / Action</th>
                  <th className="hidden sm:table-cell text-left font-semibold px-3 py-2">Session</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item) => (
                  <tr
                    key={item.id}
                    onClick={() => selectRecord(item.id)}
                    className={`border-b border-slate-100 last:border-b-0 cursor-pointer transition ${
                      selectedId === item.id ? 'bg-teal-50' : 'hover:bg-slate-50'
                    }`}
                  >
                    <td className="px-3 py-2 text-slate-500 whitespace-nowrap">{new Date(item.timestamp).toLocaleString()}</td>
                    <td className="px-3 py-2">
                      <span className={`text-[9px] px-1.5 py-0.5 rounded-full uppercase tracking-wide whitespace-nowrap ${PAGE_SOURCE_STYLE[item.page_source]}`}>
                        {item.page_source}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-slate-700 max-w-[160px] sm:max-w-md truncate">{item.query_summary}</td>
                    <td className="hidden sm:table-cell px-3 py-2 text-slate-400 font-mono">{item.session_id.slice(0, 12)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {data && total > 0 && (
          <div className="flex items-center justify-between text-[11px] text-slate-500">
            <span>
              {from}–{to} of {total}
            </span>
            <div className="flex gap-1">
              <button
                onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                disabled={offset === 0}
                className="px-2.5 py-1 rounded-full bg-slate-100 text-slate-600 disabled:opacity-40 hover:bg-slate-200 transition"
              >
                Prev
              </button>
              <button
                onClick={() => setOffset((o) => o + PAGE_SIZE)}
                disabled={to >= total}
                className="px-2.5 py-1 rounded-full bg-slate-100 text-slate-600 disabled:opacity-40 hover:bg-slate-200 transition"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
      </div>
    </div>
  )
}
