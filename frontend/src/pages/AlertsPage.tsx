import { useEffect, useState } from 'react'
import type { AlertRecord, AlertsResponse } from '../types'

const ALERT_ICON: Record<AlertRecord['alert_type'], string> = {
  cyclone: '🌀',
  lightning: '⚡',
}

const SEVERITY_STYLE: Record<AlertRecord['severity'], string> = {
  high: 'bg-rose-100 text-rose-700 border-rose-300',
  moderate: 'bg-amber-100 text-amber-700 border-amber-300',
}

function AlertCard({ alert }: { alert: AlertRecord }) {
  return (
    <div className={`rounded-xl border-l-4 bg-white p-4 shadow-sm ${alert.severity === 'high' ? 'border-l-rose-500' : 'border-l-amber-500'} border-y border-r border-slate-200`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xl" aria-hidden>
            {ALERT_ICON[alert.alert_type]}
          </span>
          <div>
            <h3 className="text-sm font-bold text-slate-800">{alert.zone_name}</h3>
            {alert.near && <p className="text-[11px] text-slate-500">near {alert.near}</p>}
          </div>
        </div>
        <span
          className={`shrink-0 text-[10px] px-2 py-0.5 rounded-full border font-semibold uppercase tracking-wide ${SEVERITY_STYLE[alert.severity]}`}
        >
          {alert.severity}
        </span>
      </div>
      <p className="text-xs text-slate-600 mt-2">{alert.detail}</p>
      <div className="text-[10px] text-slate-400 mt-2 font-mono">ID: {alert.zone_id}</div>
    </div>
  )
}

export function AlertsPage() {
  const [data, setData] = useState<AlertsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  function fetchAlerts() {
    fetch('/api/alerts')
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
        return res.json() as Promise<AlertsResponse>
      })
      .then((d) => {
        setData(d)
        setError(null)
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    fetchAlerts()
  }, [])

  function handleRefresh() {
    setLoading(true)
    setError(null)
    fetchAlerts()
  }

  const cycloneCount = data?.alerts.filter((a) => a.alert_type === 'cyclone').length ?? 0
  const lightningCount = data?.alerts.filter((a) => a.alert_type === 'lightning').length ?? 0

  return (
    <div className="h-full overflow-y-auto">
      {/* Intro block — ported from Mock's .page-hero (badge + H1 + subtitle) */}
      <div className="shrink-0 px-4 sm:px-6 pt-6 pb-2 text-center">
        <span className="inline-flex items-center gap-1.5 text-[11px] tracking-[.06em] uppercase px-2.5 py-1.5 rounded-full bg-teal-500/[.14] border border-teal-500/[.28] text-teal-700">
          Alerts · Smart India Hackathon 2026 · ISRO
        </span>
        <h1 className="mt-2.5 text-[clamp(26px,3.6vw,38px)] leading-none tracking-[-.03em] text-[#0d0c0b] font-normal">
          Alerts &amp; Advisories
        </h1>
        <p className="mt-2.5 text-sm leading-relaxed text-[#0d0c0b]/64 max-w-[52ch] mx-auto">
          Active cyclone and lightning advisories across tracked fishing zones.
        </p>
      </div>

      <div className="max-w-3xl mx-auto px-4 pb-8">
      <div className="rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg p-4 space-y-4">
        <div className="flex items-center justify-between gap-3 pb-2 border-b border-white/40">
          <span className="text-[11px] text-slate-500">
            {data ? `${data.checked_zones} zones checked` : loading ? 'Checking…' : ''}
          </span>
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="text-[11px] px-2.5 py-1 rounded-full bg-slate-900 text-white disabled:opacity-40 hover:bg-black transition"
          >
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>

        {error && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
            Couldn't load alerts: {error}. Is the backend running on :8000?
          </div>
        )}

        {data && data.alerts.length > 0 && (
          <div className="flex gap-2 flex-wrap">
            {cycloneCount > 0 && (
              <span className="text-[11px] px-2.5 py-1 rounded-full bg-rose-100 text-rose-700 font-semibold">
                🌀 {cycloneCount} cyclone {cycloneCount === 1 ? 'alert' : 'alerts'}
              </span>
            )}
            {lightningCount > 0 && (
              <span className="text-[11px] px-2.5 py-1 rounded-full bg-amber-100 text-amber-700 font-semibold">
                ⚡ {lightningCount} lightning {lightningCount === 1 ? 'alert' : 'alerts'}
              </span>
            )}
          </div>
        )}

        {loading && !data && <div className="text-xs text-slate-400 px-1">Checking all tracked zones for active alerts…</div>}

        {data && data.alerts.length === 0 && (
          <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-6 text-center">
            <div className="text-2xl mb-2">✅</div>
            <p className="text-sm font-semibold text-emerald-800">No active cyclone or lightning alerts</p>
            <p className="text-xs text-emerald-700 mt-1">
              All {data.checked_zones} tracked fishing zones currently report calm conditions.
            </p>
          </div>
        )}

        {data && data.alerts.length > 0 && (
          <div className="space-y-3">
            {data.alerts.map((a) => (
              <AlertCard key={`${a.zone_id}-${a.alert_type}`} alert={a} />
            ))}
          </div>
        )}

        {data && (
          <div className="text-[10px] text-slate-400 text-center pt-2">
            Last checked {new Date(data.generated_at).toLocaleString()}
          </div>
        )}
      </div>
      </div>
    </div>
  )
}
