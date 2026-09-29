import { useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { AnalyticsResponse, AnalyticsSeries, ZoneRecord } from '../types'
import { ensureOk, friendlyError, devBackendHint } from '../lib/httpError'

const RANGE_OPTIONS = [
  { label: '7 Days', days: 7 },
  { label: '30 Days', days: 30 },
  { label: '90 Days', days: 90 },
]

// Direction is circular data (0deg wraps to 360deg) — a line chart implies
// an ordering/trend that doesn't exist for a compass bearing, so these are
// shown as a data table instead. moon_phase is similarly cyclic/categorical
// (0=1=new moon), shown as a table of phase names rather than its raw
// 0-1 fraction. Mirrors analytics_service.py's DIRECTIONAL_VARIABLES /
// NON_CHART_VARIABLES — kept as a small local constant here rather than a
// new API surface just to ship one set of strings across the wire, same as
// VARIABLE_META below already duplicates backend display labels.
const DIRECTIONAL_VARIABLES = new Set(['wave_direction_deg', 'swell_wave_direction_deg', 'current_direction_deg', 'wind_direction_deg'])
const NON_CHART_VARIABLES = new Set(['moon_phase'])

const CATEGORIES: { key: string; label: string; variables: string[] }[] = [
  { key: 'temperature', label: 'Temperature', variables: ['sst_celsius', 'air_temp_celsius'] },
  {
    key: 'waves',
    label: 'Waves',
    variables: ['wave_height_m', 'wave_direction_deg', 'wave_period_s', 'swell_wave_height_m', 'swell_wave_direction_deg', 'swell_wave_period_s'],
  },
  { key: 'currents', label: 'Currents', variables: ['current_velocity_kmh', 'current_direction_deg'] },
  { key: 'wind', label: 'Wind', variables: ['wind_kmh', 'wind_direction_deg', 'wind_gusts_kmh'] },
  { key: 'atmospheric', label: 'Atmospheric', variables: ['precipitation_mm', 'cloud_cover_pct', 'pressure_hpa', 'humidity_pct'] },
  { key: 'biological', label: 'Biological', variables: ['chlorophyll_mg_m3', 'salinity_psu', 'dissolved_oxygen_mmol_m3'] },
  { key: 'astronomical', label: 'Astronomical', variables: ['sunrise_hour_ist', 'sunset_hour_ist', 'moon_phase'] },
]

const VARIABLE_META: Record<string, { title: string; color: string }> = {
  sst_celsius: { title: 'Sea Surface Temperature', color: '#0ea5e9' },
  air_temp_celsius: { title: 'Air Temperature', color: '#f97316' },
  wave_height_m: { title: 'Wave Height', color: '#6366f1' },
  wave_direction_deg: { title: 'Wave Direction', color: '#6366f1' },
  wave_period_s: { title: 'Wave Period', color: '#8b5cf6' },
  swell_wave_height_m: { title: 'Swell Wave Height', color: '#0891b2' },
  swell_wave_direction_deg: { title: 'Swell Wave Direction', color: '#0891b2' },
  swell_wave_period_s: { title: 'Swell Wave Period', color: '#06b6d4' },
  current_velocity_kmh: { title: 'Current Velocity', color: '#ec4899' },
  current_direction_deg: { title: 'Current Direction', color: '#ec4899' },
  wind_kmh: { title: 'Wind Speed', color: '#f59e0b' },
  wind_direction_deg: { title: 'Wind Direction', color: '#14b8a6' },
  wind_gusts_kmh: { title: 'Wind Gusts', color: '#eab308' },
  precipitation_mm: { title: 'Precipitation', color: '#3b82f6' },
  cloud_cover_pct: { title: 'Cloud Cover', color: '#64748b' },
  pressure_hpa: { title: 'Sea-Level Pressure', color: '#7c3aed' },
  humidity_pct: { title: 'Relative Humidity', color: '#0ea5e9' },
  chlorophyll_mg_m3: { title: 'Chlorophyll-a', color: '#16a34a' },
  salinity_psu: { title: 'Salinity', color: '#0d9488' },
  dissolved_oxygen_mmol_m3: { title: 'Dissolved Oxygen', color: '#2563eb' },
  sunrise_hour_ist: { title: 'Sunrise (IST)', color: '#f59e0b' },
  sunset_hour_ist: { title: 'Sunset (IST)', color: '#ea580c' },
  moon_phase: { title: 'Moon Phase', color: '#64748b' },
}

const STATUS_STYLE: Record<AnalyticsSeries['status'], string> = {
  ok: 'bg-emerald-100 text-emerald-700',
  partial: 'bg-amber-100 text-amber-700',
  error: 'bg-rose-100 text-rose-700',
}

const COMPASS_POINTS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

function compassPoint(deg: number): string {
  return COMPASS_POINTS[Math.round(deg / 22.5) % 16]
}

function isoDate(d: Date): string {
  return d.toISOString().slice(0, 10)
}

function ChartCard({ variable, series }: { variable: string; series: AnalyticsSeries }) {
  const meta = VARIABLE_META[variable] ?? { title: variable, color: '#64748b' }
  const data = series.dates.map((d, i) => ({ date: d.slice(5), value: series.values[i] ?? null }))
  const validValues = series.values.filter((v): v is number => v !== null && v !== undefined)
  const latest = validValues.length ? validValues[validValues.length - 1] : null
  const min = validValues.length ? Math.min(...validValues) : null
  const max = validValues.length ? Math.max(...validValues) : null

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between mb-0.5 gap-2">
        <h3 className="text-xs font-bold tracking-wide uppercase text-slate-600">{meta.title}</h3>
        <div className="flex items-center gap-1.5 shrink-0">
          {series.status !== 'ok' && (
            <span className={`text-[9px] px-1.5 py-0.5 rounded-full uppercase tracking-wide ${STATUS_STYLE[series.status]}`}>
              {series.status}
            </span>
          )}
          {latest !== null && (
            <span className="text-sm font-bold text-slate-800">
              {latest} <span className="text-[10px] font-normal text-slate-400">{series.unit}</span>
            </span>
          )}
        </div>
      </div>
      {min !== null && max !== null && (
        <div className="text-[10px] text-slate-400 mb-2">
          range {min} – {max} {series.unit}
        </div>
      )}
      {validValues.length === 0 ? (
        <div className="h-40 flex items-center justify-center text-xs text-slate-400">No data for this range</div>
      ) : (
        <div style={{ width: '100%', height: 160 }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 5, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={{ stroke: '#e2e8f0' }} />
              <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #e2e8f0' }}
                formatter={(value) => [`${value} ${series.unit}`, meta.title]}
              />
              <Line type="monotone" dataKey="value" stroke={meta.color} strokeWidth={2} dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

function DirectionTable({ variable, series }: { variable: string; series: AnalyticsSeries }) {
  const meta = VARIABLE_META[variable] ?? { title: variable, color: '#64748b' }
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-xs font-bold tracking-wide uppercase text-slate-600">{meta.title}</h3>
        <div className="flex items-center gap-2">
          {series.status !== 'ok' && (
            <span className={`text-[9px] px-1.5 py-0.5 rounded-full uppercase tracking-wide ${STATUS_STYLE[series.status]}`}>
              {series.status}
            </span>
          )}
          <span className="text-[10px] text-slate-400">compass bearing — shown as a table, not a line chart</span>
        </div>
      </div>
      <div className="max-h-48 overflow-y-auto">
        <table className="w-full text-xs">
          <tbody>
            {series.dates.map((d, i) => {
              const v = series.values[i]
              return (
                <tr key={d} className="border-b border-slate-100 last:border-b-0">
                  <td className="py-1 text-slate-500">{d}</td>
                  <td className="py-1 text-right font-mono text-slate-700">{v !== null && v !== undefined ? `${v}°` : '—'}</td>
                  <td className="py-1 pl-2 text-right font-semibold text-slate-800 w-12">
                    {v !== null && v !== undefined ? compassPoint(v) : ''}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function MoonPhaseTable({ series }: { series: AnalyticsSeries }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-xs font-bold tracking-wide uppercase text-slate-600">Moon Phase</h3>
        <span className="text-[10px] text-slate-400">cyclic — shown as a table, not a line chart</span>
      </div>
      <div className="max-h-48 overflow-y-auto">
        <table className="w-full text-xs">
          <tbody>
            {series.dates.map((d, i) => (
              <tr key={d} className="border-b border-slate-100 last:border-b-0">
                <td className="py-1 text-slate-500">{d}</td>
                <td className="py-1 text-right font-semibold text-slate-800">{series.names?.[i] ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function AnalyticsPage() {
  const [zones, setZones] = useState<ZoneRecord[] | null>(null)
  const [zonesError, setZonesError] = useState<string | null>(null)
  const [zoneId, setZoneId] = useState('')
  const [rangeDays, setRangeDays] = useState(7)

  const [data, setData] = useState<AnalyticsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/zones')
      .then(ensureOk)
      .then((res) => res.json() as Promise<{ zones: ZoneRecord[] }>)
      .then((d) => setZones(d.zones))
      .catch((e: unknown) => setZonesError(friendlyError(e)))
  }, [])

  const zoneOptions = useMemo(() => zones?.filter((z) => z.type === 'pfz' && z.coordinates) ?? [], [zones])
  const selectedZone = zoneOptions.find((z) => z.id === zoneId) ?? null
  const lat = selectedZone?.coordinates?.lat
  const lon = selectedZone?.coordinates?.lon

  useEffect(() => {
    if (lat === undefined || lon === undefined) return
    // `cancelled` guards against a slower, now-stale request's response
    // overwriting a newer one (e.g. user re-selects a different zone before
    // the first zone's response has come back) — without this, whichever
    // response resolves last wins regardless of which selection is current.
    let cancelled = false
    const end = new Date()
    const start = new Date()
    start.setDate(start.getDate() - rangeDays)

    fetch(`/api/analytics/historical?lat=${lat}&lon=${lon}&start_date=${isoDate(start)}&end_date=${isoDate(end)}`)
      .then(ensureOk)
      .then((res) => res.json() as Promise<AnalyticsResponse>)
      .then((d) => {
        if (!cancelled) setData(d)
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(friendlyError(e))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [lat, lon, rangeDays])

  function selectZone(id: string) {
    setZoneId(id)
    if (id) {
      setLoading(true)
      setError(null)
      setData(null)
    }
  }

  function selectRange(days: number) {
    setRangeDays(days)
    if (selectedZone) {
      setLoading(true)
      setError(null)
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      {/* Intro block — ported from Mock's .page-hero (badge + H1 + subtitle) */}
      <div className="shrink-0 px-4 sm:px-6 pt-6 pb-2 text-center">
        <span className="inline-flex items-center gap-1.5 text-[11px] tracking-[.06em] uppercase px-2.5 py-1.5 rounded-full bg-teal-500/[.14] border border-teal-500/[.28] text-teal-700">
          Analytics · Smart India Hackathon 2026 · ISRO
        </span>
        <h1 className="mt-2.5 text-[clamp(26px,3.6vw,38px)] leading-none tracking-[-.03em] text-[#0d0c0b] font-normal">
          Analytics Dashboard
        </h1>
        <p className="mt-2.5 text-sm leading-relaxed text-[#0d0c0b]/64 max-w-[52ch] mx-auto">
          Historical trends for sea surface temperature, waves, currents, and more.
        </p>
      </div>

      <div className="max-w-5xl mx-auto px-4 pb-8">
      <div className="rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg p-4 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-white/40">
          <h2 className="text-xs font-semibold tracking-widest uppercase text-slate-600">Historical Trends</h2>
          <div className="flex items-center gap-1 bg-slate-100 rounded-full p-0.5">
            {RANGE_OPTIONS.map((r) => (
              <button
                key={r.days}
                onClick={() => selectRange(r.days)}
                className={`text-[11px] px-3 py-1 rounded-full transition ${
                  rangeDays === r.days ? 'bg-white shadow text-slate-800 font-semibold' : 'text-slate-500'
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>

        {zonesError && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
            Couldn't load zones: {zonesError}{devBackendHint()}
          </div>
        )}
        {error && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
            Couldn't load historical data: {error}
          </div>
        )}

        <div className="rounded-xl border border-slate-200 bg-white p-3">
          <select
            value={zoneId}
            onChange={(e) => selectZone(e.target.value)}
            className="w-full sm:w-96 rounded-lg border border-slate-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
          >
            <option value="">{zones ? 'Select a fishing zone…' : 'Loading zones…'}</option>
            {zoneOptions.map((z) => (
              <option key={z.id} value={z.id}>
                {z.name} — near {z.near}
              </option>
            ))}
          </select>
        </div>

        {!selectedZone && <div className="text-xs text-slate-400 px-1">Pick a zone to see its historical trends.</div>}

        {selectedZone && loading && !data && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="rounded-xl border border-slate-200 bg-white p-4 h-48 flex items-center justify-center text-xs text-slate-400">
                Loading…
              </div>
            ))}
          </div>
        )}

        {data && (
          <div className="space-y-6">
            {CATEGORIES.map((c) => {
              const hasData = c.variables.some((v) => data.series[v])
              if (!hasData) return null
              return (
                <div key={c.key}>
                  <h2 className="text-[11px] font-semibold tracking-widest uppercase text-slate-500 mb-2 px-1">{c.label}</h2>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {c.variables.map((variable) => {
                      const series = data.series[variable]
                      if (!series) return null
                      if (variable === 'moon_phase') return <MoonPhaseTable key={variable} series={series} />
                      if (DIRECTIONAL_VARIABLES.has(variable)) return <DirectionTable key={variable} variable={variable} series={series} />
                      if (NON_CHART_VARIABLES.has(variable)) return null
                      return <ChartCard key={variable} variable={variable} series={series} />
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {data && (
          <div className="text-[10px] text-slate-400 text-center pt-1">
            {data.start_date} to {data.end_date} · {data.lat.toFixed(4)}, {data.lon.toFixed(4)}
          </div>
        )}
      </div>
      </div>
    </div>
  )
}
