import { useMemo, useState, useEffect } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ForecastResponse, WeatherSnapshot, ZoneRecord } from '../types'

// Layout — a single-zone live-conditions card, a real 7-day trend chart,
// a "Run ML Forecast" button, and a separate visually-distinct ML section
// below it — mirrors A.T.M.O.S.'s own page structure (frontend/index.html:
// #weatherSection's cards-grid -> trend chart -> predict-section, then a
// SEPARATE #predictSection below). "Compare Two Zones" sits at the bottom
// as its own section (ATMOS's #compareSection), not a top toggle switching
// the whole view. No ATMOS CSS/JS or model files were copied — the ML
// section calls ORCA's own trained models (see the phase report for why
// ATMOS's Bangalore-only models weren't reused).

type LocationOption = { id: string; label: string; lat: number; lon: number }

function zonesToOptions(zones: ZoneRecord[]): LocationOption[] {
  return zones
    .filter((z) => z.coordinates)
    .map((z) => ({
      id: z.id,
      label: z.type === 'pfz' ? `${z.name} — near ${z.near ?? 'unknown'}` : `${z.name} (restricted)`,
      lat: z.coordinates!.lat,
      lon: z.coordinates!.lon,
    }))
}

async function fetchWeather(lat: number, lon: number): Promise<WeatherSnapshot> {
  const res = await fetch(`/api/weather?lat=${lat}&lon=${lon}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<WeatherSnapshot>
}

async function fetchTrend(lat: number, lon: number): Promise<{ date: string; value: number | null }[]> {
  const end = new Date()
  const start = new Date(Date.now() - 6 * 24 * 60 * 60 * 1000)
  const iso = (d: Date) => d.toISOString().slice(0, 10)
  const res = await fetch(`/api/analytics/historical?lat=${lat}&lon=${lon}&start_date=${iso(start)}&end_date=${iso(end)}&variables=wave_height_m`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  const data = await res.json()
  const series = data.series?.wave_height_m
  if (!series) return []
  return series.dates.map((d: string, i: number) => ({ date: d.slice(5), value: series.values[i] ?? null }))
}

function formatHour(h: number | null): string {
  if (h === null || h === undefined) return 'unavailable'
  const hh = Math.floor(h)
  const mm = Math.round((h - hh) * 60)
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')} IST`
}

function LocationSelect({
  options,
  value,
  onChange,
  placeholder,
}: {
  options: LocationOption[]
  value: string
  onChange: (id: string) => void
  placeholder: string
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="flex-1 min-w-0 rounded-lg border border-slate-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
    >
      <option value="">{placeholder}</option>
      {options.map((o) => (
        <option key={o.id} value={o.id}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-slate-100 last:border-b-0 text-xs">
      <span className="text-slate-500">{label}</span>
      <span className="font-semibold text-slate-800">{value}</span>
    </div>
  )
}

function WeatherCard({ title, snapshot, loading }: { title: string; snapshot: WeatherSnapshot | null; loading: boolean }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2 mb-2">
        <h3 className="text-sm font-bold text-slate-800">{title}</h3>
        {snapshot && (
          <span
            className={`text-[9px] px-1.5 py-0.5 rounded-full uppercase tracking-wide ${
              snapshot.status === 'ok'
                ? 'bg-emerald-100 text-emerald-700'
                : snapshot.status === 'partial'
                  ? 'bg-amber-100 text-amber-700'
                  : 'bg-rose-100 text-rose-700'
            }`}
          >
            {snapshot.status}
          </span>
        )}
      </div>

      {loading && <div className="text-xs text-slate-400 py-4 text-center">Fetching live conditions…</div>}

      {!loading && !snapshot && <div className="text-xs text-slate-400 py-4 text-center">Pick a zone and fetch conditions.</div>}

      {!loading && snapshot && (
        <>
          {(snapshot.cyclone_alert || snapshot.lightning_alert) && (
            <div className="flex gap-1.5 mb-2 flex-wrap">
              {snapshot.cyclone_alert && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-rose-100 text-rose-700 font-semibold">⛈ Cyclone alert</span>
              )}
              {snapshot.lightning_alert && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 font-semibold">⚡ Lightning alert</span>
              )}
            </div>
          )}
          <div>
            <Row label="Air Temperature" value={snapshot.air_temperature_c !== null ? `${snapshot.air_temperature_c}°C` : 'unavailable'} />
            <Row label="Sea Surface Temp" value={snapshot.sst_celsius !== null ? `${snapshot.sst_celsius}°C` : 'unavailable'} />
            <Row label="Wind Speed" value={snapshot.wind_kmh !== null ? `${snapshot.wind_kmh} km/h` : 'unavailable'} />
            <Row label="Max Wind (today)" value={snapshot.wind_max_kmh !== null ? `${snapshot.wind_max_kmh} km/h` : 'unavailable'} />
            <Row label="Wave Height" value={snapshot.wave_height_m !== null ? `${snapshot.wave_height_m} m` : 'unavailable'} />
            <Row label="Precipitation (today)" value={snapshot.precipitation_mm !== null ? `${snapshot.precipitation_mm} mm` : 'unavailable'} />
            <Row label="Sunrise" value={formatHour(snapshot.sunrise_hour_ist)} />
            <Row label="Sunset" value={formatHour(snapshot.sunset_hour_ist)} />
            <Row label="Chlorophyll-a" value={snapshot.chlorophyll_mg_m3 !== null ? `${snapshot.chlorophyll_mg_m3} mg/m³` : 'unavailable'} />
          </div>
          <div className="text-[10px] text-slate-400 mt-2 font-mono">
            {snapshot.lat.toFixed(4)}, {snapshot.lon.toFixed(4)} · as of {new Date(snapshot.source_timestamp).toLocaleTimeString()}
          </div>
        </>
      )}
    </div>
  )
}

function StatTile({ icon, label, value }: { icon: string; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-center gap-1.5 text-[10px] font-semibold tracking-wide uppercase text-slate-500">
        <span aria-hidden>{icon}</span>
        {label}
      </div>
      <div className="text-base font-bold text-slate-800 mt-1">{value}</div>
    </div>
  )
}

// Single-zone live-conditions display only — a grid of small stat tiles
// instead of WeatherCard's stacked Row list. Deliberately a SEPARATE
// component rather than a variant of WeatherCard: WeatherCard is also
// used side-by-side in "Compare Two Zones" below, which stays untouched.
function WeatherStatsCard({ title, snapshot, loading }: { title: string; snapshot: WeatherSnapshot | null; loading: boolean }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2 mb-3">
        <h3 className="text-sm font-bold text-slate-800">{title}</h3>
        {snapshot && (
          <span
            className={`text-[9px] px-1.5 py-0.5 rounded-full uppercase tracking-wide ${
              snapshot.status === 'ok'
                ? 'bg-emerald-100 text-emerald-700'
                : snapshot.status === 'partial'
                  ? 'bg-amber-100 text-amber-700'
                  : 'bg-rose-100 text-rose-700'
            }`}
          >
            {snapshot.status}
          </span>
        )}
      </div>

      {loading && <div className="text-xs text-slate-400 py-4 text-center">Fetching live conditions…</div>}

      {!loading && !snapshot && <div className="text-xs text-slate-400 py-4 text-center">Pick a zone and fetch conditions.</div>}

      {!loading && snapshot && (
        <>
          {(snapshot.cyclone_alert || snapshot.lightning_alert) && (
            <div className="flex gap-1.5 mb-3 flex-wrap">
              {snapshot.cyclone_alert && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-rose-100 text-rose-700 font-semibold">⛈ Cyclone alert</span>
              )}
              {snapshot.lightning_alert && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 font-semibold">⚡ Lightning alert</span>
              )}
            </div>
          )}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
            <StatTile icon="🌡️" label="Air Temperature" value={snapshot.air_temperature_c !== null ? `${snapshot.air_temperature_c}°C` : 'unavailable'} />
            <StatTile icon="🌊" label="Sea Surface Temp" value={snapshot.sst_celsius !== null ? `${snapshot.sst_celsius}°C` : 'unavailable'} />
            <StatTile icon="💨" label="Wind Speed" value={snapshot.wind_kmh !== null ? `${snapshot.wind_kmh} km/h` : 'unavailable'} />
            <StatTile icon="🌬️" label="Max Wind (today)" value={snapshot.wind_max_kmh !== null ? `${snapshot.wind_max_kmh} km/h` : 'unavailable'} />
            <StatTile icon="🌊" label="Wave Height" value={snapshot.wave_height_m !== null ? `${snapshot.wave_height_m} m` : 'unavailable'} />
            <StatTile icon="🌧️" label="Precipitation (today)" value={snapshot.precipitation_mm !== null ? `${snapshot.precipitation_mm} mm` : 'unavailable'} />
            <StatTile icon="🌅" label="Sunrise" value={formatHour(snapshot.sunrise_hour_ist)} />
            <StatTile icon="🌇" label="Sunset" value={formatHour(snapshot.sunset_hour_ist)} />
            <StatTile icon="🧪" label="Chlorophyll-a" value={snapshot.chlorophyll_mg_m3 !== null ? `${snapshot.chlorophyll_mg_m3} mg/m³` : 'unavailable'} />
          </div>
          <div className="text-[10px] text-slate-400 mt-3 font-mono">
            {snapshot.lat.toFixed(4)}, {snapshot.lon.toFixed(4)} · as of {new Date(snapshot.source_timestamp).toLocaleTimeString()}
          </div>
        </>
      )}
    </div>
  )
}

function TrendChart({ data, loading }: { data: { date: string; value: number | null }[]; loading: boolean }) {
  const validValues = data.map((d) => d.value).filter((v): v is number => v !== null)
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="text-xs font-bold tracking-wide uppercase text-slate-600 mb-2">7-Day Wave Height Trend</h3>
      {loading && <div className="h-40 flex items-center justify-center text-xs text-slate-400">Loading trend…</div>}
      {!loading && validValues.length === 0 && (
        <div className="h-40 flex items-center justify-center text-xs text-slate-400">No historical data for this zone</div>
      )}
      {!loading && validValues.length > 0 && (
        <div style={{ width: '100%', height: 160 }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 5, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={{ stroke: '#e2e8f0' }} />
              <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #e2e8f0' }} formatter={(v) => [`${v} m`, 'Wave Height']} />
              <Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

function ForecastSection({ forecast, loading }: { forecast: ForecastResponse | null; loading: boolean }) {
  if (!loading && !forecast) return null
  return (
    <div className="rounded-xl border-2 border-dashed border-indigo-300 bg-indigo-50/60 p-4">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-600 text-white font-bold uppercase tracking-wide">Predicted</span>
        <h3 className="text-xs font-bold tracking-wide uppercase text-indigo-800">ML Forecast — Next 7 Days</h3>
      </div>
      <p className="text-[10px] text-indigo-600 mb-3">
        Model output from ORCA's own trained forecast models — not live data. See below for the exact model.
      </p>

      {loading && <div className="text-xs text-indigo-500 py-4 text-center">Running forecast model…</div>}

      {!loading && forecast?.status === 'error' && (
        <div className="text-xs text-rose-600 py-2">Forecast unavailable: {forecast.reason}</div>
      )}

      {!loading && forecast?.status === 'ok' && (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-indigo-500 border-b border-indigo-200">
                  <th className="py-1 pr-3 font-semibold">Date</th>
                  <th className="py-1 pr-3 font-semibold">Wave Height</th>
                  <th className="py-1 pr-3 font-semibold">Wind Speed</th>
                </tr>
              </thead>
              <tbody>
                {forecast.forecast.map((f) => (
                  <tr key={f.horizon} className="border-b border-indigo-100 last:border-b-0">
                    <td className="py-1.5 pr-3 text-slate-700">
                      {f.date} <span className="text-indigo-400">(+{f.horizon}d)</span>
                    </td>
                    <td className="py-1.5 pr-3 font-semibold text-slate-800">
                      {f.wave_height_m} m {f.wave_height_mae !== null && <span className="text-[10px] text-slate-400 font-normal">±{f.wave_height_mae}</span>}
                    </td>
                    <td className="py-1.5 pr-3 font-semibold text-slate-800">
                      {f.wind_kmh} km/h {f.wind_kmh_mae !== null && <span className="text-[10px] text-slate-400 font-normal">±{f.wind_kmh_mae}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-[10px] text-indigo-500 mt-3">
            Based on conditions as of {forecast.based_on_date}. ± values are the model's own validation-set mean absolute error per day.
          </div>
          <div className="text-[10px] text-slate-400 mt-1">{forecast.model}</div>
        </>
      )}
    </div>
  )
}

export function WeatherPage() {
  const [zones, setZones] = useState<ZoneRecord[] | null>(null)
  const [zonesError, setZonesError] = useState<string | null>(null)

  const [zoneA, setZoneA] = useState('')
  const [snapshotA, setSnapshotA] = useState<WeatherSnapshot | null>(null)
  const [trend, setTrend] = useState<{ date: string; value: number | null }[]>([])
  const [loading, setLoading] = useState(false)
  const [trendLoading, setTrendLoading] = useState(false)
  const [fetchError, setFetchError] = useState<string | null>(null)

  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [forecastLoading, setForecastLoading] = useState(false)

  const [compareA, setCompareA] = useState('')
  const [compareB, setCompareB] = useState('')
  const [compareSnapA, setCompareSnapA] = useState<WeatherSnapshot | null>(null)
  const [compareSnapB, setCompareSnapB] = useState<WeatherSnapshot | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareError, setCompareError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/zones')
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
        return res.json() as Promise<{ zones: ZoneRecord[] }>
      })
      .then((d) => setZones(d.zones))
      .catch((e: unknown) => setZonesError(e instanceof Error ? e.message : String(e)))
  }, [])

  const options = useMemo(() => (zones ? zonesToOptions(zones) : []), [zones])

  async function handleFetch() {
    const a = options.find((o) => o.id === zoneA)
    if (!a) return
    setLoading(true)
    setTrendLoading(true)
    setFetchError(null)
    setForecast(null)
    try {
      const snap = await fetchWeather(a.lat, a.lon)
      setSnapshotA(snap)
      setLoading(false)
      const trendData = await fetchTrend(a.lat, a.lon)
      setTrend(trendData)
    } catch (e: unknown) {
      setFetchError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
      setTrendLoading(false)
    }
  }

  async function handleRunForecast() {
    if (!zoneA) return
    setForecastLoading(true)
    try {
      const res = await fetch(`/api/weather/forecast?zone=${encodeURIComponent(zoneA)}`)
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      setForecast((await res.json()) as ForecastResponse)
    } catch (e: unknown) {
      setForecast({ zone: zoneA, lat: 0, lon: 0, status: 'error', reason: e instanceof Error ? e.message : String(e), forecast: [], generated_at: '' })
    } finally {
      setForecastLoading(false)
    }
  }

  async function handleCompare() {
    const a = options.find((o) => o.id === compareA)
    const b = options.find((o) => o.id === compareB)
    if (!a || !b) return
    setCompareLoading(true)
    setCompareError(null)
    try {
      const [resA, resB] = await Promise.all([fetchWeather(a.lat, a.lon), fetchWeather(b.lat, b.lon)])
      setCompareSnapA(resA)
      setCompareSnapB(resB)
    } catch (e: unknown) {
      setCompareError(e instanceof Error ? e.message : String(e))
    } finally {
      setCompareLoading(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      {/* Intro block — ported from Mock's .page-hero (badge + H1 + subtitle) */}
      <div className="shrink-0 px-4 sm:px-6 pt-6 pb-2 text-center">
        <span className="inline-flex items-center gap-1.5 text-[11px] tracking-[.06em] uppercase px-2.5 py-1.5 rounded-full bg-teal-500/[.14] border border-teal-500/[.28] text-teal-700">
          Weather · Smart India Hackathon 2026 · ISRO
        </span>
        <h1 className="mt-2.5 text-[clamp(26px,3.6vw,38px)] leading-none tracking-[-.03em] text-[#0d0c0b] font-normal">
          Weather
        </h1>
        <p className="mt-2.5 text-sm leading-relaxed text-[#0d0c0b]/64 max-w-[52ch] mx-auto">
          Check conditions for a single zone, or compare two zones side by side.
        </p>
      </div>

      <div className="max-w-4xl mx-auto px-4 pb-8">
      <div className="rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg p-4 space-y-4">
        {zonesError && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
            Couldn't load zones: {zonesError}. Is the backend running on :8000?
          </div>
        )}
        {fetchError && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
            Couldn't fetch weather: {fetchError}.
          </div>
        )}

        {/* Single zone */}
        <div className="rounded-xl border border-slate-200 bg-white p-3 flex flex-wrap items-center gap-2">
          <LocationSelect options={options} value={zoneA} onChange={setZoneA} placeholder={zones ? 'Select a zone…' : 'Loading zones…'} />
          <button
            onClick={() => void handleFetch()}
            disabled={!zoneA || loading}
            className="shrink-0 rounded-lg bg-slate-900 text-white px-4 py-2 text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-black transition"
          >
            {loading ? 'Fetching…' : 'Get Weather'}
          </button>
        </div>

        <WeatherStatsCard title={options.find((o) => o.id === zoneA)?.label ?? 'Weather'} snapshot={snapshotA} loading={loading} />

        {snapshotA && <TrendChart data={trend} loading={trendLoading} />}

        {snapshotA && (
          <div className="flex justify-center">
            <button
              onClick={() => void handleRunForecast()}
              disabled={forecastLoading}
              className="rounded-full bg-indigo-600 text-white px-6 py-2.5 text-sm font-semibold disabled:opacity-40 hover:bg-indigo-700 transition"
            >
              {forecastLoading ? 'Running model…' : '🔮 Run ML Forecast'}
            </button>
          </div>
        )}

        <ForecastSection forecast={forecast} loading={forecastLoading} />

        {/* Compare — bottom-of-page section, matching ATMOS's own layout */}
        <div className="pt-6 border-t border-slate-200">
          <h2 className="text-xs font-semibold tracking-widest uppercase text-slate-600 mb-3">Compare Two Zones</h2>
          {compareError && (
            <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2 mb-3">
              Couldn't fetch weather: {compareError}.
            </div>
          )}
          <div className="rounded-xl border border-slate-200 bg-white p-3 flex flex-col sm:flex-row sm:flex-wrap sm:items-center gap-2 mb-4">
            <LocationSelect options={options} value={compareA} onChange={setCompareA} placeholder={zones ? 'Select a zone…' : 'Loading zones…'} />
            <span className="text-xs text-slate-400 shrink-0 text-center sm:text-left">vs</span>
            <LocationSelect options={options} value={compareB} onChange={setCompareB} placeholder={zones ? 'Select a zone…' : 'Loading zones…'} />
            <button
              onClick={() => void handleCompare()}
              disabled={!compareA || !compareB || compareLoading}
              className="w-full sm:w-auto shrink-0 rounded-lg bg-slate-900 text-white px-4 py-2 text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-black transition"
            >
              {compareLoading ? 'Fetching…' : 'Compare'}
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <WeatherCard title={options.find((o) => o.id === compareA)?.label ?? 'Zone A'} snapshot={compareSnapA} loading={compareLoading} />
            <WeatherCard title={options.find((o) => o.id === compareB)?.label ?? 'Zone B'} snapshot={compareSnapB} loading={compareLoading} />
          </div>
        </div>
      </div>
      </div>
    </div>
  )
}
