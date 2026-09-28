import { useEffect, useMemo, useState } from 'react'
import type { ZoneRecord } from '../types'

const VARIABLE_GROUPS: { title: string; variables: string[] }[] = [
  { title: 'Temperature', variables: ['sst_celsius', 'air_temp_celsius'] },
  {
    title: 'Waves',
    variables: ['wave_height_m', 'wave_direction_deg', 'wave_period_s', 'swell_wave_height_m', 'swell_wave_direction_deg', 'swell_wave_period_s'],
  },
  { title: 'Currents', variables: ['current_velocity_kmh', 'current_direction_deg'] },
  { title: 'Wind', variables: ['wind_kmh', 'wind_direction_deg', 'wind_gusts_kmh'] },
  { title: 'Atmospheric', variables: ['precipitation_mm', 'cloud_cover_pct', 'pressure_hpa', 'humidity_pct'] },
  { title: 'Biological / Water Quality', variables: ['chlorophyll_mg_m3', 'salinity_psu', 'dissolved_oxygen_mmol_m3'] },
  { title: 'Astronomical', variables: ['sunrise_hour_ist', 'sunset_hour_ist', 'moon_phase'] },
]

const VARIABLE_LABELS: Record<string, string> = {
  sst_celsius: 'Sea Surface Temperature',
  air_temp_celsius: 'Air Temperature',
  wave_height_m: 'Wave Height',
  wave_direction_deg: 'Wave Direction',
  wave_period_s: 'Wave Period',
  swell_wave_height_m: 'Swell Wave Height',
  swell_wave_direction_deg: 'Swell Wave Direction',
  swell_wave_period_s: 'Swell Wave Period',
  current_velocity_kmh: 'Current Velocity',
  current_direction_deg: 'Current Direction',
  wind_kmh: 'Wind Speed',
  wind_direction_deg: 'Wind Direction',
  wind_gusts_kmh: 'Wind Gusts',
  precipitation_mm: 'Precipitation',
  cloud_cover_pct: 'Cloud Cover',
  pressure_hpa: 'Sea-Level Pressure',
  humidity_pct: 'Relative Humidity',
  chlorophyll_mg_m3: 'Chlorophyll-a',
  salinity_psu: 'Salinity',
  dissolved_oxygen_mmol_m3: 'Dissolved Oxygen',
  sunrise_hour_ist: 'Sunrise (IST)',
  sunset_hour_ist: 'Sunset (IST)',
  moon_phase: 'Moon Phase',
}

const ALL_VARIABLES = VARIABLE_GROUPS.flatMap((g) => g.variables)

const QUICK_RANGES = [
  { label: '7 Days', days: 7 },
  { label: '30 Days', days: 30 },
  { label: '90 Days', days: 90 },
]

function isoDate(d: Date): string {
  return d.toISOString().slice(0, 10)
}

function daysAgoIso(days: number): string {
  return isoDate(new Date(Date.now() - days * 24 * 60 * 60 * 1000))
}

export function DownloadPage() {
  const [zones, setZones] = useState<ZoneRecord[] | null>(null)
  const [zonesError, setZonesError] = useState<string | null>(null)
  const [zoneId, setZoneId] = useState('')

  const [startDate, setStartDate] = useState(() => daysAgoIso(7))
  const [endDate, setEndDate] = useState(() => isoDate(new Date()))

  const [selectedVars, setSelectedVars] = useState<Set<string>>(new Set(ALL_VARIABLES))
  const [format, setFormat] = useState<'csv' | 'json' | 'pdf' | 'docx'>('csv')
  const [exportError, setExportError] = useState<string | null>(null)
  const [exporting, setExporting] = useState(false)

  useEffect(() => {
    fetch('/api/zones')
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
        return res.json() as Promise<{ zones: ZoneRecord[] }>
      })
      .then((d) => setZones(d.zones))
      .catch((e: unknown) => setZonesError(e instanceof Error ? e.message : String(e)))
  }, [])

  const zoneOptions = useMemo(() => zones?.filter((z) => z.type === 'pfz' && z.coordinates) ?? [], [zones])

  function toggleVariable(v: string) {
    setSelectedVars((prev) => {
      const next = new Set(prev)
      if (next.has(v)) next.delete(v)
      else next.add(v)
      return next
    })
  }

  function selectAllVariables() {
    setSelectedVars(new Set(ALL_VARIABLES))
  }

  function clearAllVariables() {
    setSelectedVars(new Set())
  }

  function applyQuickRange(days: number) {
    setStartDate(daysAgoIso(days))
    setEndDate(isoDate(new Date()))
  }

  const canExport = Boolean(zoneId) && Boolean(startDate) && Boolean(endDate) && startDate <= endDate && selectedVars.size > 0

  async function handleExport(exportFormat: 'csv' | 'json' | 'pdf' | 'docx') {
    if (!canExport) return
    setExportError(null)
    setExporting(true)
    const params = new URLSearchParams({
      zone: zoneId,
      start_date: startDate,
      end_date: endDate,
      variables: Array.from(selectedVars).join(','),
      format: exportFormat,
    })
    try {
      const res = await fetch(`/api/export?${params.toString()}`)
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`${res.status} ${txt}`)
      }
      const blob = await res.blob()
      const disposition = res.headers.get('Content-Disposition') ?? ''
      const match = /filename="?([^"]+)"?/.exec(disposition)
      const filename = match ? match[1] : `orca-export.${exportFormat}`
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e: unknown) {
      setExportError(e instanceof Error ? e.message : String(e))
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      {/* Intro block — ported from Mock's .page-hero (badge + H1 + subtitle) */}
      <div className="shrink-0 px-4 sm:px-6 pt-6 pb-2 text-center">
        <span className="inline-flex items-center gap-1.5 text-[11px] tracking-[.06em] uppercase px-2.5 py-1.5 rounded-full bg-teal-500/[.14] border border-teal-500/[.28] text-teal-700">
          Download · Smart India Hackathon 2026 · ISRO
        </span>
        <h1 className="mt-2.5 text-[clamp(26px,3.6vw,38px)] leading-none tracking-[-.03em] text-[#0d0c0b] font-normal">
          Download
        </h1>
        <p className="mt-2.5 text-sm leading-relaxed text-[#0d0c0b]/64 max-w-[52ch] mx-auto">
          Export ocean variables for a chosen zone and date range.
        </p>
      </div>

      <div className="max-w-3xl mx-auto px-4 pb-8">
      <div className="rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg p-4 space-y-4">
        {zonesError && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
            Couldn't load zones: {zonesError}. Is the backend running on :8000?
          </div>
        )}
        {exportError && (
          <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
            Export failed: {exportError}
          </div>
        )}

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-500 mb-1.5">Zone</div>
          <select
            value={zoneId}
            onChange={(e) => setZoneId(e.target.value)}
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
          >
            <option value="">{zones ? 'Select a fishing zone…' : 'Loading zones…'}</option>
            {zoneOptions.map((z) => (
              <option key={z.id} value={z.id}>
                {z.name} — near {z.near}
              </option>
            ))}
          </select>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between mb-1.5">
            <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-500">Date Range</div>
            <div className="flex gap-1">
              {QUICK_RANGES.map((r) => (
                <button
                  key={r.days}
                  onClick={() => applyQuickRange(r.days)}
                  className="text-[10px] px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200"
                >
                  {r.label}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              max={endDate}
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            />
            <span className="text-slate-400 text-xs">to</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              min={startDate}
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            />
          </div>
          {startDate > endDate && <p className="text-[11px] text-rose-600 mt-1.5">Start date must be before end date.</p>}
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-500">
              Variables ({selectedVars.size}/{ALL_VARIABLES.length})
            </div>
            <div className="flex gap-1">
              <button onClick={selectAllVariables} className="text-[10px] px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200">
                Select all
              </button>
              <button onClick={clearAllVariables} className="text-[10px] px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200">
                Clear
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {VARIABLE_GROUPS.map((group) => (
              <div key={group.title}>
                <div className="text-xs font-semibold text-slate-600 mb-1.5">{group.title}</div>
                <div>
                  {group.variables.map((v) => (
                    <label key={v} className="flex items-center gap-2 py-1.5 text-xs text-slate-700 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={selectedVars.has(v)}
                        onChange={() => toggleVariable(v)}
                        className="w-4 h-4 shrink-0 rounded border-slate-300 text-sky-600 focus:ring-sky-500"
                      />
                      {VARIABLE_LABELS[v]}
                    </label>
                  ))}
                </div>
              </div>
            ))}
          </div>
          {selectedVars.size === 0 && <p className="text-[11px] text-rose-600 mt-2">Select at least one variable.</p>}
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-500 mb-1.5">Format</div>
          <div className="flex items-center gap-1 bg-slate-100 rounded-full p-0.5 w-fit">
            {(['csv', 'json', 'pdf', 'docx'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFormat(f)}
                className={`text-[11px] px-3.5 py-1 rounded-full transition uppercase ${
                  format === f ? 'bg-white shadow text-slate-800 font-semibold' : 'text-slate-500'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={() => handleExport(format)}
          disabled={!canExport || exporting}
          className="w-full rounded-lg bg-slate-900 text-white px-4 py-2.5 text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-black transition"
        >
          {exporting ? 'Exporting…' : `⬇ Export ${format.toUpperCase()}`}
        </button>

        <p className="text-[10px] text-slate-400 text-center">
          Some variables are derived, computed, or have limited coverage (e.g. current direction, sunrise/sunset,
          salinity near the coast) — every downloaded file documents this itself, including which of your selected
          variables actually came back with data for this range.
        </p>
      </div>
      </div>
    </div>
  )
}
