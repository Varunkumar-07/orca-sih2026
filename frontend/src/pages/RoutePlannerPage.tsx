import { useEffect, useMemo, useState } from 'react'
import { MapView } from '../components/MapView'
import type { MapPayload, RouteResponse, ZoneRecord } from '../types'
import { ensureOk, friendlyError, devBackendHint } from '../lib/httpError'

function getUserLocation(): Promise<{ lat: number; lon: number } | null> {
  if (typeof navigator === 'undefined' || !navigator.geolocation) return Promise.resolve(null)
  return new Promise((resolve) => {
    let settled = false
    const timer = window.setTimeout(() => {
      if (!settled) {
        settled = true
        resolve(null)
      }
    }, 5000)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude })
      },
      () => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        resolve(null)
      },
      { enableHighAccuracy: false, timeout: 4000, maximumAge: 60000 },
    )
  })
}

export function RoutePlannerPage() {
  const [zones, setZones] = useState<ZoneRecord[] | null>(null)
  const [zonesError, setZonesError] = useState<string | null>(null)

  const [startLat, setStartLat] = useState('13.08')
  const [startLon, setStartLon] = useState('80.27')
  const [destinationZoneId, setDestinationZoneId] = useState('')
  const [locating, setLocating] = useState(false)

  const [result, setResult] = useState<RouteResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetch('/api/zones')
      .then(ensureOk)
      .then((res) => res.json() as Promise<{ zones: ZoneRecord[] }>)
      .then((data) => {
        if (!cancelled) setZones(data.zones)
      })
      .catch((e: unknown) => {
        if (!cancelled) setZonesError(friendlyError(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Only PFZ entries carry a point `coordinates` field — a restricted area
  // has a polygon `geometry` instead and isn't a valid routing destination.
  const destinationOptions = useMemo(() => zones?.filter((z) => z.type === 'pfz' && z.coordinates) ?? [], [zones])
  const restrictedZones = useMemo(() => zones?.filter((z) => z.type === 'restricted') ?? [], [zones])
  const destinationZone = destinationOptions.find((z) => z.id === destinationZoneId) ?? null

  async function handleUseMyLocation() {
    setLocating(true)
    const coords = await getUserLocation()
    setLocating(false)
    if (coords) {
      setStartLat(coords.lat.toFixed(4))
      setStartLon(coords.lon.toFixed(4))
    } else {
      setError('Could not get your location — check browser permissions, or enter coordinates manually.')
    }
  }

  async function handlePlanRoute() {
    const lat = parseFloat(startLat)
    const lon = parseFloat(startLon)
    if (Number.isNaN(lat) || Number.isNaN(lon) || !destinationZoneId) return

    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch('/api/route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start: { lat, lon }, destination_zone_id: destinationZoneId }),
      })
      await ensureOk(res)
      setResult((await res.json()) as RouteResponse)
    } catch (e: unknown) {
      setError(friendlyError(e))
    } finally {
      setLoading(false)
    }
  }

  const mapPayload: MapPayload | null = useMemo(() => {
    if (!zones) return null
    const overlays: MapPayload['overlays'] = restrictedZones
      .filter((z) => z.geometry)
      .map((z) => ({ type: 'restricted_area', name: z.id, geojson: z.geometry!, highlighted: false }))

    const pins: MapPayload['pins'] = []
    const lat = parseFloat(startLat)
    const lon = parseFloat(startLon)
    if (!Number.isNaN(lat) && !Number.isNaN(lon)) {
      pins.push({ lat, lon, label: 'Start', type: 'query' })
    }
    if (destinationZone?.coordinates) {
      pins.push({
        lat: destinationZone.coordinates.lat,
        lon: destinationZone.coordinates.lon,
        label: destinationZone.name,
        type: 'pfz',
        zone_id: destinationZone.id,
      })
    }

    return { pins, overlays, route: result?.route ?? null }
  }, [zones, restrictedZones, startLat, startLon, destinationZone, result])

  const canPlan = !Number.isNaN(parseFloat(startLat)) && !Number.isNaN(parseFloat(startLon)) && Boolean(destinationZoneId)

  return (
    <div className="h-full flex flex-col">
      {/* Intro block — ported from Mock's .page-hero (badge + H1 + subtitle) */}
      <div className="shrink-0 px-4 sm:px-6 pt-6 pb-2 text-center">
        <span className="inline-flex items-center gap-1.5 text-[11px] tracking-[.06em] uppercase px-2.5 py-1.5 rounded-full bg-teal-500/[.14] border border-teal-500/[.28] text-teal-700">
          Route Planner · Smart India Hackathon 2026 · ISRO
        </span>
        <h1 className="mt-2.5 text-[clamp(26px,3.6vw,38px)] leading-none tracking-[-.03em] text-[#0d0c0b] font-normal">
          Route Planner
        </h1>
        <p className="mt-2.5 text-sm leading-relaxed text-[#0d0c0b]/64 max-w-[52ch] mx-auto">
          Plan a sea route to a fishing zone that avoids land and protected areas.
        </p>
      </div>

      <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-y-auto lg:overflow-hidden gap-3 p-3 pb-6">
        {/* Map — glass card matching the Assistant/Zones pages' panels */}
        <div className="flex-1 min-w-0 min-h-[320px] lg:min-h-0 rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg flex flex-col overflow-hidden">
          <div className="px-4 py-2 border-b border-white/40 bg-white/15 flex items-center justify-between shrink-0">
            <h2 className="text-xs font-semibold tracking-widest uppercase text-slate-600">Route Planner · Sea Route A*</h2>
            <span className="text-[11px] text-slate-500">{restrictedZones.length} restricted areas avoided</span>
          </div>
          <div className="flex-1 min-h-[300px]">
            <MapView payload={mapPayload} />
          </div>
        </div>

        <div className="w-full lg:w-[340px] shrink-0 rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg min-h-0 overflow-y-auto p-3 space-y-3">
          {zonesError && (
            <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
              Couldn't load zones: {zonesError}{devBackendHint()}
            </div>
          )}
          {error && (
            <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">{error}</div>
          )}

          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-3">
            <div>
              <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-500 mb-1.5">Start</div>
              <div className="flex gap-2">
                <input
                  value={startLat}
                  onChange={(e) => setStartLat(e.target.value)}
                  placeholder="lat"
                  inputMode="decimal"
                  className="w-1/2 rounded-lg border border-slate-300 px-3 py-2.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
                />
                <input
                  value={startLon}
                  onChange={(e) => setStartLon(e.target.value)}
                  placeholder="lon"
                  inputMode="decimal"
                  className="w-1/2 rounded-lg border border-slate-300 px-3 py-2.5 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
                />
              </div>
              <button
                onClick={() => void handleUseMyLocation()}
                disabled={locating}
                className="mt-1.5 py-1.5 text-[13px] text-sky-600 hover:text-sky-800 disabled:opacity-50"
              >
                {locating ? 'Locating…' : '📍 Use my location'}
              </button>
            </div>

            <div>
              <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-500 mb-1.5">Destination (PFZ)</div>
              <select
                value={destinationZoneId}
                onChange={(e) => setDestinationZoneId(e.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
              >
                <option value="">{zones ? 'Select a fishing zone…' : 'Loading zones…'}</option>
                {destinationOptions.map((z) => (
                  <option key={z.id} value={z.id}>
                    {z.name} — near {z.near}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={() => void handlePlanRoute()}
              disabled={!canPlan || loading}
              className="w-full rounded-lg bg-slate-900 text-white px-4 py-2 text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-black transition"
            >
              {loading ? 'Planning route…' : 'Plan Route'}
            </button>
          </div>

          {result && (
            <div className="rounded-xl border border-slate-200 bg-white p-4">
              {result.route ? (
                <>
                  <div className="flex items-center gap-1.5 text-emerald-700 font-semibold text-sm mb-2">
                    <span>✅</span> Route found
                  </div>
                  <div className="text-xs text-slate-600 space-y-1">
                    <div>
                      <span className="font-medium text-slate-500">Distance:</span> {result.distance_km} km
                    </div>
                    <div>
                      <span className="font-medium text-slate-500">Waypoints:</span> {result.waypoint_count}
                    </div>
                    {Boolean(result.start_offset_km) && (
                      <div className="text-slate-500">
                        Your start is on land — the sea route begins at the nearest open water, {result.start_offset_km} km away.
                      </div>
                    )}
                    {Boolean(result.end_offset_km) && (
                      <div className="text-slate-500">
                        The route ends at the nearest reachable open water, {result.end_offset_km} km from the zone.
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-1.5 text-rose-700 font-semibold text-sm mb-2">
                    <span>⛔</span> No route found
                  </div>
                  <p className="text-xs text-slate-600">{result.reason}</p>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
