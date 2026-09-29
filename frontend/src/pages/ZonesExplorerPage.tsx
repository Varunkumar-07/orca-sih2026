import { useEffect, useMemo, useState } from 'react'
import { MapView } from '../components/MapView'
import type { MapPayload, ZoneRecord, ZonesResponse } from '../types'
import { ensureOk, friendlyError, devBackendHint } from '../lib/httpError'

function zonesToMapPayload(zones: ZoneRecord[]): MapPayload {
  const overlays: MapPayload['overlays'] = []

  for (const z of zones) {
    if (z.type === 'pfz' && z.coordinates) {
      overlays.push({
        type: 'pfz_zone',
        name: z.name,
        geojson: { type: 'Point', coordinates: [z.coordinates.lon, z.coordinates.lat] },
        zone_id: z.id,
        center: z.coordinates,
        distance_km: z.distance_km,
        sst_celsius: z.sst_celsius,
        chlorophyll_mg_m3: z.chlorophyll_mg_m3,
        advisory: z.advisory,
      })
    } else if (z.type === 'restricted' && z.geometry) {
      overlays.push({
        type: 'restricted_area',
        name: z.name,
        geojson: z.geometry as { type: string; coordinates: unknown },
        highlighted: false,
      })
    }
  }

  return { pins: [], overlays }
}

function ZoneListRow({ zone, active, onClick }: { zone: ZoneRecord; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 rounded-lg border text-xs transition ${
        active
          ? 'bg-teal-50 border-teal-300 ring-1 ring-teal-300'
          : 'bg-white border-slate-200 hover:border-teal-300 hover:bg-teal-50/40'
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold text-slate-800">{zone.name}</span>
        <span
          className={`shrink-0 text-[9px] px-1.5 py-0.5 rounded-full uppercase tracking-wide ${
            zone.type === 'pfz' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
          }`}
        >
          {zone.type === 'pfz' ? 'PFZ' : 'Restricted'}
        </span>
      </div>
      {zone.type === 'pfz' && zone.near && <div className="text-slate-500 mt-0.5">near {zone.near}</div>}
    </button>
  )
}

function ZoneDetailPanel({ zone, onClose }: { zone: ZoneRecord; onClose: () => void }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <div>
          <span
            className={`text-[9px] px-1.5 py-0.5 rounded-full uppercase tracking-wide ${
              zone.type === 'pfz' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
            }`}
          >
            {zone.type === 'pfz' ? 'Potential Fishing Zone' : 'Restricted Area'}
          </span>
          <h3 className="text-sm font-bold text-slate-800 mt-1.5">{zone.name}</h3>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-sm leading-none">
          ✕
        </button>
      </div>

      <div className="mt-3 space-y-1.5 text-xs text-slate-600">
        {zone.near && (
          <div>
            <span className="font-medium text-slate-500">Near:</span> {zone.near}
          </div>
        )}
        {zone.coordinates && (
          <div className="font-mono">
            {zone.coordinates.lat.toFixed(4)}, {zone.coordinates.lon.toFixed(4)}
          </div>
        )}
        {zone.distance_km !== undefined && (
          <div>
            <span className="font-medium text-slate-500">Distance:</span> {zone.distance_km} km
          </div>
        )}
        {zone.sst_celsius !== undefined && (
          <div>
            <span className="font-medium text-slate-500">SST:</span> {zone.sst_celsius}°C
          </div>
        )}
        {zone.chlorophyll_mg_m3 !== undefined && (
          <div>
            <span className="font-medium text-slate-500">Chlorophyll:</span> {zone.chlorophyll_mg_m3} mg/m³
          </div>
        )}
        {zone.advisory && <div className="italic text-slate-500 pt-1">{zone.advisory}</div>}
        {zone.type === 'restricted' && (
          <div className="text-slate-500 pt-1">Fishing is prohibited inside this boundary.</div>
        )}
        <div className="text-[10px] text-slate-400 pt-1 font-mono">ID: {zone.id}</div>
      </div>
    </div>
  )
}

export function ZonesExplorerPage() {
  const [zones, setZones] = useState<ZoneRecord[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedZone, setSelectedZone] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetch('/api/zones')
      .then(ensureOk)
      .then((res) => res.json() as Promise<ZonesResponse>)
      .then((data) => {
        if (!cancelled) setZones(data.zones)
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(friendlyError(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  const mapPayload = useMemo(() => (zones ? zonesToMapPayload(zones) : null), [zones])
  const pfzZones = useMemo(() => zones?.filter((z) => z.type === 'pfz') ?? [], [zones])
  const restrictedZones = useMemo(() => zones?.filter((z) => z.type === 'restricted') ?? [], [zones])
  const activeZone = zones?.find((z) => z.id === selectedZone) ?? null

  return (
    <div className="h-full flex flex-col">
      {/* Intro block — ported from Mock's .page-hero (badge + H1 + subtitle) */}
      <div className="shrink-0 px-4 sm:px-6 pt-6 pb-2 text-center">
        <span className="inline-flex items-center gap-1.5 text-[11px] tracking-[.06em] uppercase px-2.5 py-1.5 rounded-full bg-teal-500/[.14] border border-teal-500/[.28] text-teal-700">
          Zones · Smart India Hackathon 2026 · ISRO
        </span>
        <h1 className="mt-2.5 text-[clamp(26px,3.6vw,38px)] leading-none tracking-[-.03em] text-[#0d0c0b] font-normal">
          Zones Explorer
        </h1>
        <p className="mt-2.5 text-sm leading-relaxed text-[#0d0c0b]/64 max-w-[52ch] mx-auto">
          Browse potential fishing zones and restricted marine areas on a full-screen map.
        </p>
      </div>

      <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-y-auto lg:overflow-hidden gap-3 p-3 pb-6">
        {/* Map — glass card matching the Assistant page's panels */}
        <div className="flex-1 min-w-0 min-h-[320px] lg:min-h-0 rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg flex flex-col overflow-hidden">
          <div className="px-4 py-2 border-b border-white/40 bg-white/15 flex items-center justify-between shrink-0">
            <h2 className="text-xs font-semibold tracking-widest uppercase text-slate-600">
              Zones Explorer · PFZ & Restricted Areas
            </h2>
            <span className="text-[11px] text-slate-500">
              {zones ? `${pfzZones.length} PFZ · ${restrictedZones.length} restricted` : error ? 'Failed to load' : 'Loading…'}
            </span>
          </div>
          <div className="flex-1 min-h-[300px]">
            <MapView payload={mapPayload} selectedZone={selectedZone} onSelectZone={setSelectedZone} />
          </div>
        </div>

        <div className="w-full lg:w-[340px] shrink-0 rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg min-h-0 overflow-y-auto p-3 space-y-3">
          {error && (
            <div className="rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs px-3 py-2">
              Couldn't load zones: {error}{devBackendHint()}
            </div>
          )}

          {activeZone && <ZoneDetailPanel zone={activeZone} onClose={() => setSelectedZone(null)} />}

          {!zones && !error && <div className="text-xs text-slate-400 px-1">Fetching zones…</div>}

          {zones && (
            <>
              <div>
                <div className="text-[10px] font-semibold tracking-widest uppercase text-emerald-600 mb-1.5 px-1">
                  Potential Fishing Zones ({pfzZones.length})
                </div>
                <div className="space-y-1.5">
                  {pfzZones.map((z) => (
                    <ZoneListRow
                      key={z.id}
                      zone={z}
                      active={selectedZone === z.id}
                      onClick={() => setSelectedZone((prev) => (prev === z.id ? null : z.id))}
                    />
                  ))}
                </div>
              </div>

              <div>
                <div className="text-[10px] font-semibold tracking-widest uppercase text-amber-600 mb-1.5 px-1">
                  Restricted Areas ({restrictedZones.length})
                </div>
                <div className="space-y-1.5">
                  {restrictedZones.map((z) => (
                    <ZoneListRow
                      key={z.id}
                      zone={z}
                      active={selectedZone === z.id}
                      onClick={() => setSelectedZone((prev) => (prev === z.id ? null : z.id))}
                    />
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
