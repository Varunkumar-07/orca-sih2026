import { useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Polygon, Polyline, CircleMarker, ZoomControl, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import type { MapPayload } from '../types'

// Fix default icon paths for Vite
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'

// @ts-ignore
delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
})

// Leaflet's own zoom control already doubles its hit area on a detected
// touch device (26px -> 30px) — PFZ candidate CircleMarkers don't get that
// for free, so we do the same bump ourselves: on touch, a tight cluster of
// candidate zones needs a much bigger finger target than a mouse pointer does.
const PFZ_RADIUS = L.Browser.touch ? 13 : 8
const PFZ_RADIUS_SELECTED = L.Browser.touch ? 17 : 12

// GeoJSON rings are [lon, lat] pairs; Leaflet wants [lat, lon].
function ringToLatLngs(ring: number[][]): [number, number][] {
  return ring.map(([lon, lat]) => [lat, lon] as [number, number])
}

// Every point an overlay contributes to the map's viewport — a pfz_zone
// overlay is a single point, a restricted_area is a Polygon/MultiPolygon
// with real boundary vertices. Used so the map actually fits itself to
// overlay-only responses (e.g. a bare/ambiguous query that resolves no
// pins but still draws every active restricted area) instead of silently
// staying wherever it happened to be.
function overlayLatLngs(overlays: MapPayload['overlays']): [number, number][] {
  const points: [number, number][] = []
  for (const o of overlays) {
    if (o.type === 'pfz_zone') {
      const coords = o.geojson.coordinates as [number, number] // [lon, lat]
      points.push([coords[1], coords[0]])
      continue
    }
    const geo = o.geojson as { type: string; coordinates: unknown }
    if (geo.type === 'Polygon') {
      points.push(...ringToLatLngs((geo.coordinates as number[][][])[0]))
    } else if (geo.type === 'MultiPolygon') {
      for (const polygon of geo.coordinates as number[][][][]) {
        points.push(...ringToLatLngs(polygon[0]))
      }
    }
  }
  return points
}

// The Navigation Agent's A* route is a real sequence of hazard-avoiding
// waypoints, but on an open stretch with nothing to route around they can
// end up nearly collinear — drawn as straight segments, that reads as one
// rigid straight line. This fits a Catmull-Rom spline through the exact
// same waypoints (it passes through every one, never displacing the real
// start/end/turn points) to render a smooth, gently curved line instead —
// styling only, the underlying route/distance/waypoint data is untouched.
function catmullRomSpline(points: [number, number][], segmentsPerPoint = 12): [number, number][] {
  if (points.length < 3) return points

  const at = (i: number) => points[Math.max(0, Math.min(points.length - 1, i))]
  const curved: [number, number][] = [points[0]]

  for (let i = 0; i < points.length - 1; i++) {
    const [lat0, lon0] = at(i - 1)
    const [lat1, lon1] = at(i)
    const [lat2, lon2] = at(i + 1)
    const [lat3, lon3] = at(i + 2)

    for (let step = 1; step <= segmentsPerPoint; step++) {
      const t = step / segmentsPerPoint
      const t2 = t * t
      const t3 = t2 * t
      const lat = 0.5 * (2 * lat1 + (-lat0 + lat2) * t + (2 * lat0 - 5 * lat1 + 4 * lat2 - lat3) * t2 + (-lat0 + 3 * lat1 - 3 * lat2 + lat3) * t3)
      const lon = 0.5 * (2 * lon1 + (-lon0 + lon2) * t + (2 * lon0 - 5 * lon1 + 4 * lon2 - lon3) * t2 + (-lon0 + 3 * lon1 - 3 * lon2 + lon3) * t3)
      curved.push([lat, lon])
    }
  }
  return curved
}

function FitBounds({ pins, overlays }: { pins: MapPayload['pins']; overlays: MapPayload['overlays'] }) {
  const map = useMap()
  useEffect(() => {
    const points: [number, number][] = [
      ...pins.map((p) => [p.lat, p.lon] as [number, number]),
      ...overlayLatLngs(overlays),
    ]
    if (points.length === 0) return

    function applyFit() {
      if (points.length === 1) {
        map.setView(points[0], 9)
        return
      }
      const bounds = L.latLngBounds(points)
      map.fitBounds(bounds, { padding: [30, 30] })
    }

    applyFit()

    // Leaflet measures its container's pixel size once per fitBounds() call
    // and never re-checks on its own. ChatMapPage's layout-reveal animates
    // pane widths in via CSS, so this can mount while the container is
    // still mid-shrink (chat panel not yet at its final width) — the very
    // first applyFit() above may compute against that transitional (too
    // small) size, landing on a wildly over-zoomed view that never
    // recovers, since invalidateSize() alone keeps the *current* zoom
    // valid for the new size rather than re-fitting it. Re-running the
    // same fit (not just invalidateSize) on every resize is idempotent —
    // it keeps converging on the correct zoom for whatever size the
    // container actually has, so it self-corrects once the transition
    // settles, and keeps working for any later resize (window resize,
    // sidebar toggle, etc.) too.
    const container = map.getContainer()
    const observer = new ResizeObserver(() => {
      map.invalidateSize()
      applyFit()
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [pins, overlays, map])
  return null
}

export function MapView({
  payload,
  selectedZone,
  onSelectZone,
}: {
  payload: MapPayload | null
  selectedZone?: string | null
  onSelectZone?: (zoneId: string | null) => void
}) {
  if (!payload || (payload.pins.length === 0 && payload.overlays.length === 0)) {
    return (
      <div className="h-full flex flex-col items-center justify-center bg-slate-100 text-slate-400 p-6 text-center">
        <div className="text-2xl mb-2">🗺️</div>
        <p className="text-sm font-medium">Map will appear here</p>
        <p className="text-xs mt-1">PFZ pins and restricted-area overlays from MapPayload</p>
      </div>
    )
  }

  const center: [number, number] =
    payload.pins.length > 0 ? [payload.pins[0].lat, payload.pins[0].lon] : [13.08, 80.27]

  // Separate overlays
  const restricted = payload.overlays.filter((o) => o.type === 'restricted_area')
  const pfzPoints = payload.overlays.filter((o) => o.type === 'pfz_zone')

  return (
    <MapContainer
      center={center}
      zoom={7}
      zoomControl={false}
      style={{ height: '100%', width: '100%' }}
      className="z-0"
    >
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {/* Bottom-right — reachable by thumb in one-handed mobile use, unlike
          Leaflet's top-left default which sits under the panel header. */}
      <ZoomControl position="bottomright" />
      <FitBounds pins={payload.pins} overlays={payload.overlays} />

      {/* Route polyline — Phase 5 Navigation Agent (A*). Dashed + spline-smoothed
          for style only; the real waypoints (and CircleMarker start/end below)
          are unchanged. */}
      {payload.route && payload.route.length > 0 && (
        <Polyline
          positions={catmullRomSpline(payload.route.map((p) => [p.lat, p.lon] as [number, number]))}
          pathOptions={{ color: '#2563eb', weight: 4, opacity: 0.85, dashArray: '10 8' }}
        />
      )}
      {payload.route && payload.route.length > 0 && payload.route.map((p, idx) =>
        idx === 0 || idx === payload.route!.length - 1 ? (
          <CircleMarker
            key={`route-${idx}`}
            center={[p.lat, p.lon]}
            radius={5}
            pathOptions={{
              color: idx === 0 ? '#0ea5e9' : '#dc2626',
              fillColor: idx === 0 ? '#38bdf8' : '#f87171',
              fillOpacity: 0.9,
              weight: 2,
            }}
          >
            <Popup>
              <div className="text-xs font-semibold">{idx === 0 ? 'Route start' : 'Destination'}</div>
              <div className="text-[11px] font-mono">
                {p.lat.toFixed(4)}, {p.lon.toFixed(4)}
              </div>
            </Popup>
          </CircleMarker>
        ) : null,
      )}

      {/* Pins — PFZ candidates are distinct selectable pins (Phase 4 multi-zone) */}
      {payload.pins.map((pin, i) => {
        const isPFZ = pin.type === 'pfz'
        const isSelected = isPFZ && selectedZone === pin.label
        return (
          <Marker
            key={`pin-${i}-${pin.label}`}
            position={[pin.lat, pin.lon]}
            eventHandlers={
              isPFZ && onSelectZone
                ? {
                    click: () => {
                      onSelectZone(isSelected ? null : pin.label)
                    },
                  }
                : undefined
            }
            opacity={isSelected ? 1 : isPFZ ? 0.9 : 1}
          >
            <Popup>
              <div className="text-xs space-y-0.5">
                <div className="font-semibold">
                  {pin.label}
                  {isSelected && <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 border border-emerald-300">Selected</span>}
                </div>
                <div className="font-mono text-[11px]">
                  {pin.lat.toFixed(4)}, {pin.lon.toFixed(4)}
                </div>
                <div className="text-[11px] capitalize text-slate-500">{pin.type}{isPFZ && ' · click to select/inspect'}</div>
                {pin.zone_id && <div className="text-[11px] text-slate-600">ID: {pin.zone_id}</div>}
                {pin.center && (
                  <div className="text-[11px] font-mono text-slate-600">
                    center: {pin.center.lat.toFixed(4)}, {pin.center.lon.toFixed(4)}
                  </div>
                )}
                {pin.distance_km !== undefined && <div className="text-[11px] text-slate-600">{pin.distance_km} km away</div>}
                {pin.sst_celsius !== undefined && <div className="text-[11px] text-slate-600">SST {pin.sst_celsius}°C</div>}
                {pin.chlorophyll_mg_m3 !== undefined && <div className="text-[11px] text-slate-600">Chlorophyll {pin.chlorophyll_mg_m3} mg/m³</div>}
                {pin.advisory && <div className="mt-1 text-[10px] text-slate-500 italic">{pin.advisory}</div>}
              </div>
            </Popup>
          </Marker>
        )
      })}

      {/* PFZ points as CircleMarkers (green) — each candidate distinct & selectable */}
      {pfzPoints.map((o, i) => {
        const coords = o.geojson.coordinates as [number, number] // [lon, lat]
        const lat = coords[1] as number
        const lon = coords[0] as number
        const isSelected = selectedZone === o.name
        return (
          <CircleMarker
            key={`pfz-${i}-${o.name}`}
            center={[lat, lon]}
            radius={isSelected ? PFZ_RADIUS_SELECTED : PFZ_RADIUS}
            pathOptions={{
              color: isSelected ? '#065f46' : '#059669',
              fillColor: isSelected ? '#059669' : '#10b981',
              fillOpacity: isSelected ? 0.85 : 0.6,
              weight: isSelected ? 3 : 2,
            }}
            eventHandlers={
              onSelectZone
                ? {
                    click: () => {
                      onSelectZone(isSelected ? null : o.name)
                    },
                  }
                : undefined
            }
          >
            <Popup>
              <div className="text-xs space-y-0.5">
                <div className="font-semibold">
                  {o.name} (PFZ)
                  {isSelected && <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 border border-emerald-300">Selected</span>}
                </div>
                <div className="text-[11px] font-mono">
                  {lat.toFixed(4)}, {lon.toFixed(4)}
                </div>
                {o.zone_id && <div className="text-[11px] text-slate-600">ID: {o.zone_id}</div>}
                {o.center && (
                  <div className="text-[11px] font-mono text-slate-600">
                    center: {o.center.lat.toFixed(4)}, {o.center.lon.toFixed(4)}
                  </div>
                )}
                {o.distance_km !== undefined && <div className="text-[11px] text-slate-600">{o.distance_km} km away</div>}
                {o.sst_celsius !== undefined && <div className="text-[11px] text-slate-600">SST {o.sst_celsius}°C</div>}
                {o.chlorophyll_mg_m3 !== undefined && <div className="text-[11px] text-slate-600">Chlorophyll {o.chlorophyll_mg_m3} mg/m³</div>}
                {o.advisory && <div className="mt-1 text-[10px] text-slate-500 italic">{o.advisory}</div>}
                <div className="text-[10px] text-slate-500 mt-1">{isSelected ? 'Selected — click to deselect' : 'Click to select/inspect'}</div>
              </div>
            </Popup>
          </CircleMarker>
        )
      })}

      {/* Restricted area polygons — outer ring only, no holes (matches WDPA source data) */}
      {restricted.flatMap((o, i) => {
        const geo = o.geojson as { type: string; coordinates: unknown }
        const highlighted = Boolean(o.highlighted)
        const pathOptions = {
          color: highlighted ? '#dc2626' : '#f59e0b',
          fillColor: highlighted ? '#fecaca' : '#fde68a',
          fillOpacity: highlighted ? 0.45 : 0.25,
          weight: highlighted ? 3 : 2,
          dashArray: highlighted ? undefined : '6 6',
        }
        const renderPopup = () => (
          <Popup>
            <div className="text-xs">
              <div className="font-semibold">{o.name}</div>
              <div className="text-[11px]">{highlighted ? '⚠️ Inside — fishing prohibited' : 'Restricted area'}</div>
            </div>
          </Popup>
        )

        if (geo.type === 'Polygon') {
          // GeoJSON Polygon coordinates are [[[lon,lat]...]] — outer ring only
          const coords = geo.coordinates as number[][][]
          return [
            <Polygon key={`restricted-${i}`} positions={ringToLatLngs(coords[0])} pathOptions={pathOptions}>
              {renderPopup()}
            </Polygon>,
          ]
        }

        if (geo.type === 'MultiPolygon') {
          // GeoJSON MultiPolygon coordinates are [[[[lon,lat]...]]...] — one
          // ring-array per polygon. Render one <Polygon> per polygon (outer
          // ring only, same as above) so disjoint parts of the same MPA
          // (e.g. real WDPA data before buffering merges them) all render
          // instead of silently corrupting into garbled positions.
          const coords = geo.coordinates as number[][][][]
          return coords.map((polygon, j) => (
            <Polygon
              key={`restricted-${i}-${j}`}
              positions={ringToLatLngs(polygon[0])}
              pathOptions={pathOptions}
            >
              {renderPopup()}
            </Polygon>
          ))
        }

        console.warn(`MapView: unexpected restricted-area geometry type "${geo.type}" for "${o.name}" — skipping`)
        return []
      })}
    </MapContainer>
  )
}
