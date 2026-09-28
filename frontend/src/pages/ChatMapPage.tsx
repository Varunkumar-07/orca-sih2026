import { useState, useRef, useEffect } from 'react'
import { ChatPanel } from '../components/ChatPanel'
import { MapView } from '../components/MapView'
import { TracePanel } from '../components/TracePanel'
import type { FinalResponse, MapPayload, TraceStep } from '../types'

type Message = { role: 'user' | 'orca'; text: string }



export default function ChatMapPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [trace, setTrace] = useState<TraceStep[]>([])
  const [visibleCount, setVisibleCount] = useState(0)
  const [mapPayload, setMapPayload] = useState<MapPayload | null>(null)
  const [selectedZone, setSelectedZone] = useState<string | null>(null)
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null)
  const [responseLanguage, setResponseLanguage] = useState<string | null>(null)
  const [demoMode, setDemoMode] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Persisted to sessionStorage (not component state alone) so navigating
  // to another page and back to Assistant doesn't silently start a brand
  // new backend session — sessionStorage survives a remount, unlike state.
  const [sessionId, setSessionId] = useState<string | null>(() => {
    try {
      return sessionStorage.getItem('orca_session_id')
    } catch {
      return null
    }
  })
  const intervalRef = useRef<number | null>(null)

  // main.py always sets map_payload on a successful response (empty
  // pins/overlays for a greeting, populated otherwise) — so "has a
  // MapPayload arrived yet" is exactly "has the first query succeeded",
  // and stays true for the rest of the session even if a later query errors.
  const hasResponded = mapPayload !== null
  const [panelsVisible, setPanelsVisible] = useState(false)

  // Two-step reveal: mount the map/trace panels at opacity-0 first, then
  // flip to opacity-100 a frame later so the transition actually animates
  // instead of just appearing already-visible on the first paint.
  useEffect(() => {
    if (!hasResponded) return
    const id = requestAnimationFrame(() => setPanelsVisible(true))
    return () => cancelAnimationFrame(id)
  }, [hasResponded])

  // Animate trace reveal. visibleCount is already reset to 0 by sendQuery
  // (synchronously, before trace is ever set) — this effect only starts the
  // reveal timer, it doesn't also need to reset the counter itself.
  useEffect(() => {
    if (trace.length === 0) return
    let idx = 0
    intervalRef.current = window.setInterval(() => {
      idx += 1
      setVisibleCount(idx)
      if (idx >= trace.length && intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }, 350)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [trace])

  // Reset selected zone whenever new map data arrives (multi-candidate
  // selection). Adjusted directly during render (React's documented pattern
  // for "reset state when a prop/value changes") rather than in a useEffect
  // — avoids an extra post-commit render pass just to clear one field.
  const [prevMapPayload, setPrevMapPayload] = useState(mapPayload)
  if (mapPayload !== prevMapPayload) {
    setPrevMapPayload(mapPayload)
    setSelectedZone(null)
  }

  // Phase 5.1: frontend geolocation capture with graceful fallback
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

  async function sendQuery(q: string) {
    const query = q.trim()
    // Re-entrancy guard: without this, a second query sent while the first
    // is still in flight can have its response arrive first and get
    // silently overwritten by the (now-stale) first response, or vice
    // versa — this is the actual entry point both the Send button and the
    // Enter-key handler funnel through, so guarding here covers both.
    if (!query || loading) return
    setMessages((m) => [...m, { role: 'user', text: query }])
    setInput('')
    setLoading(true)
    setError(null)
    setTrace([])
    setVisibleCount(0)

    try {
      // Capture browser geolocation — if denied/unavailable/times out, fall back
      // gracefully to text-based location resolution (no error, no retry prompt).
      const userCoords = await getUserLocation()
      const endpoint = demoMode ? '/api/query/demo' : '/api/query/full'
      const payload: Record<string, unknown> = { query, session_id: sessionId }
      if (userCoords) {
        payload.user_location = userCoords
      }

      // Phase 4.3/5.3: if the user selected a PFZ candidate from the
      // previous response's map, send its coordinates as
      // selected_destination — this is what triggers Navigation Agent
      // routing on the backend. Coordinates (not the zone_id) are sent
      // because a fresh query can return a different candidate list than
      // the one actually shown/clicked. selectedZone resets automatically
      // once a new mapPayload arrives (see the effect above), so this only
      // ever carries forward to the single next query.
      if (selectedZone) {
        const pin = mapPayload?.pins.find((p) => p.label === selectedZone)
        const overlay = mapPayload?.overlays.find((o) => o.name === selectedZone)
        const destination = pin ? { lat: pin.lat, lon: pin.lon } : overlay?.center
        if (destination) {
          payload.selected_destination = destination
        }
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`${res.status} ${txt}`)
      }
      // Extract Session Header — persist across turns for multi-turn memory
      const newSessionId = res.headers.get('X-Session-Id')
      if (newSessionId) {
        setSessionId(newSessionId)
        try {
          sessionStorage.setItem('orca_session_id', newSessionId)
        } catch {
          // Private browsing / blocked storage — session still works for
          // the rest of this page's lifetime via component state alone.
        }
      }
      const data: FinalResponse = await res.json()
      setMessages((m) => [...m, { role: 'orca', text: data.answer_text }])
      setTrace(data.reasoning_trace ?? [])
      setMapPayload(data.map_payload ?? null)
      setDetectedLanguage(data.detected_language ?? null)
      setResponseLanguage(data.response_language ?? null)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      setMessages((m) => [
        ...m,
        { role: 'orca', text: `Sorry — request failed: ${msg}\nTry demo mode or check backend is running on :8000.` },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleSend() {
    void sendQuery(input)
  }

  const chatPanel = (
    <ChatPanel
      messages={messages}
      input={input}
      setInput={setInput}
      onSend={handleSend}
      loading={loading}
      onExampleClick={(q) => void sendQuery(q)}
      detectedLanguage={detectedLanguage}
      responseLanguage={responseLanguage}
    />
  )

  return (
    <div className="h-full flex flex-col">
      {/* Intro block — ported from Mock's .page-hero (badge + H1 + subtitle) */}
      <div className="shrink-0 px-4 sm:px-6 pt-6 pb-2 text-center">
        <span className="inline-flex items-center gap-1.5 text-[11px] tracking-[.06em] uppercase px-2.5 py-1.5 rounded-full bg-teal-500/[.14] border border-teal-500/[.28] text-teal-700">
          Assistant · Smart India Hackathon 2026 · ISRO
        </span>
        <h1 className="mt-2.5 text-[clamp(26px,3.6vw,38px)] leading-none tracking-[-.03em] text-[#0d0c0b] font-normal">
          ORCA Assistant
        </h1>
        <p className="mt-2.5 text-sm leading-relaxed text-[#0d0c0b]/64 max-w-[52ch] mx-auto">
          Ask a question — the map and multi-agent trace appear alongside the answer.
        </p>
      </div>

      {/* Slim control strip — Demo/Live toggle only; branding lives in the
          global nav above and the language badge lives in ChatPanel's own
          head, so this page no longer duplicates either. Capped to the same
          max-width as the chat card below so it doesn't stretch edge to
          edge as a full-bleed bar. */}
      <div className="shrink-0 flex justify-center px-4 sm:px-6 pb-2">
        <div className="w-full max-w-[720px] flex items-center justify-end gap-3">
          <label className="flex items-center gap-2 text-xs cursor-pointer select-none">
            <span className={demoMode ? 'text-teal-700 font-semibold' : 'text-[#0d0c0b]/50'}>Demo</span>
            <button
              onClick={() => setDemoMode((v) => !v)}
              className={`w-10 h-5 rounded-full p-0.5 flex items-center transition ${demoMode ? 'bg-teal-500 justify-start' : 'bg-slate-400 justify-end'}`}
              aria-label="toggle demo/live"
            >
              <span className="w-4 h-4 rounded-full bg-white shadow" />
            </button>
            <span className={!demoMode ? 'text-amber-700 font-semibold' : 'text-[#0d0c0b]/50'}>Live</span>
          </label>
          <span className="hidden sm:inline text-[11px] text-[#0d0c0b]/50">
            {demoMode ? 'Fixture-based · no API key needed' : 'Full LLM pipeline · /query/full'}
          </span>
        </div>
      </div>

      {error && (
        <div className="shrink-0 bg-rose-50/90 backdrop-blur border-b border-rose-200 text-rose-700 text-xs px-4 py-2">{error}</div>
      )}

      {/* Main layout — ONE persistent flex row. The chat column never
          unmounts or changes DOM parent between the two layout modes (only
          its own width/border classes change), so its width transition
          animates smoothly instead of snapping — a conditionally-swapped
          wrapper element would remount ChatPanel and defeat any CSS
          transition entirely. Map + trace mount only once mapPayload
          exists (hasResponded) and reveal with a staggered fade/slide/scale
          via panelsVisible; once mounted they stay mounted for the rest of
          the session even if a later query errors. */}
      <div className={`flex-1 flex flex-col lg:flex-row min-h-0 overflow-y-auto lg:overflow-hidden gap-3 p-3 pb-6 ${hasResponded ? '' : 'lg:items-start'}`}>
        {/* Chat — glass card capped at Mock's chat-stage width (720px) and
            height (content-sized, not stretched) while it's the only thing
            on screen; once map/trace mount it docks to the existing narrow
            left column and stretches full height like before. */}
        <div
          className={`w-full min-h-[420px] lg:min-h-0 rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg flex flex-col overflow-hidden transition-[width] duration-700 ease-in-out ${
            hasResponded ? 'lg:w-[380px] xl:w-[420px]' : 'max-w-[720px] mx-auto lg:max-h-[600px]'
          }`}
        >
          {chatPanel}
        </div>

        {hasResponded && (
        <>
        {/* Map */}
        <div
          className={`flex-1 min-w-0 min-h-[320px] lg:min-h-0 rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg flex flex-col overflow-hidden transition-all duration-500 ease-out ${
            panelsVisible ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-3 scale-[0.98]'
          }`}
        >
          <div className="px-4 py-2 border-b border-white/40 bg-white/15 flex items-center justify-between">
            <h3 className="text-xs font-semibold tracking-widest uppercase text-slate-600">Map · PFZ & Restricted Areas</h3>
            <span className="text-[11px] text-slate-500">
              {mapPayload ? `${mapPayload.pins.length} pins · ${mapPayload.overlays.length} overlays` : 'No data'}
            </span>
          </div>
          <div className="flex-1 min-h-[300px]">
            <MapView payload={mapPayload} selectedZone={selectedZone} onSelectZone={setSelectedZone} />
          </div>
          {selectedZone &&
            (() => {
              const pin = mapPayload?.pins.find((p) => p.label === selectedZone) as unknown as Record<string, unknown> | undefined
              const overlay = mapPayload?.overlays.find((o) => o.name === selectedZone) as unknown as Record<string, unknown> | undefined
              const src = (pin ?? overlay) as Record<string, unknown> | undefined
              const center = ((pin as unknown as { center?: { lat: number; lon: number } })?.center ??
                (overlay as unknown as { center?: { lat: number; lon: number } })?.center) as { lat: number; lon: number } | undefined
              const zoneId = src?.zone_id as string | undefined
              const distanceKm = src?.distance_km as number | undefined
              const sst = src?.sst_celsius as number | undefined
              const chl = src?.chlorophyll_mg_m3 as number | undefined
              const advisory = src?.advisory as string | undefined
              return (
                <div className="px-3 py-2 bg-emerald-50/90 border-t border-emerald-200 text-xs text-emerald-800">
                  <div className="flex items-center justify-between gap-2">
                    <span>
                      Selected PFZ: <strong>{selectedZone}</strong>
                      {zoneId && zoneId !== selectedZone && <> — ID: {zoneId}</>}
                      {center && <> — center: {center.lat.toFixed(4)}, {center.lon.toFixed(4)}</>}
                      {distanceKm !== undefined && <> — {distanceKm} km</>}
                    </span>
                    <div className="ml-2 shrink-0 flex items-center gap-2">
                      <button
                        onClick={() => void sendQuery('Get me a route to the selected fishing zone.')}
                        disabled={loading}
                        className="text-[11px] px-2 py-1 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        Navigate here
                      </button>
                      <button
                        onClick={() => setSelectedZone(null)}
                        className="text-[11px] px-2 py-1 rounded bg-white border border-emerald-300 hover:bg-emerald-100"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                  {(sst !== undefined || chl !== undefined || advisory) && (
                    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-emerald-900/80">
                      {sst !== undefined && <span>SST {sst}°C</span>}
                      {chl !== undefined && <span>Chlorophyll {chl} mg/m³</span>}
                      {advisory && <span className="italic text-emerald-700">{advisory}</span>}
                    </div>
                  )}
                  <div className="mt-0.5 text-[10px] text-emerald-700/70">Click another pin to inspect, click "Navigate here" for a route, or click again to deselect.</div>
                </div>
              )
            })()}
          <div className="px-3 py-2 bg-white/15 border-t border-white/40 text-[11px] text-slate-500 flex gap-3 flex-wrap">
            <span className="inline-flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-sky-500" /> Query
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500" /> PFZ
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="w-3 h-2 bg-red-200 border border-red-600" /> Gulf of Mannar (restricted)
            </span>
          </div>
        </div>

        {/* Trace */}
        <div
          className={`w-full lg:w-[380px] xl:w-[400px] rounded-[20px] bg-white/15 backdrop-blur-[18px] border border-white/40 shadow-lg min-h-[280px] lg:min-h-0 flex flex-col overflow-hidden transition-all duration-500 ease-out delay-200 ${
            panelsVisible ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-3 scale-[0.98]'
          }`}
        >
          <TracePanel trace={trace} visibleCount={visibleCount} />
        </div>
        </>
        )}
      </div>

      <footer className="shrink-0 h-6 bg-slate-900/85 backdrop-blur text-slate-400 text-[11px] flex items-center justify-center px-4">
        ORCA · Conversational multi-agent marine intelligence · Judged trace panel §6.5 · Coordinates (lat,lon) → GeoJSON (lon,lat) at visualization boundary
      </footer>
    </div>
  )
}
