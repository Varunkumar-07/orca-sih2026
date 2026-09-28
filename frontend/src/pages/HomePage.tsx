import { Link } from 'react-router-dom'
import { useEffect, useRef } from 'react'

const ORCA_FONT = "'Inter Tight', 'Helvetica Neue', Helvetica, Arial, sans-serif"

// Three full-screen panels that fade in/out on scroll, ported from Mock's
// index.html panel structure — content rewritten from SIH's own copy
// (HomePage's hero text and its FEATURES descriptions) instead of Mock's.
const PANELS: { eyebrow: string; heading: string; sub: string; ctaLabel: string; ctaTo: string }[] = [
  {
    eyebrow: 'Smart India Hackathon 2026 · ISRO',
    heading: 'ORCA',
    sub: 'Conversational multi-agent marine intelligence for fishermen & coastal operators — fishing zones, weather, hazards, and safe routes, in natural language.',
    ctaLabel: 'Open Assistant →',
    ctaTo: '/chat',
  },
  {
    eyebrow: 'Assistant · Zones Explorer · Weather',
    heading: 'Ask. Explore. Check the weather.',
    sub: 'Ask in natural language and get PFZ pins, restricted-area overlays, and a live reasoning trace.',
    ctaLabel: 'Explore zones',
    ctaTo: '/zones',
  },
  {
    eyebrow: 'Route Planner · Alerts & Advisories · Analytics Dashboard',
    heading: 'Plan safe routes.',
    sub: 'Plan a safe route between two points that avoids restricted waters.',
    ctaLabel: 'Plan a route',
    ctaTo: '/route',
  },
]

const DRIFT = 22

// Cue table per panel: [fadeInStart, fadeInEnd, fadeOutStart, fadeOutEnd] as
// a fraction of scroll progress through the panel track — lifted directly
// from Mock's CUES table. The first panel is visible immediately (no
// fade-in) and the last panel's fade-out window sits past 1, so it never
// actually finishes fading before the track ends.
const CUES: [number, number, number, number][] = [
  [0.0, 0.0, 0.15, 0.23],
  [0.35, 0.43, 0.57, 0.65],
  [0.77, 0.85, 1.1, 1.2],
]

function clamp(v: number, a: number, b: number) {
  return v < a ? a : v > b ? b : v
}

function smooth(t: number) {
  return t * t * (3 - 2 * t)
}

function ramp(p: number, a: number, b: number) {
  if (b <= a) return p >= b ? 1 : 0
  return smooth(clamp((p - a) / (b - a), 0, 1))
}

export function HomePage() {
  const scrollRef = useRef<HTMLDivElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const panelRefs = useRef<(HTMLDivElement | null)[]>([])

  useEffect(() => {
    const container = scrollRef.current
    const track = trackRef.current
    if (!container || !track) return

    let trackTop = 0
    let trackScrollable = 1

    const measure = () => {
      const containerRect = container.getBoundingClientRect()
      const trackRect = track.getBoundingClientRect()
      trackTop = trackRect.top - containerRect.top + container.scrollTop
      trackScrollable = Math.max(track.offsetHeight - container.clientHeight, 1)
    }

    const paint = () => {
      const progress = clamp((container.scrollTop - trackTop) / trackScrollable, 0, 1)
      panelRefs.current.forEach((el, i) => {
        if (!el) return
        const [a, b, c, d] = CUES[i]
        const enter = ramp(progress, a, b)
        const leave = ramp(progress, c, d)
        const o = enter * (1 - leave)
        const y = (1 - enter) * DRIFT - leave * DRIFT
        el.style.opacity = String(o)
        el.style.transform = `translate3d(0, ${y}px, 0)`
        el.style.pointerEvents = o > 0.6 ? 'auto' : 'none'
      })
    }

    const onScroll = () => paint()
    const onResize = () => {
      measure()
      paint()
    }

    measure()
    paint()
    container.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onResize)

    return () => {
      container.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onResize)
    }
  }, [])

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto" style={{ fontFamily: ORCA_FONT }}>
      <div ref={trackRef} className="relative" style={{ height: '560vh', minHeight: '3200px' }}>
        <div className="sticky top-0 h-[calc(100vh-4rem)] overflow-hidden">
          {PANELS.map((p, i) => (
            <div
              key={p.heading}
              ref={(el) => {
                panelRefs.current[i] = el
              }}
              className="absolute inset-0 flex flex-col items-center justify-center text-center px-6"
              style={{ opacity: i === 0 ? 1 : 0, willChange: 'opacity, transform' }}
            >
              <div className="flex items-center justify-center flex-wrap gap-x-3 gap-y-1.5 text-[12.5px] tracking-[.045em] text-[#0d0c0b]/64 mb-5 max-w-[46ch]">
                {p.eyebrow}
              </div>
              <h1 className="font-normal text-[clamp(34px,7.1vw,104px)] leading-[.98] tracking-[-.036em] text-[#0d0c0b] max-w-[15ch] text-balance">
                {p.heading}
              </h1>
              <p className="mt-6 text-[clamp(15px,1.28vw,19px)] leading-[1.5] tracking-[-.008em] text-[#0d0c0b]/64 max-w-[46ch]">
                {p.sub}
              </p>
              <Link
                to={p.ctaTo}
                className="mt-9 inline-flex items-center h-12 px-7 rounded-full bg-black text-white text-[15px] font-medium transition hover:-translate-y-0.5"
              >
                {p.ctaLabel}
              </Link>
            </div>
          ))}
        </div>
      </div>

      <footer className="text-center text-[11px] text-[#0d0c0b]/42 pb-8 bg-slate-50">
        ORCA · Conversational multi-agent marine intelligence
      </footer>
    </div>
  )
}
