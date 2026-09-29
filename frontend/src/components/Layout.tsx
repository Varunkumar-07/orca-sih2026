import { Suspense, useEffect, useRef, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'

function PageLoadingFallback() {
  // Shown only while a lazily-loaded page chunk is still fetching (see
  // App.tsx's React.lazy() imports) — the nav bar above stays interactive
  // throughout since Suspense here only covers the Outlet, not Layout.
  return (
    <div className="h-full flex items-center justify-center text-slate-400 text-xs gap-2">
      <span className="w-3 h-3 rounded-full border-2 border-slate-300 border-t-teal-500 animate-spin" />
      Loading…
    </div>
  )
}

const NAV_ITEMS: { to: string; label: string }[] = [
  { to: '/', label: 'Home' },
  { to: '/chat', label: 'Assistant' },
  { to: '/zones', label: 'Zones Explorer' },
  { to: '/weather', label: 'Weather' },
  { to: '/route', label: 'Route Planner' },
  { to: '/alerts', label: 'Alerts' },
  { to: '/analytics', label: 'Analytics' },
  { to: '/download', label: 'Download' },
  { to: '/history', label: 'History' },
]

const VIDEO_URL = 'https://videos.pexels.com/video-files/1409899/1409899-uhd_2560_1440_25fps.mp4'

// Scroll-scrubbed video, ported from Mock/index.html: the clip never plays on
// its own — scroll position maps to a point in its duration, and a small
// easing loop nudges currentTime toward that target each frame. window's
// capture-phase scroll listener picks up scrolling from whichever page's own
// internal scroll container fired it (each route manages its own overflow),
// not just document/window scroll.
//
// Also ported from Mock: the clip is fully fetched (with a % boot screen)
// and played back from a blob URL instead of the network URL, so scrubbing
// never stalls on an unbuffered range once it's ready; a scroll-progress
// meter and a one-time touch/wheel/key "unlock" (to satisfy iOS autoplay
// rules for currentTime seeks) come along with it.
function useScrollScrubbedVideo(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  meterRef: React.RefObject<HTMLDivElement | null>,
  bootBarRef: React.RefObject<HTMLDivElement | null>,
  bootPctRef: React.RefObject<HTMLParagraphElement | null>,
  isHomeRef: React.RefObject<boolean>,
  onReady: () => void,
) {
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let duration = 0
    let progress = 0
    let seekTo = 0
    let seekAt = 0
    let ready = false
    let started = false
    let attached = false
    let wasHome = isHomeRef.current
    let raf = 0

    const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v)

    function readScroll(target: EventTarget | null) {
      if (target instanceof HTMLElement) {
        const max = target.scrollHeight - target.clientHeight
        progress = max > 0 ? clamp(target.scrollTop / max, 0, 1) : 0
      } else {
        const max = document.documentElement.scrollHeight - window.innerHeight
        progress = max > 0 ? clamp(window.scrollY / max, 0, 1) : 0
      }
      if (duration) seekTo = progress * duration
    }

    function onScroll(e: Event) {
      readScroll(e.target instanceof HTMLElement ? e.target : null)
    }

    function setBootProgress(f: number) {
      const pct = Math.round(f * 100)
      if (bootBarRef.current) bootBarRef.current.style.transform = `scaleX(${f})`
      if (bootPctRef.current) bootPctRef.current.textContent = `LOADING ${pct}%`
    }

    function start() {
      if (started) return
      started = true
      ready = true
      onReady()
      readScroll(null)
      seekAt = seekTo
    }

    function onLoadedMetadata() {
      duration = video!.duration || 0
      video!.pause()
      readScroll(null)
      seekAt = seekTo
      try {
        video!.currentTime = seekAt
      } catch {
        // ignore — seeking before the video is fully ready can throw
      }
    }

    function attach(src: string) {
      if (attached) return
      attached = true
      video!.addEventListener('loadedmetadata', onLoadedMetadata)
      video!.addEventListener('loadeddata', start)
      video!.addEventListener('canplaythrough', start)
      video!.addEventListener('error', start)
      video!.src = src
      video!.load()
      setTimeout(start, 12000)
    }

    function preload() {
      const controller = typeof AbortController !== 'undefined' ? new AbortController() : null
      const bail = setTimeout(() => {
        controller?.abort()
        setBootProgress(1)
        attach(VIDEO_URL)
      }, 15000)

      fetch(VIDEO_URL, { signal: controller?.signal })
        .then((res) => {
          if (!res.ok || !res.body) throw new Error('Network response was not ok')
          const totalHeader = res.headers.get('content-length')
          const total = totalHeader ? parseInt(totalHeader, 10) : 0
          const reader = res.body.getReader()
          const chunks: Uint8Array[] = []
          let got = 0
          function pump(): Promise<Blob> {
            return reader.read().then((result) => {
              if (result.done) return new Blob(chunks as BlobPart[], { type: 'video/mp4' })
              const chunk = result.value
              chunks.push(chunk)
              got += chunk.length
              const fraction = total ? got / total : Math.min(got / 11e6, 0.95)
              setBootProgress(fraction)
              return pump()
            })
          }
          return pump()
        })
        .then((blob) => {
          clearTimeout(bail)
          setBootProgress(1)
          attach(URL.createObjectURL(blob))
        })
        .catch(() => {
          clearTimeout(bail)
          setBootProgress(1)
          attach(VIDEO_URL)
        })
    }

    // Priming play-then-pause for iOS autoplay rules, used only by Home's
    // scroll-scrub — away from Home the clip is meant to keep playing on
    // its own, so this must not pause it back out from under that.
    function unlock() {
      const p = video!.play()
      if (p && typeof p.then === 'function') {
        p.then(() => {
          if (isHomeRef.current) video!.pause()
        }).catch(() => {})
      } else if (isHomeRef.current) {
        video!.pause()
      }
    }

    function frame() {
      // Coming back to Home: the clip kept playing on the other page, so
      // seekAt no longer matches where it really is, and Home's fresh
      // scroller starts at the top. Re-sync both, or Home would show the
      // frame playback stopped on and ignore scrolling until it passed the
      // old position.
      if (isHomeRef.current && !wasHome) {
        readScroll(null)
        seekAt = video!.currentTime
      }
      wasHome = isHomeRef.current
      // Scroll-scrubbing only drives the clip on Home — everywhere else it
      // just plays normally (see the separate play/pause effect below), so
      // forcing currentTime here would fight that native playback.
      if (isHomeRef.current && ready && duration) {
        const gap = seekTo - seekAt
        if (Math.abs(gap) > 0.0008) {
          seekAt += gap * 0.115
          if (video!.readyState >= 2 && !video!.seeking) {
            try {
              video!.currentTime = seekAt
            } catch {
              // ignore — seeking can throw if the clip isn't ready yet
            }
          }
        }
      }
      if (meterRef.current) meterRef.current.style.transform = `scaleX(${progress})`
      raf = requestAnimationFrame(frame)
    }

    const unlockEvents = ['touchstart', 'pointerdown', 'wheel', 'keydown'] as const
    unlockEvents.forEach((ev) => window.addEventListener(ev, unlock, { once: true, passive: true }))
    window.addEventListener('scroll', onScroll, { capture: true, passive: true })
    window.addEventListener('resize', () => readScroll(null))
    readScroll(null)
    preload()
    raf = requestAnimationFrame(frame)

    return () => {
      video.removeEventListener('loadedmetadata', onLoadedMetadata)
      video.removeEventListener('loadeddata', start)
      video.removeEventListener('canplaythrough', start)
      video.removeEventListener('error', start)
      window.removeEventListener('scroll', onScroll, true)
      unlockEvents.forEach((ev) => window.removeEventListener(ev, unlock))
      cancelAnimationFrame(raf)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoRef, meterRef, bootBarRef, bootPctRef, isHomeRef])
}

export function Layout() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const meterRef = useRef<HTMLDivElement>(null)
  const bootBarRef = useRef<HTMLDivElement>(null)
  const bootPctRef = useRef<HTMLParagraphElement>(null)
  const [bootDone, setBootDone] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const location = useLocation()
  const isHome = location.pathname === '/'
  const isHomeRef = useRef(isHome)
  isHomeRef.current = isHome

  // Close the mobile nav menu whenever the route changes (e.g. after
  // tapping a link) rather than leaving it open over the new page.
  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location.pathname])

  // Away from Home the clip just plays on its own — slowed down for a
  // smooth "slow motion" feel — instead of being scroll-scrubbed; this
  // reacts to route changes (Layout itself never remounts) as well as to
  // the clip becoming ready for the first time while already off Home.
  const playSlowly = () => {
    const v = videoRef.current
    if (!v) return
    v.loop = true
    v.playbackRate = 0.5
    const p = v.play()
    if (p && typeof p.catch === 'function') p.catch(() => {})
  }

  useScrollScrubbedVideo(videoRef, meterRef, bootBarRef, bootPctRef, isHomeRef, () => {
    setBootDone(true)
    if (!isHomeRef.current) playSlowly()
  })

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    if (isHome) {
      v.pause()
      v.loop = false
      v.playbackRate = 1
    } else {
      playSlowly()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isHome])

  // Belt-and-braces: browsers can pause background <video> for reasons
  // outside our control (a permission prompt like the geolocation one this
  // page's chat triggers on send, switching tabs, minimizing, ...) — resume
  // automatically once the tab is visible again, on any non-Home page.
  useEffect(() => {
    function onVisibility() {
      if (!document.hidden && !isHomeRef.current) playSlowly()
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => document.removeEventListener('visibilitychange', onVisibility)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="h-screen flex flex-col relative">
      <div
        className={`fixed inset-0 z-50 flex flex-col items-center justify-center gap-4 bg-[#f2f0ec] transition-[opacity,visibility] duration-700 ${
          bootDone ? 'opacity-0 invisible' : 'opacity-100 visible'
        }`}
      >
        <div className="w-[150px] h-px bg-black/10 overflow-hidden">
          <div ref={bootBarRef} className="h-full w-full bg-[#0d0c0b] origin-left scale-x-0" />
        </div>
        <p
          ref={bootPctRef}
          className="text-[12.5px] tracking-[.05em] text-[#0d0c0b]/42 m-0"
          style={{ fontFamily: "'Inter Tight', 'Helvetica Neue', Helvetica, Arial, sans-serif" }}
        >
          LOADING 0%
        </p>
      </div>

      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none bg-[#f2f0ec]">
        <video
          ref={videoRef}
          muted
          playsInline
          preload="auto"
          disablePictureInPicture
          className="absolute top-1/2 left-1/2 w-full h-full object-cover -translate-x-1/2 -translate-y-1/2 scale-[1.02]"
          style={{ filter: 'contrast(1.02)' }}
        />
        <div
          className="absolute inset-0"
          style={{
            background:
              'linear-gradient(to bottom, rgba(242,240,236,.62) 0%, rgba(242,240,236,.12) 22%, rgba(242,240,236,.12) 78%, rgba(242,240,236,.66) 100%), radial-gradient(100% 80% at 50% 48%, rgba(242,240,236,0) 0%, rgba(242,240,236,.34) 100%), rgba(242,240,236,.20)',
          }}
        />
        <div
          className="absolute -inset-1/2 opacity-[.13] mix-blend-multiply"
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3'/></filter><rect width='140' height='140' filter='url(%23n)' opacity='.5'/></svg>\")",
          }}
        />
      </div>

      <div ref={meterRef} className="fixed top-0 left-0 z-20 h-0.5 w-full origin-left scale-x-0 bg-[#0d0c0b]/55" />

      <nav className="relative z-10 h-16 shrink-0 flex items-center gap-1 px-6 bg-gradient-to-b from-[#f2f0ec]/95 via-[#f2f0ec]/80 to-transparent text-[#0d0c0b]">
        <NavLink
          to="/"
          className="flex items-center gap-2 mr-4 shrink-0 text-[15px] tracking-tight text-[#0d0c0b]"
          end
        >
          <span className="w-[22px] h-[22px] rounded-md bg-teal-500 flex items-center justify-center text-white text-[10px]">◈</span>
          ORCA
        </NavLink>
        <div className="hidden lg:flex flex-1 items-center justify-center gap-1 overflow-x-auto min-w-0">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `shrink-0 text-[13.5px] px-2.5 py-1.5 rounded-full transition whitespace-nowrap ${
                  isActive ? 'bg-black/8 border border-black/10 font-medium' : 'text-[#0d0c0b]/90 hover:bg-black/5 border border-transparent'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
        <div className="flex-1 lg:hidden" />
        <button
          type="button"
          onClick={() => setMobileMenuOpen((v) => !v)}
          aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={mobileMenuOpen}
          className="lg:hidden shrink-0 w-10 h-10 flex items-center justify-center rounded-full hover:bg-black/5 transition"
        >
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
            {mobileMenuOpen ? (
              <path d="M3 3l12 12M15 3L3 15" />
            ) : (
              <path d="M2 4.5h14M2 9h14M2 13.5h14" />
            )}
          </svg>
        </button>
        <Link
          to="/chat"
          className="ml-3 shrink-0 inline-flex items-center h-10 px-3.5 sm:px-5 rounded-full bg-black text-white text-[13px] sm:text-[14px] font-medium transition hover:-translate-y-0.5"
        >
          Open Assistant
        </Link>
      </nav>
      {mobileMenuOpen && (
        <div className="lg:hidden absolute top-16 left-0 right-0 z-20 mx-3 rounded-2xl bg-[#f2f0ec]/97 backdrop-blur-xl border border-black/10 shadow-lg py-2 flex flex-col">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `px-5 py-3 text-[15px] transition ${
                  isActive ? 'bg-black/8 font-medium text-[#0d0c0b]' : 'text-[#0d0c0b]/85 hover:bg-black/5'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      )}
      <div className="relative z-10 flex-1 min-h-0">
        <Suspense fallback={<PageLoadingFallback />}>
          <Outlet />
        </Suspense>
      </div>
      {isHome && (
        <footer className="fixed bottom-0 left-0 right-0 z-10 flex justify-center px-6 py-3 text-[11px] tracking-[.02em] text-[#0d0c0b]/42 text-center pointer-events-none bg-gradient-to-t from-[#f2f0ec]/92 via-[#f2f0ec]/72 to-transparent">
          ORCA &nbsp;&middot;&nbsp; Smart India Hackathon 2026 — ISRO &nbsp;&middot;&nbsp; Marine Intelligence for Bharat
        </footer>
      )}
    </div>
  )
}
