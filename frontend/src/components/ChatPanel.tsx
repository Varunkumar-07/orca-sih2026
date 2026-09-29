type Message = { role: 'user' | 'orca'; text: string }

const VERDICT_ICONS = ['✅', '⚠️', '❓', '⛔']
// Best-effort only: bolds a "Label: content" prefix when the line happens
// to be in English. Deliberately NOT relied on for structure — Bhashini
// translates the label word itself (e.g. "Weather" -> "వాతావరణం") and often
// the punctuation after it too (Telugu renders it "వాతావరణంః", using "ః",
// not ASCII ":"), so a fixed English/ASCII pattern can't be the thing that
// decides whether a line becomes a bullet.
const LABEL_LINE = /^([A-Za-z][A-Za-z ]*):\s*(.*)$/

const VERDICT_COLOR: Record<string, string> = {
  '✅': 'text-emerald-700',
  '⚠️': 'text-amber-700',
  '❓': 'text-slate-600',
  '⛔': 'text-red-700',
}

/** Renders reporting.py's plain-text answer_text as structured, scannable
 * UI instead of one dense paragraph — works the same in every language,
 * not just English.
 *
 * _build_answer (backend/agents/deterministic/reporting.py) always emits
 * the same FIXED line order regardless of branch or language: "Query: ..."
 * first, "Location: ..." second, then a headline line starting with one of
 * VERDICT_ICONS, then supporting-detail lines, occasionally ending with a
 * closing note. Translation (language_agent.py) preserves line order and
 * count — it translates the whole block as one string but doesn't reorder
 * or merge lines — and always leaves the verdict emoji untouched. So this
 * parses by POSITION (lines 1-2) and EMOJI (the headline), never by
 * matching English label words — those get translated and would silently
 * stop matching in every non-English response otherwise. Every remaining
 * line becomes its own bullet, in whatever language it's in.
 *
 * Falls back to the raw text unchanged if the shape is too thin to be this
 * (fewer than 3 lines) rather than force structure onto text that doesn't
 * have it. */
function FormattedAnswer({ text }: { text: string }) {
  const lines = text.split('\n').map((l) => l.trim()).filter((l) => l.length > 0)

  if (lines.length < 3) {
    return <span className="whitespace-pre-wrap">{text}</span>
  }

  const [queryLine, locationLine, ...rest] = lines
  const headlineIdx = rest.findIndex((l) => VERDICT_ICONS.some((icon) => l.startsWith(icon)))

  // No verdict-icon headline anywhere means this 3+-line message isn't
  // actually _build_answer's fixed Query/Location/headline shape (e.g. a
  // demo-fallback or otherwise non-conforming message) — rendering the
  // first two lines as small "query/location" captions regardless would
  // misrepresent arbitrary text as if it had that structure.
  if (headlineIdx === -1) {
    return <span className="whitespace-pre-wrap">{text}</span>
  }

  const headline = rest[headlineIdx]
  const detailLines = [...rest.slice(0, headlineIdx), ...rest.slice(headlineIdx + 1)]
  const verdictIcon = VERDICT_ICONS.find((icon) => headline.startsWith(icon))

  return (
    <div className="space-y-2">
      <div className="text-[11px] text-slate-400 space-y-0.5">
        <div>{queryLine}</div>
        <div>{locationLine}</div>
      </div>
      {headline && (
        <div className={`font-semibold text-[15px] leading-snug ${verdictIcon ? VERDICT_COLOR[verdictIcon] : ''}`}>
          {headline}
        </div>
      )}
      {detailLines.length > 0 && (
        <ul className="space-y-1 list-disc pl-4 marker:text-slate-400">
          {detailLines.map((line, i) => {
            const match = line.match(LABEL_LINE)
            return (
              <li key={i} className="text-slate-700">
                {match && match[2] ? (
                  <>
                    <span className="font-medium text-slate-800">{match[1]}:</span> {match[2]}
                  </>
                ) : (
                  line
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

export function ChatPanel({
  messages,
  input,
  setInput,
  onSend,
  loading,
  onExampleClick,
  detectedLanguage,
  responseLanguage,
}: {
  messages: Message[]
  input: string
  setInput: (v: string) => void
  onSend: () => void
  loading: boolean
  onExampleClick: (q: string) => void
  detectedLanguage?: string | null
  responseLanguage?: string | null
}) {
  const examples = [
    'is it safe to go out tomorrow near Chennai?',
    'where is the nearest fishing zone near Chennai?',
    'can I fish near Gulf of Mannar?',
    'is it safe near Kochi with cyclone alert?',
  ]

  return (
    <div className="flex flex-col h-full bg-transparent">
      <div className="px-[18px] pt-[18px] pb-0">
        <h2 className="text-[14px] font-semibold tracking-[-.02em] text-[#0d0c0b] m-0">ORCA Chat</h2>
        <p className="text-[12.5px] text-[#0d0c0b]/64 mt-1 tracking-[.02em]">Ask in natural language · marine + weather + risk</p>
      </div>
      {(detectedLanguage || responseLanguage) && (
        <div className="px-4 py-2 bg-white/30 border-b border-white/50 flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 text-[11px] px-2.5 py-1 rounded-full bg-white border border-slate-200 text-slate-700 shadow-sm">
            <span aria-hidden>🌐</span>
            {detectedLanguage && <span className="font-medium">{detectedLanguage}</span>}
            {detectedLanguage && responseLanguage && <span className="text-slate-400">→</span>}
            {responseLanguage && <span className="font-semibold">{responseLanguage}</span>}
          </span>
          <span className="text-[11px] text-slate-500">active languages</span>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-3.5 pt-3.5 space-y-2.5 bg-transparent">
        {messages.length === 0 && (
          <div className="space-y-2">
            <div className="rounded-[14px] bg-[#fcfcfa] border border-slate-200 p-3.5 text-[13px] leading-[1.5] text-[#475569]">
              <div className="font-semibold text-[13px] text-[#0f172a] mb-1.5">Welcome to ORCA</div>
              Conversational marine intelligence for fishermen & coastal operators. Try an example:
            </div>
            <div className="grid gap-2">
              {examples.map((ex) => (
                <button
                  key={ex}
                  onClick={() => onExampleClick(ex)}
                  className="text-left text-[13px] leading-[1.4] text-[#334155] bg-white border border-slate-200 rounded-[12px] px-3 py-[11px] hover:border-[#99f6e4] hover:bg-[#f0fdfa] hover:-translate-y-px transition"
                >
                  “{ex}”
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`rounded-[14px] border px-3.5 py-3 text-[13px] leading-[1.5] break-words ${
              m.role === 'user' ? 'bg-[#f0fdfa] border-[#99f6e4] text-[#134e4a]' : 'bg-white border-slate-200 text-[#334155]'
            }`}
          >
            <strong
              className={`block text-[10px] tracking-[.06em] uppercase mb-1 ${
                m.role === 'user' ? 'text-[#0f766e]' : 'text-[#0f172a]'
              }`}
            >
              {m.role === 'user' ? 'You' : 'ORCA'}
            </strong>
            {m.role === 'orca' ? <FormattedAnswer text={m.text} /> : <span className="whitespace-pre-wrap">{m.text}</span>}
          </div>
        ))}

        {loading && (
          <div className="rounded-[14px] border border-slate-200 bg-white px-3.5 py-3 text-xs text-slate-500 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse" />
            ORCA is reasoning… (planning → marine → weather → risk → analytics → geospatial → reporting)
          </div>
        )}
      </div>

      <div className="flex gap-2.5 items-center p-3 border-t border-white/28 bg-white/10 backdrop-blur-[12px] mt-1">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !loading && input.trim()) {
              e.preventDefault()
              onSend()
            }
          }}
          disabled={loading}
          placeholder="Ask e.g. is it safe to go out tomorrow near Chennai?"
          className="flex-1 h-11 rounded-full border border-slate-200 px-4 text-[13.5px] bg-[#f8fafc] focus:outline-none focus:border-[#2dd4bf] focus:bg-white focus:ring-[3px] focus:ring-[#2dd4bf]/15 disabled:opacity-60 transition"
        />
        <button
          onClick={onSend}
          disabled={loading || !input.trim()}
          className="h-11 rounded-full bg-black text-white px-5 text-[14px] font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:-translate-y-px transition"
        >
          Send
        </button>
      </div>
      <div className="text-[11px] text-[#94a3b8] text-center px-3.5 pb-3">
        Demo mode shows sample answers · Live mode uses real-time data and AI reasoning
      </div>
    </div>
  )
}
