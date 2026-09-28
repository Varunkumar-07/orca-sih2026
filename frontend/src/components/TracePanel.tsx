import type { TraceStep } from '../types'

const agentColor: Record<string, string> = {
  planning_agent: 'bg-indigo-500',
  marine_data_agent: 'bg-sky-500',
  weather_agent: 'bg-amber-500',
  risk_assessment_agent: 'bg-rose-500',
  ocean_analytics: 'bg-teal-500',
  geospatial: 'bg-emerald-500',
  reporting: 'bg-violet-500',
  visualization: 'bg-fuchsia-500',
  navigation_agent: 'bg-blue-600',
  demo_fallback: 'bg-slate-500',
}

const agentLabel: Record<string, string> = {
  planning_agent: 'Planning / Orchestrator',
  marine_data_agent: 'Marine Data Discovery',
  weather_agent: 'Weather Intelligence',
  risk_assessment_agent: 'Risk Assessment',
  ocean_analytics: 'Ocean Analytics',
  geospatial: 'Geospatial Reasoning',
  reporting: 'Reporting',
  visualization: 'Visualization',
  navigation_agent: 'Navigation Agent (A*)',
  demo_fallback: 'Demo Fallback',
}

function formatTime(iso: string) {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return iso
  }
}

export function TracePanel({
  trace,
  visibleCount,
}: {
  trace: TraceStep[]
  visibleCount: number
}) {
  if (trace.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center text-slate-400">
        <div className="w-10 h-10 rounded-full border-2 border-dashed border-slate-300 flex items-center justify-center mb-3">
          <span className="text-lg">≡</span>
        </div>
        <p className="text-sm font-medium">No trace yet</p>
        <p className="text-xs mt-1">Send a query to see the multi-agent execution flow.</p>
        <div className="mt-4 text-[10px] tracking-widest uppercase text-slate-400">
          Planning → Marine/Weather → Risk → Analytics/Geospatial → Reporting
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-white/40 bg-white/15 flex items-center justify-between">
        <h3 className="text-xs font-semibold tracking-widest uppercase text-slate-600">
          Live Reasoning Trace
        </h3>
        <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-900 text-white font-mono">
          {Math.min(visibleCount, trace.length)}/{trace.length} steps
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {trace.slice(0, visibleCount).map((step, idx) => (
          <div
            key={idx}
            className="rounded-lg border border-slate-200 bg-white shadow-sm overflow-hidden animate-[fadeIn_0.3s_ease]"
          >
            <div className="flex items-start gap-3 p-3">
              <div
                className={`w-2.5 h-2.5 rounded-full mt-1.5 flex-shrink-0 ${agentColor[step.agent_name] ?? 'bg-slate-400'}`}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-bold text-slate-800">
                    {idx + 1}. {agentLabel[step.agent_name] ?? step.agent_name}
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    {formatTime(step.timestamp)}
                  </span>
                </div>
                <div className="text-[10px] font-mono text-slate-500 mt-0.5 break-all">
                  {step.agent_name}
                </div>
              </div>
              <span className="text-[11px] font-mono bg-slate-100 border border-slate-200 px-1.5 py-0.5 rounded">
                #{idx + 1}
              </span>
            </div>

            <div className="px-3 pb-3 space-y-2">
              <div className="rounded bg-slate-50 border border-slate-200 p-2">
                <div className="text-[10px] font-semibold tracking-wide uppercase text-slate-500 mb-1">
                  Input
                </div>
                <div className="text-xs font-mono text-slate-700 break-words leading-relaxed">
                  {step.input_summary}
                </div>
              </div>
              <div className="rounded bg-emerald-50 border border-emerald-200 p-2">
                <div className="text-[10px] font-semibold tracking-wide uppercase text-emerald-700 mb-1">
                  Output
                </div>
                <div className="text-xs font-mono text-emerald-900 break-words leading-relaxed">
                  {step.output_summary}
                </div>
              </div>
            </div>
          </div>
        ))}

        {visibleCount < trace.length && (
          <div className="flex items-center gap-2 text-xs text-slate-400 py-2">
            <div className="h-px flex-1 bg-slate-200" />
            <span className="animate-pulse">executing…</span>
            <div className="h-px flex-1 bg-slate-200" />
          </div>
        )}
      </div>

      <div className="px-3 py-2 border-t border-white/40 bg-white/15 text-[11px] text-slate-500">
        Decomposition: <span className="font-mono">Planning → Marine/Weather → Risk → Analytics/Geospatial → Reporting/Visualization</span>
      </div>
    </div>
  )
}
