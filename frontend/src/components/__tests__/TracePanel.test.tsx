import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { TracePanel } from '../TracePanel'
import type { TraceStep } from '../../types'

function step(overrides: Partial<TraceStep> = {}): TraceStep {
  return {
    agent_name: 'planning_agent',
    input_summary: 'query text',
    output_summary: 'plan produced',
    timestamp: '2026-01-01T00:00:00Z',
    ...overrides,
  }
}

describe('TracePanel — empty state', () => {
  it('shows the "no trace yet" placeholder when trace is empty', () => {
    render(<TracePanel trace={[]} visibleCount={0} />)
    expect(screen.getByText('No trace yet')).toBeInTheDocument()
  })
})

describe('TracePanel — with steps', () => {
  const trace = [
    step({ agent_name: 'planning_agent', output_summary: 'planned' }),
    step({ agent_name: 'marine_data_agent', output_summary: 'zones found' }),
    step({ agent_name: 'unknown_future_agent', output_summary: 'fallback label' }),
  ]

  it('only renders steps up to visibleCount, and shows the counter as N/total', () => {
    render(<TracePanel trace={trace} visibleCount={2} />)

    expect(screen.getByText('2/3 steps')).toBeInTheDocument()
    expect(screen.getByText('planned')).toBeInTheDocument()
    expect(screen.getByText('zones found')).toBeInTheDocument()
    expect(screen.queryByText('fallback label')).not.toBeInTheDocument()
  })

  it('shows the "executing…" indicator while more steps remain than are visible', () => {
    render(<TracePanel trace={trace} visibleCount={1} />)
    expect(screen.getByText('executing…')).toBeInTheDocument()
  })

  it('hides the "executing…" indicator once every step is visible', () => {
    render(<TracePanel trace={trace} visibleCount={3} />)
    expect(screen.queryByText('executing…')).not.toBeInTheDocument()
  })

  it('falls back to the raw agent_name when there is no known display label', () => {
    render(<TracePanel trace={trace} visibleCount={3} />)
    // "unknown_future_agent" has no entry in TracePanel's agentLabel map, so
    // the numbered heading falls back to the raw name — appears twice (once
    // as the heading fallback, once as the mono sub-label under it).
    expect(screen.getAllByText(/unknown_future_agent/).length).toBeGreaterThan(0)
  })

  it('numbers steps in order starting from 1', () => {
    render(<TracePanel trace={trace} visibleCount={3} />)
    expect(screen.getByText(/1\. Planning \/ Orchestrator/)).toBeInTheDocument()
    expect(screen.getByText(/2\. Marine Data Discovery/)).toBeInTheDocument()
  })
})
