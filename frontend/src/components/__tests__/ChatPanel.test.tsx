import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ChatPanel } from '../ChatPanel'

function baseProps(overrides: Partial<Parameters<typeof ChatPanel>[0]> = {}) {
  return {
    messages: [],
    input: '',
    setInput: vi.fn(),
    onSend: vi.fn(),
    loading: false,
    onExampleClick: vi.fn(),
    ...overrides,
  }
}

describe('ChatPanel — empty state', () => {
  it('shows the welcome message and example queries when there are no messages', () => {
    render(<ChatPanel {...baseProps()} />)
    expect(screen.getByText('Welcome to ORCA')).toBeInTheDocument()
    expect(screen.getByText(/is it safe to go out tomorrow near Chennai/)).toBeInTheDocument()
  })

  it('clicking an example query calls onExampleClick with its text', async () => {
    const user = userEvent.setup()
    const onExampleClick = vi.fn()
    render(<ChatPanel {...baseProps({ onExampleClick })} />)

    await user.click(screen.getByText(/can I fish near Gulf of Mannar\?/))
    expect(onExampleClick).toHaveBeenCalledWith('can I fish near Gulf of Mannar?')
  })
})

describe('ChatPanel — sending', () => {
  it('Send is disabled when input is blank, enabled once there is text', () => {
    const { rerender } = render(<ChatPanel {...baseProps({ input: '' })} />)
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()

    rerender(<ChatPanel {...baseProps({ input: 'hello' })} />)
    expect(screen.getByRole('button', { name: 'Send' })).not.toBeDisabled()
  })

  it('clicking Send calls onSend', async () => {
    const user = userEvent.setup()
    const onSend = vi.fn()
    render(<ChatPanel {...baseProps({ input: 'is it safe?', onSend })} />)

    await user.click(screen.getByRole('button', { name: 'Send' }))
    expect(onSend).toHaveBeenCalledTimes(1)
  })

  it('pressing Enter (without shift) sends, but Shift+Enter does not', async () => {
    const user = userEvent.setup()
    const onSend = vi.fn()
    render(<ChatPanel {...baseProps({ input: 'is it safe?', onSend })} />)

    const inputEl = screen.getByPlaceholderText(/Ask e.g\./)
    await user.type(inputEl, '{Shift>}{Enter}{/Shift}')
    expect(onSend).not.toHaveBeenCalled()

    await user.type(inputEl, '{Enter}')
    expect(onSend).toHaveBeenCalledTimes(1)
  })

  it('disables the input and Send button while loading', () => {
    render(<ChatPanel {...baseProps({ input: 'hello', loading: true })} />)
    expect(screen.getByPlaceholderText(/Ask e.g\./)).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
  })

  it('shows the reasoning indicator while loading', () => {
    render(<ChatPanel {...baseProps({ loading: true })} />)
    expect(screen.getByText(/ORCA is reasoning/)).toBeInTheDocument()
  })
})

describe('ChatPanel — message rendering', () => {
  it('renders a user message verbatim, without the structured-answer formatting', () => {
    render(<ChatPanel {...baseProps({ messages: [{ role: 'user', text: 'is it safe?' }] })} />)
    expect(screen.getByText('is it safe?')).toBeInTheDocument()
  })

  it('formats a 3+ line ORCA answer into query/location header, headline, and bullet details', () => {
    const text = 'Query: is it safe?\nLocation: 13.0800, 80.2700\n✅ Safe to go (confidence 88%) — Calm seas.\nWind: 12 km/h'
    render(<ChatPanel {...baseProps({ messages: [{ role: 'orca', text }] })} />)

    expect(screen.getByText('Query: is it safe?')).toBeInTheDocument()
    expect(screen.getByText('Location: 13.0800, 80.2700')).toBeInTheDocument()
    expect(screen.getByText(/Safe to go \(confidence 88%\)/)).toBeInTheDocument()
    expect(screen.getByText('Wind:')).toBeInTheDocument()
    expect(screen.getByText('12 km/h')).toBeInTheDocument()
  })

  it('falls back to raw text (no structure) for a short ORCA answer under 3 lines', () => {
    render(<ChatPanel {...baseProps({ messages: [{ role: 'orca', text: 'just one short line' }] })} />)
    expect(screen.getByText('just one short line')).toBeInTheDocument()
  })

  it('shows the language pill only when a language is present', () => {
    const { rerender } = render(<ChatPanel {...baseProps()} />)
    expect(screen.queryByText('active languages')).not.toBeInTheDocument()

    rerender(<ChatPanel {...baseProps({ detectedLanguage: 'Telugu', responseLanguage: 'Telugu' })} />)
    expect(screen.getByText('active languages')).toBeInTheDocument()
    expect(screen.getAllByText('Telugu').length).toBeGreaterThan(0)
  })
})
