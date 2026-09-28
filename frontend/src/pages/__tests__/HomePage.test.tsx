import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { HomePage } from '../HomePage'

describe('HomePage', () => {
  it('renders the ORCA hero and a link into the chat assistant', () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>,
    )

    expect(screen.getByText('ORCA')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Open Assistant/ })).toHaveAttribute('href', '/chat')
  })

  it('renders all three scroll panels with their eyebrow, heading, and CTA link', () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>,
    )

    const expected: [string, string, string][] = [
      ['Smart India Hackathon 2026 · ISRO', 'ORCA', '/chat'],
      ['Assistant · Zones Explorer · Weather', 'Ask. Explore. Check the weather.', '/zones'],
      ['Route Planner · Alerts & Advisories · Analytics Dashboard', 'Plan safe routes.', '/route'],
    ]

    for (const [eyebrow, heading, href] of expected) {
      expect(screen.getByText(eyebrow)).toBeInTheDocument()
      const headingEl = screen.getByRole('heading', { name: heading })
      const panel = headingEl.closest('div')
      expect(panel?.querySelector('a')).toHaveAttribute('href', href)
    }
  })
})
