import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

describe('vitest + jsdom + React Testing Library toolchain', () => {
  it('renders a component and finds it by text', () => {
    render(<div>hello from vitest</div>)
    expect(screen.getByText('hello from vitest')).toBeInTheDocument()
  })
})
