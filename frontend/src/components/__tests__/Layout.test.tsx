import { render, screen, waitFor } from '@testing-library/react'
import { lazy, type ReactElement } from 'react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { Layout } from '../Layout'

function renderAt(path: string, childElement = <div>page content</div>) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<div>home content</div>} />
          <Route path="/chat" element={childElement} />
          <Route path="/zones" element={childElement} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

describe('Layout — navigation', () => {
  it('renders the ORCA brand link and all 9 page links', () => {
    renderAt('/')

    expect(screen.getByRole('link', { name: /ORCA/ })).toHaveAttribute('href', '/')
    const expectedLabels = [
      'Home', 'Assistant', 'Zones Explorer', 'Weather', 'Route Planner',
      'Alerts', 'Analytics', 'Download', 'History',
    ]
    for (const label of expectedLabels) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument()
    }
  })

  it('renders the matched child route inside the Outlet', () => {
    renderAt('/', <div>should not show</div>)
    expect(screen.getByText('home content')).toBeInTheDocument()
  })

  it('highlights the active nav link and leaves the others un-highlighted', () => {
    renderAt('/zones')

    const activeLink = screen.getByRole('link', { name: 'Zones Explorer' })
    const inactiveLink = screen.getByRole('link', { name: 'Weather' })

    expect(activeLink.className).toContain('bg-black/8')
    expect(inactiveLink.className).not.toContain('bg-black/8')
  })

  it('the Home link is only active at the exact root path, not on every route (its `end` prop)', () => {
    renderAt('/zones')

    const homeLink = screen.getByRole('link', { name: 'Home' })
    expect(homeLink.className).not.toContain('bg-black/8')
  })
})

describe('Layout — Suspense boundary around the Outlet', () => {
  it('shows the loading fallback while a lazy child is pending, then the real content once it resolves — nav stays rendered throughout', async () => {
    let resolveImport!: (mod: { default: () => ReactElement }) => void
    const LazyChild = lazy(
      () =>
        new Promise<{ default: () => ReactElement }>((resolve) => {
          resolveImport = resolve
        }),
    )

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/chat" element={<LazyChild />} />
          </Route>
        </Routes>
      </MemoryRouter>,
    )

    // Nav renders immediately, before the lazy import ever resolves.
    expect(screen.getByRole('link', { name: 'Assistant' })).toBeInTheDocument()
    expect(screen.getByText('Loading…')).toBeInTheDocument()

    resolveImport({ default: () => <div>lazy child resolved</div> })

    await waitFor(() => expect(screen.getByText('lazy child resolved')).toBeInTheDocument())
    expect(screen.queryByText('Loading…')).not.toBeInTheDocument()
    // Still there, unaffected by the Outlet's own content swapping.
    expect(screen.getByRole('link', { name: 'Assistant' })).toBeInTheDocument()
  })
})
