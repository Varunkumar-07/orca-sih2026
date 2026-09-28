import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Unmount every component tree after each test — without this, a
// component left mounted by one test (e.g. an unresolved async fetch
// still updating state) can leak into the next test's assertions.
afterEach(() => {
  cleanup()
})
