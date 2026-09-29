import { afterEach, describe, expect, it, vi } from 'vitest'
import { HttpError, devBackendHint, ensureOk, friendlyError } from '../httpError'

function response(status: number, body: unknown = null, statusText = 'Error') {
  return { ok: status >= 200 && status < 300, status, statusText, json: async () => body } as unknown as Response
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
})

describe('ensureOk', () => {
  it('passes a 2xx response through unchanged', async () => {
    const res = response(200)
    await expect(ensureOk(res)).resolves.toBe(res)
  })

  it('throws an HttpError carrying the status', async () => {
    await expect(ensureOk(response(503))).rejects.toMatchObject({ name: 'HttpError', status: 503 })
  })

  it("keeps a 400's backend-written detail", async () => {
    await expect(ensureOk(response(400, { detail: 'invalid variables: foo' }))).rejects.toMatchObject({
      detail: 'invalid variables: foo',
    })
  })
})

describe('friendlyError', () => {
  it.each([
    [500, /Something went wrong on the server/],
    [503, /temporarily unavailable/],
    [429, /Too many requests/],
    [404, /not found/],
  ])('maps HTTP %i to a clean message', (status, expected) => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const message = friendlyError(new HttpError(status, 'Error'))
    expect(message).toMatch(expected)
    expect(message).not.toMatch(/\d{3}/)
  })

  it('shows a 400 detail from our own backend as-is', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(friendlyError(new HttpError(400, 'Bad Request', 'invalid variables: foo'))).toBe('invalid variables: foo')
  })

  it('maps a network failure (fetch TypeError) to a reachability message', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(friendlyError(new TypeError('Failed to fetch'))).toMatch(/Couldn't reach the server/)
  })

  it('never echoes an arbitrary error message', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(friendlyError(new Error('https://api.example.com/secret?x=1 blew up'))).not.toMatch(/https?:/)
  })

  it('logs the raw error to the console for debugging', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const err = new HttpError(500, 'Internal Server Error')
    friendlyError(err)
    expect(spy).toHaveBeenCalledWith(err)
  })
})

describe('devBackendHint', () => {
  it('is empty in production builds', () => {
    vi.stubEnv('DEV', false)
    expect(devBackendHint()).toBe('')
  })

  it('mentions the local backend port in dev', () => {
    vi.stubEnv('DEV', true)
    expect(devBackendHint()).toMatch(/:8000/)
  })
})
