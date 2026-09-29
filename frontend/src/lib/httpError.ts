// User-facing error text for failed /api/* calls. Pages used to render the
// raw failure (`500 Internal Server Error`, a response body, or a bare
// "Failed to fetch") plus a dev-only hint — "Is the backend running on
// :8000?" — that also showed up in production. Pages now show a short
// message from friendlyError(); the raw detail goes to the console.

export class HttpError extends Error {
  readonly status: number
  // Our own backend's `detail` for a 400 (e.g. "invalid variables: x") —
  // hand-written validation text, safe and useful to show as-is.
  readonly detail: string | null

  constructor(status: number, statusText: string, detail: string | null = null) {
    super(`${status} ${statusText}`.trim())
    this.name = 'HttpError'
    this.status = status
    this.detail = detail
  }
}

async function readDetail(res: Response): Promise<string | null> {
  try {
    const body: unknown = await res.json()
    if (body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string') {
      return body.detail
    }
  } catch {
    // Not JSON (or no body) — nothing safe to show.
  }
  return null
}

/** Throws an HttpError for a non-2xx response; returns it unchanged otherwise. */
export async function ensureOk(res: Response): Promise<Response> {
  if (res.ok) return res
  const detail = res.status === 400 ? await readDetail(res) : null
  throw new HttpError(res.status, res.statusText ?? '', detail)
}

export function friendlyError(e: unknown): string {
  console.error(e)
  if (e instanceof HttpError) {
    if (e.status === 400 && e.detail) return e.detail
    if (e.status === 404) return 'The requested item was not found.'
    if (e.status === 429) return 'Too many requests right now — please wait a minute and try again.'
    if (e.status === 502 || e.status === 503 || e.status === 504) {
      return 'The server is temporarily unavailable (it may be starting up) — please try again in a minute.'
    }
    if (e.status >= 500) return 'Something went wrong on the server — please try again shortly.'
    return 'The request could not be completed — please try again.'
  }
  // fetch() rejects with a TypeError when the server can't be reached at all.
  if (e instanceof TypeError) {
    return "Couldn't reach the server — check your connection, or try again in a minute if it's starting up."
  }
  return 'Something went wrong — please try again.'
}

/** Local-dev-only troubleshooting hint; empty in production builds. */
export function devBackendHint(): string {
  return import.meta.env.DEV ? ' (dev: is the backend running on :8000?)' : ''
}
