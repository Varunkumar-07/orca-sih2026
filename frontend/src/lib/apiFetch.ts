// Attaches X-API-Key to same-origin /api/* requests when VITE_ORCA_API_KEY is
// set at build time — the frontend half of backend/auth.py's opt-in auth.
// No-op (native fetch, untouched) when the var is unset, so local dev and
// any deploy that leaves ORCA_API_KEY unset keep working exactly as before.
const apiKey = import.meta.env.VITE_ORCA_API_KEY as string | undefined

function isApiRequest(input: RequestInfo | URL): boolean {
  const raw = input instanceof Request ? input.url : input.toString()
  try {
    return new URL(raw, window.location.origin).pathname.startsWith('/api/')
  } catch {
    return false
  }
}

export function installApiKeyFetch(): void {
  if (!apiKey) return

  const nativeFetch = window.fetch.bind(window)
  window.fetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
    if (!isApiRequest(input)) return nativeFetch(input, init)

    const headers = new Headers(init.headers)
    headers.set('X-API-Key', apiKey)
    return nativeFetch(input, { ...init, headers })
  }
}
