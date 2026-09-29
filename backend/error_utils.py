"""Log-friendly exception descriptions.

str(exc) alone is not enough for upstream-failure logs: several of the
exceptions this backend sees in production stringify to an EMPTY string —
most importantly asyncio.TimeoutError from asyncio.wait_for — which left
log lines like "Chlorophyll fetch failed ... after 2 attempt(s): " with no
way to tell a timeout from an auth failure. Always lead with the type.
"""


def describe_exception(exc: BaseException | None, timeout: float | None = None) -> str:
    """"TypeName: message", or just "TypeName" when the message is empty.
    Pass `timeout` for a wait_for-bounded call so a timeout also says how
    long it waited."""
    if exc is None:
        return "no exception recorded"
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) and timeout is not None:
        return f"{name} (no result within {timeout:g}s)"
    message = str(exc).strip()
    return f"{name}: {message}" if message else name
