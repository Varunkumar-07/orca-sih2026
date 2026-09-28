"""
Shared Groq JSON-mode call — the one low-level piece genuinely identical
across every reasoning agent's classification/reasoning step
(weather_agent._classify_intent, marine_data_agent._classify_intent,
planning_agent._classify_query, risk_assessment_agent._reason_over_data):
build an AsyncGroq client, call chat.completions.create with the same fixed
params, parse the JSON response body.

What's NOT shared, and stays in each caller: the system prompt, the user
content (some pass the raw query text, risk_assessment_agent passes a
JSON-encoded evidence payload instead), how many tokens to budget, and —
importantly — how to react to a failure. Some callers catch internally and
degrade to a safe default (weather/marine's intent classifiers, planning's
query classifier); others let it propagate to their own agent-level
try/except (risk_assessment_agent). This helper only makes the call and
parses JSON; it does not decide what a failure means for the caller.
"""

import asyncio
import json
import logging
import os

from groq import APIConnectionError, AsyncGroq, RateLimitError

logger = logging.getLogger("orca.groq_client")

_GROQ_MODEL = "openai/gpt-oss-20b"

# Retried transient failures only — a rate limit (429) or a connection-level
# issue (APITimeoutError is a subclass of this) are worth one or two quick
# retries since they're often gone a moment later; everything else (auth,
# model errors, invalid JSON-mode output) is not retried, exactly as before.
_RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError)
_MAX_RETRIES = 2  # up to 3 total attempts
_RETRY_BASE_DELAY_SECONDS = 0.5


async def call_groq_json(
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    model: str = _GROQ_MODEL,
) -> dict:
    """Raises RuntimeError if GROQ_API_KEY isn't set, or whatever the Groq
    SDK / json.loads raises once retries (see _RETRYABLE_EXCEPTIONS) are
    exhausted — rate limit, timeout, auth, an empty/invalid JSON completion
    (confirmed to happen occasionally in production). Never catches on the
    caller's behalf beyond that bounded, transient-only retry."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    client = AsyncGroq(api_key=api_key)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=max_tokens,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except _RETRYABLE_EXCEPTIONS:
            if attempt == _MAX_RETRIES:
                raise
            delay = _RETRY_BASE_DELAY_SECONDS * (2**attempt)
            logger.warning(
                "Groq call failed (attempt %d/%d), retrying in %.1fs",
                attempt + 1,
                _MAX_RETRIES + 1,
                delay,
            )
            await asyncio.sleep(delay)
