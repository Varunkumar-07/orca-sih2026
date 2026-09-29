"""
Language Agent — Team Claude, wraps the pipeline boundary (input + output).

Input side: detects the language of the raw user query and translates it to
English before it reaches run_planning_agent — every reasoning agent keeps
operating on English internally, unchanged, regardless of what language the
conversation is conducted in. Output side: translates the final English
answer_text back into that same detected language before it's returned.

Language detection: Bhashini's officially documented pipeline task types are
only "asr", "translation", and "tts" (confirmed against
https://bhashini.gitbook.io/bhashini-apis as of this writing) — there is no
documented language-identification pipeline task in the public API
reference, only a separately-hosted "Text Language Detection" model page
with no documented pipeline integration. Rather than depend on an
undocumented/unstable endpoint for a step that doesn't need one, detection
here is a local Unicode-script heuristic: an ASCII-only query is English
(skip translation entirely — the common case costs nothing); a non-ASCII
query maps its first recognized non-ASCII character's Unicode block to the
single most common Bhashini-supported language for that script (e.g.
Devanagari -> "hi"). This is correct for the large majority of real queries
and is a defensible best guess for a script shared by multiple languages
(Devanagari is also Marathi/Nepali/Sanskrit) rather than a second network
call to an endpoint that isn't part of Bhashini's stable documented surface.

Translation: the real Bhashini/ULCA two-step flow — a Pipeline Config Call
(resolves, for one (task_type, sourceLanguage, targetLanguage) combination,
the compute endpoint URL, its auth header, and the serviceId to send with
the actual request; cached per combination for the process lifetime) followed
by a Pipeline Compute Call (the actual translation). Request/response shapes
match https://bhashini.gitbook.io/bhashini-apis/pipeline-compute-call and the
community reference implementation this was checked against
(github.com/AdityaKukreti/bhashini-api).

Never raises across the agent boundary — any failure (missing API key,
network error, unsupported language pair, malformed response) degrades to
returning the original text untranslated, logged via this module's logger,
exactly like every other agent in this pipeline degrades on failure rather
than breaking the response.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_ULCA_CONFIG_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
# MeitY's public default pipeline (ASR + Translation + TTS task combinations
# across Bhashini's supported Indian-language pairs) — the same pipelineId
# used by Bhashini's own reference clients and community integrations; not a
# secret, only the API keys used alongside it are.
_DEFAULT_PIPELINE_ID = "64392f96daac500b55c543cd"
_REQUEST_TIMEOUT_S = 15

# Unicode block -> single most common Bhashini-supported ISO-639 code for
# that script. Ordered narrowest-relevant-range first; only the scripts
# Bhashini actually supports translation for are listed.
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0600, 0x06FF, "ur"),  # Arabic block, used here for Urdu
    (0x0900, 0x097F, "hi"),  # Devanagari — Hindi (also Marathi/Nepali/Sanskrit)
    (0x0980, 0x09FF, "bn"),  # Bengali
    (0x0A00, 0x0A7F, "pa"),  # Gurmukhi — Punjabi
    (0x0A80, 0x0AFF, "gu"),  # Gujarati
    (0x0B00, 0x0B7F, "or"),  # Oriya
    (0x0B80, 0x0BFF, "ta"),  # Tamil
    (0x0C00, 0x0C7F, "te"),  # Telugu
    (0x0C80, 0x0CFF, "kn"),  # Kannada
    (0x0D00, 0x0D7F, "ml"),  # Malayalam
]

# (task_type, source_lang, target_lang) -> resolved pipeline config. Module-
# level, memoized for the process lifetime — same pattern as geospatial.py's
# _restricted_areas_cache: one successful config call per language pair
# covers every subsequent translation for that pair.
_pipeline_config_cache: dict[tuple[str, str, str], dict] = {}

# Verdict icons reporting.py starts an answer's headline with (see
# translate_output_from_english for why they're handled separately).
_VERDICT_ICONS = ("✅", "⚠️", "❓", "⛔")


def _detect_language(text: str) -> str:
    """Local Unicode-script heuristic — see module docstring for why this
    isn't a Bhashini API call. Returns "en" for ASCII-only text or any
    script this table doesn't recognize (never guesses wrong on purpose)."""
    for ch in text:
        if ord(ch) < 128:
            continue
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= ord(ch) <= hi:
                return lang
    return "en"


def _post_json_once(url: str, body: dict, headers: dict) -> dict | None:
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", **headers},
        )
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        logger.warning("language_agent: HTTP %s from %s: %s", exc.code, url, body_text[:300])
        return None
    except Exception as exc:  # noqa: BLE001 - network/parse errors must never raise
        logger.warning("language_agent: request to %s failed: %s", url, exc)
        return None


def _post_json(url: str, body: dict, headers: dict) -> dict | None:
    """POST a JSON body and parse the JSON response, with one retry on
    failure. Never raises — returns None if both attempts fail.

    Bhashini's endpoints have shown occasional transient failures (observed
    live: a request that fails once succeeds immediately on a bare retry,
    no backoff needed) — the same class of intermittent flakiness already
    seen from Groq elsewhere in this pipeline. One retry meaningfully cuts
    how often a real, working translation falls back to untranslated text
    over a one-off blip, without masking a genuinely broken endpoint (the
    second failure's specific cause is still logged).
    """
    result = _post_json_once(url, body, headers)
    if result is not None:
        return result
    logger.info("language_agent: retrying %s after first attempt failed", url)
    return _post_json_once(url, body, headers)


def _get_pipeline_config(task_type: str, source_lang: str, target_lang: str) -> dict | None:
    """Pipeline Config Call for one (task_type, source_lang, target_lang)
    combination. Returns None (triggering the caller's untranslated
    fallback) if BHASHINI_USER_ID / BHASHINI_ULCA_API_KEY aren't set, or on
    any request/response failure.
    """
    cache_key = (task_type, source_lang, target_lang)
    if cache_key in _pipeline_config_cache:
        return _pipeline_config_cache[cache_key]

    user_id = os.getenv("BHASHINI_USER_ID", "")
    ulca_api_key = os.getenv("BHASHINI_ULCA_API_KEY", "")
    if not user_id or not ulca_api_key or ulca_api_key.startswith("your-"):
        return None

    request_body = {
        "pipelineTasks": [
            {
                "taskType": task_type,
                "config": {"language": {"sourceLanguage": source_lang, "targetLanguage": target_lang}},
            }
        ],
        "pipelineRequestConfig": {"pipelineId": _DEFAULT_PIPELINE_ID},
    }
    data = _post_json(_ULCA_CONFIG_URL, request_body, headers={"userID": user_id, "ulcaApiKey": ulca_api_key})
    if data is None:
        return None

    try:
        endpoint = data["pipelineInferenceAPIEndPoint"]
        auth = endpoint["inferenceApiKey"]
        service_id = data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
        config = {
            "callback_url": endpoint["callbackUrl"],
            "auth_header_name": auth["name"],
            "auth_header_value": auth["value"],
            "service_id": service_id,
        }
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning(
            "language_agent: unexpected pipeline config response shape for %s: %s", cache_key, exc
        )
        return None

    _pipeline_config_cache[cache_key] = config
    return config


def translate(text: str, source_lang: str, target_lang: str) -> tuple[str, bool]:
    """Translate `text` from source_lang to target_lang via Bhashini.

    Translates line-by-line (each non-empty line as its own item in one
    batched compute call) rather than sending the whole block as one
    string. Sending it as one string was confirmed live to make Bhashini
    silently collapse every newline into a space — reporting.py's
    answer_text is deliberately line-structured (Query / Location /
    headline / supporting details, one fact per line), and losing that
    structure broke the frontend's per-line rendering for every non-English
    response. Bhashini's compute API accepts multiple {"source": ...}
    items per call and returns them in the same order, so line boundaries
    survive translation instead.

    Returns (result_text, succeeded). succeeded is True when source_lang ==
    target_lang (nothing to do, not a failure) or a real translation came
    back; False on any failure (missing API key, network error, unsupported
    language pair, malformed response) — result_text is the original `text`
    unchanged in that case. Never raises. The explicit success flag (rather
    than callers guessing from whether the text changed) is what lets
    translate_output_from_english report an honest response_language: never
    claiming a translation that didn't actually happen.
    """
    if not text or not text.strip() or source_lang == target_lang:
        return text, True

    config = _get_pipeline_config("translation", source_lang, target_lang)
    if config is None:
        return text, False

    lines = text.split("\n")
    # (original_index, line) for every non-blank line — blank lines are
    # never sent for translation, just re-inserted verbatim by position.
    to_translate = [(i, line) for i, line in enumerate(lines) if line.strip()]
    if not to_translate:
        return text, True

    request_body = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {"sourceLanguage": source_lang, "targetLanguage": target_lang},
                    "serviceId": config["service_id"],
                },
            }
        ],
        "inputData": {"input": [{"source": line} for _i, line in to_translate]},
    }
    data = _post_json(
        config["callback_url"],
        request_body,
        headers={config["auth_header_name"]: config["auth_header_value"]},
    )
    if data is None:
        return text, False

    try:
        outputs = data["pipelineResponse"][0]["output"]
        if len(outputs) != len(to_translate):
            raise ValueError(f"expected {len(to_translate)} outputs, got {len(outputs)}")
        result_lines = list(lines)
        for (original_index, _line), output in zip(to_translate, outputs):
            result_lines[original_index] = output["target"]
        return "\n".join(result_lines), True
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("language_agent: unexpected compute response shape (%s->%s): %s", source_lang, target_lang, exc)
        return text, False


def translate_input_to_english(query_text: str) -> tuple[str, str]:
    """Input-side Language Agent: detect the query's language and translate
    to English if it isn't already.

    Returns (english_text, detected_language). Detection never fails (local
    heuristic) and is reported accurately regardless of whether the
    translation call itself succeeds — if translation fails, english_text is
    simply the original (untranslated) text, degrading gracefully rather
    than blocking the pipeline, same as every other agent's failure mode.
    """
    detected = _detect_language(query_text)
    if detected == "en":
        return query_text, "en"
    english_text, _ok = translate(query_text, detected, "en")
    return english_text, detected


def translate_output_from_english(answer_text: str, target_lang: str) -> tuple[str, str]:
    """Output-side Language Agent: translate the final English answer_text
    back into target_lang.

    Returns (text, actual_language). actual_language is "en" whenever the
    returned text is genuinely in English — target_lang was "en"/unset, or
    translation was attempted but failed — so a caller populating
    FinalResponse.response_language from this never claims a translation
    that didn't actually happen.
    """
    if not target_lang or target_lang == "en":
        return answer_text, "en"

    # Bhashini drops the verdict icon (✅ / ⚠️ / ❓ / ⛔) that starts the
    # headline line, and the chat UI keys the whole verdict card off that
    # icon — so translated answers used to render as plain text. Send each
    # headline without its icon and put the icon back afterwards; translate()
    # maps lines back by position, so the icon lands on the same line.
    lines = answer_text.split("\n")
    icons: dict[int, str] = {}
    for i, line in enumerate(lines):
        icon = next((ic for ic in _VERDICT_ICONS if line.startswith(ic)), None)
        if icon is not None:
            icons[i] = icon
            lines[i] = line[len(icon):].strip()
    translated, ok = translate("\n".join(lines), "en", target_lang)
    if not ok:
        return answer_text, "en"
    out = translated.split("\n")
    for i, icon in icons.items():
        out[i] = f"{icon} {out[i].strip()}"
    return "\n".join(out), target_lang
