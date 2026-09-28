"""Fetch real MPA (Marine Protected Area) boundary polygons for Indian
coastal waters from the Protected Planet (WDPA) API and cache them locally
as GeoJSON.

This is a one-time (or periodically re-runnable) fetch — it is NOT called
from the request path. geospatial.py only ever reads the cached file this
script writes; it never calls Protected Planet live.

Usage (from the project root, so backend/.env is picked up the same way
main.py finds it):

    python -m backend.scripts.fetch_mpa_boundaries

Requires PROTECTED_PLANET_API_KEY in backend/.env — get a free key at
https://api.protectedplanet.net/request.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.protectedplanet.net/v4/protected_areas"
COUNTRY = "IND"
PER_PAGE = 50
REQUEST_DELAY_S = 0.5
MAX_RETRIES = 3
TIMEOUT_S = 30
# Protected Planet sits behind Cloudflare, which blocks urllib's default
# "Python-urllib/x.y" user agent as a bot signature (HTTP 403, Cloudflare
# error 1010). A normal browser-style UA avoids that.
USER_AGENT = (
    "Mozilla/5.0 (compatible; ORCA-MPA-fetch/1.0; "
    "+https://github.com/protectedplanet-fetch-script)"
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR.parent / "agents" / "deterministic" / "data" / "mpa_boundaries.geojson"


def _get(url: str) -> dict:
    """GET a URL and parse JSON, with retries on transient failures."""
    last_err: Exception | None = None
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"HTTP {exc.code} from {url}: {body[:500]}")
            if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                time.sleep(2**attempt)
                continue
            raise last_err
        except urllib.error.URLError as exc:
            last_err = RuntimeError(f"Network error calling {url}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(2**attempt)
                continue
            raise last_err
    raise last_err  # pragma: no cover — loop always returns or raises


def _fetch_search_page(token: str, page: int) -> dict:
    params = {
        "token": token,
        "country": COUNTRY,
        "marine": "true",
        "per_page": str(PER_PAGE),
        "page": str(page),
        "with_geometry": "true",
    }
    url = f"{API_BASE}/search?{urllib.parse.urlencode(params)}"
    return _get(url)


def _fetch_detail_geometry(token: str, site_id) -> dict | None:
    params = {"token": token, "with_geometry": "true"}
    url = f"{API_BASE}/{site_id}?{urllib.parse.urlencode(params)}"
    data = _get(url)
    area = data.get("protected_area", data)
    return area.get("geojson")


def _extract_areas(page_data: dict) -> list[dict]:
    return page_data.get("protected_areas") or page_data.get("protected_area_parcels") or []


def fetch_all(token: str) -> list[dict]:
    """Return a list of GeoJSON Feature dicts, one per protected area.

    The search endpoint is asked for inline geometry (`with_geometry=true`)
    first, since that's one call per page instead of one per site. If a
    given result doesn't carry a `geojson` field (API behavior here isn't
    fully pinned down without a live token), fall back to the per-site
    detail endpoint for that one area.
    """
    features: list[dict] = []
    page = 1
    total_pages: int | None = None

    while True:
        print(f"Fetching page {page}{f'/{total_pages}' if total_pages else ''}...")
        data = _fetch_search_page(token, page)
        areas = _extract_areas(data)
        if not areas:
            break

        for area in areas:
            # name_english is the curated designation name; the raw `name`
            # field is sometimes a garbled/mistransliterated local-language
            # artifact (e.g. WDPA site_id 555795353 has name="Mannar
            # Valaiguda in Tamil" but name_english="Gulf of Mannar Marine
            # Biosphere Reserve") — prefer it whenever present.
            name = area.get("name_english") or area.get("name") or "Unnamed MPA"
            site_id = area.get("site_id") or area.get("id") or area.get("wdpa_id")
            geojson = area.get("geojson")

            if not geojson:
                try:
                    geojson = _fetch_detail_geometry(token, site_id)
                except Exception as exc:
                    print(f"  ! skipping '{name}' (site_id={site_id}): {exc}")
                    continue
                time.sleep(REQUEST_DELAY_S)

            geometry = (geojson or {}).get("geometry") if isinstance(geojson, dict) else None
            if not geometry:
                print(f"  ! skipping '{name}' (site_id={site_id}): no geometry in response")
                continue

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "name": name,
                        "site_id": site_id,
                        "designation": area.get("designation"),
                        "iucn_category": area.get("iucn_category"),
                        "marine": area.get("marine"),
                        "source": "protectedplanet.net WDPA API v4",
                    },
                    "geometry": geometry,
                }
            )
            # WDPA carries some sites (often internationally-designated
            # ones without a digitized boundary) as a bare representative
            # Point rather than a polygon, with no area of its own.
            # geospatial.py buffers every cached geometry outward at load
            # time, which turns this into a circular protective radius —
            # flagged here just so that's not a silent surprise.
            note = " [Point geometry — geospatial.py gives it a buffered radius, no real boundary]" if geometry.get("type") == "Point" else ""
            print(f"  + {name} (site_id={site_id}){note}")

        pagination = data.get("pagination") or {}
        total_pages = pagination.get("total_pages", total_pages)
        if total_pages and page >= total_pages:
            break
        if len(areas) < PER_PAGE and not total_pages:
            break
        page += 1
        time.sleep(REQUEST_DELAY_S)

    return features


def main() -> int:
    token = os.environ.get("PROTECTED_PLANET_API_KEY", "").strip()
    if not token:
        print(
            "ERROR: PROTECTED_PLANET_API_KEY is not set. Add it to backend/.env "
            "(get a free key at https://api.protectedplanet.net/request).",
            file=sys.stderr,
        )
        return 1

    try:
        features = fetch_all(token)
    except Exception as exc:
        print(f"ERROR: fetch failed: {exc}", file=sys.stderr)
        return 1

    if not features:
        print("ERROR: fetched zero MPA boundaries — refusing to overwrite the cache.", file=sys.stderr)
        return 1

    collection = {
        "type": "FeatureCollection",
        "metadata": {
            "source": "Protected Planet (WDPA) API v4",
            "query": {"country": COUNTRY, "marine": True},
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "feature_count": len(features),
        },
        "features": features,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH.with_suffix(".geojson.tmp")
    tmp_path.write_text(json.dumps(collection, indent=2))
    tmp_path.replace(OUTPUT_PATH)

    print(f"\nSaved {len(features)} MPA boundaries to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
