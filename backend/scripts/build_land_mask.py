"""
Build the compact land mask the Navigation Agent routes around.

Source: Natural Earth 1:10m land + minor islands (public domain,
https://www.naturalearthdata.com), clipped to the seas around India and
simplified to ~0.005 degrees (~500m) — well under the router's 1-2km grid
cells — so the committed file stays small and cheap to load. Output:
backend/agents/deterministic/data/land_india.geojson, read once per process
by navigation_agent.py (same pattern as the WDPA cache in geospatial.py).

Run from the project root (downloads ~11MB, writes the small clipped file):
    PYTHONPATH=. python backend/scripts/build_land_mask.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

SOURCES = [
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson",
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_minor_islands.geojson",
]
# lon/lat bounds: Arabian Sea to the Andaman Sea, Sri Lanka and the
# Maldives' northern atolls up to the Gujarat/Bengal coasts — every anchor
# and any route between them, with room for the router's padded search box.
BBOX = (64.0, 2.0, 100.0, 27.0)
SIMPLIFY_DEG = 0.005
COORD_DECIMALS = 4
OUT_PATH = Path(__file__).resolve().parent.parent / "agents" / "deterministic" / "data" / "land_india.geojson"


def _load(url: str) -> list:
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read())
    return [shape(f["geometry"]) for f in data["features"] if f.get("geometry")]


def _rounded(geom_json: dict) -> dict:
    def r(coords):
        if isinstance(coords[0], (int, float)):
            return [round(coords[0], COORD_DECIMALS), round(coords[1], COORD_DECIMALS)]
        return [r(c) for c in coords]

    return {"type": geom_json["type"], "coordinates": r(geom_json["coordinates"])}


def main() -> int:
    region = box(*BBOX)
    parts = []
    for url in SOURCES:
        print(f"Downloading {url} ...")
        for geom in _load(url):
            if geom.intersects(region):
                parts.append(geom.intersection(region))
    land = unary_union(parts).simplify(SIMPLIFY_DEG, preserve_topology=True)
    polygons = list(land.geoms) if land.geom_type == "MultiPolygon" else [land]
    features = [
        {"type": "Feature", "properties": {}, "geometry": _rounded(mapping(p))}
        for p in polygons
        if not p.is_empty
    ]
    out = {
        "type": "FeatureCollection",
        "metadata": {
            "source": "Natural Earth 1:10m land + minor islands (public domain)",
            "bbox": BBOX,
            "simplify_deg": SIMPLIFY_DEG,
        },
        "features": features,
    }
    OUT_PATH.write_text(json.dumps(out, separators=(",", ":")))
    vertices = sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors) for p in polygons)
    print(f"Wrote {len(features)} polygons, {vertices} vertices, {OUT_PATH.stat().st_size / 1024:.0f} KB -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
