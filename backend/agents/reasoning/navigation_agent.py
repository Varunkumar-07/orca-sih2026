"""
Navigation Agent — Team Claude, Tier B (deterministic, no LLM call).

Plans a sea route between a start point and a chosen destination PFZ using
A* search over a grid of candidate waypoints. Impassable: land (Natural
Earth coastline, geospatial.get_land_mask) and protected areas (Geospatial
Reasoning's WDPA extents, geospatial.get_active_restricted_areas) plus a
small routing clearance around them.

This solves a genuinely different problem than road navigation: a road
router snaps a route onto a pre-existing network; open water has no such
network — a vessel can travel in any direction, constrained only by what it
must avoid. A* over a uniform obstacle grid is the direct fit for that:
guaranteed shortest (on the grid) for an admissible heuristic, and simple
enough to verify.

How the grid stays honest:
- A cell is blocked if it lies within half a cell diagonal of land. Any
  straight step between two open cells then provably never touches land
  (every point of a step is within that distance of one of its end cells),
  so a coarse grid can't "jump" a thin spit or island.
- Protected areas get max(ROUTE_CLEARANCE_KM, half a cell diagonal). That
  clearance is a routing margin only — whether a point is *inside* a
  protected area (and so refused) uses the area's actual extent.
- Start/destination points on land (every city is) or inside a clearance
  margin are snapped to the nearest open-water cell within MAX_SNAP_KM —
  never across a protected area — and the snap distance is reported. A
  point inside a protected area's actual extent is refused outright.
- If the first (tight) search box has no sea route, one wider box is
  searched (e.g. around Sri Lanka rather than through the Pamban channel,
  which is narrower than the grid can resolve). Otherwise the result says
  plainly that no sea route was found — never a line across land.

Never raises across the agent boundary — every failure comes back as a
RouteResult with route=None and a user-facing reason.
"""
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from backend.agents.deterministic.geospatial import get_land_mask, haversine_km

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid resolution (Phase 5.2a)
# ---------------------------------------------------------------------------
# Cell spacing scales with the start->destination distance: target roughly
# _TARGET_CELLS_ACROSS cells along the straight line, clamped to a sane
# per-cell range for marine distances.
_TARGET_CELLS_ACROSS = 50
_CELL_KM_MIN = 1.0
_CELL_KM_MAX = 2.0

# Hard cap on grid dimensions (per axis) so a very long route still
# terminates quickly — cells are coarsened rather than the area truncated.
_MAX_GRID_DIM = 120

# Padding around the start/destination bounding box, as a fraction of the
# straight-line distance, so even a due-north/east route has room to detour.
_BBOX_PADDING_FRACTION = 0.30
_BBOX_PADDING_MIN_CELLS = 3
# The one wider retry when the tight box has no sea route.
_WIDE_PADDING_FRACTION = 1.0
_WIDE_PADDING_MIN_KM = 150.0

# Flat degrees-per-km approximation (1 deg latitude ~= 111km; longitude
# scaled by cos(lat)) — fine across India's 8-22N coastal range.
_KM_PER_DEG_LAT = 111.0

# Routing margin kept from a protected area's actual extent. Deliberately
# separate from (and much smaller than) geospatial.MPA_PROXIMITY_KM, which
# only drives a proximity warning in the chat answer.
ROUTE_CLEARANCE_KM = 2.0
# How far a start/destination on land (or in a clearance margin) may be
# moved to reach open water.
MAX_SNAP_KM = 20.0
_SNAP_CANDIDATES = 200

# 8-directional grid movement (rook + diagonal); a diagonal step's true
# haversine length is its cost, so admissibility doesn't depend on it.
_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

Cell = tuple[int, int]

# Sentinel: "use the real land mask" (the default) vs an explicit geometry
# or None (no land at all — obstacle-logic unit tests).
_DEFAULT_LAND = object()


@dataclass
class RouteResult:
    route: list[dict] | None
    reason: str | None = None
    # How far the requested start/destination were moved to reach open
    # water (0 when they were already at sea).
    start_offset_km: float = 0.0
    end_offset_km: float = 0.0


def _km_per_deg_lon(lat: float) -> float:
    return _KM_PER_DEG_LAT * math.cos(math.radians(lat))


class _Grid:
    """Lat/lon <-> grid-cell conversion and obstacle lookup for one search.
    Built fresh per search, never shared/cached."""

    def __init__(self, start: dict, destination: dict, polygons: list, land, padding_fraction: float, min_pad_km: float):
        lat0, lon0 = float(start["lat"]), float(start["lon"])
        lat1, lon1 = float(destination["lat"]), float(destination["lon"])

        straight_km = haversine_km(lat0, lon0, lat1, lon1)
        cell_km = (
            min(_CELL_KM_MAX, max(_CELL_KM_MIN, straight_km / _TARGET_CELLS_ACROSS))
            if straight_km > 0
            else _CELL_KM_MIN
        )

        min_lat, max_lat = min(lat0, lat1), max(lat0, lat1)
        min_lon, max_lon = min(lon0, lon1), max(lon0, lon1)
        mid_lat = (min_lat + max_lat) / 2.0
        km_per_deg_lon = _km_per_deg_lon(mid_lat) or _KM_PER_DEG_LAT

        pad_km = max(straight_km * padding_fraction, cell_km * _BBOX_PADDING_MIN_CELLS, min_pad_km)
        self.min_lat = min_lat - pad_km / _KM_PER_DEG_LAT
        self.max_lat = max_lat + pad_km / _KM_PER_DEG_LAT
        self.min_lon = min_lon - pad_km / km_per_deg_lon
        self.max_lon = max_lon + pad_km / km_per_deg_lon
        self.pad_km = pad_km

        self.lat_step = cell_km / _KM_PER_DEG_LAT
        self.lon_step = cell_km / km_per_deg_lon

        n_rows = max(2, int((self.max_lat - self.min_lat) / self.lat_step) + 1)
        n_cols = max(2, int((self.max_lon - self.min_lon) / self.lon_step) + 1)
        if n_rows > _MAX_GRID_DIM:
            self.lat_step *= n_rows / _MAX_GRID_DIM
            n_rows = _MAX_GRID_DIM
        if n_cols > _MAX_GRID_DIM:
            self.lon_step *= n_cols / _MAX_GRID_DIM
            n_cols = _MAX_GRID_DIM
        self.n_rows = n_rows
        self.n_cols = n_cols

        # Half a cell diagonal, in km and (conservatively, using the
        # widest-latitude longitude scale in the box) in degrees.
        half_diag_km = 0.5 * math.hypot(self.lat_step * _KM_PER_DEG_LAT, self.lon_step * km_per_deg_lon)
        widest = max(abs(self.min_lat), abs(self.max_lat))
        deg_per_km = 1.0 / (_km_per_deg_lon(widest) or _KM_PER_DEG_LAT)

        self.lats = self.min_lat + np.arange(n_rows) * self.lat_step
        self.lons = self.min_lon + np.arange(n_cols) * self.lon_step
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        points = shapely.points(lon_grid.ravel(), lat_grid.ravel())

        blocked = np.zeros(points.shape, dtype=bool)
        if land is not None:
            box = (self.min_lon - 1, self.min_lat - 1, self.max_lon + 1, self.max_lat + 1)
            nearby_land = shapely.clip_by_rect(land, *box)
            if not nearby_land.is_empty:
                shapely.prepare(nearby_land)
                blocked |= shapely.dwithin(nearby_land, points, half_diag_km * deg_per_km)
        margin_deg = max(ROUTE_CLEARANCE_KM, half_diag_km) * deg_per_km
        for poly in polygons:
            blocked |= shapely.dwithin(poly, points, margin_deg)
        self._obstacle = blocked.reshape(n_rows, n_cols)

    def to_cell(self, lat: float, lon: float) -> Cell:
        r = round((lat - self.min_lat) / self.lat_step)
        c = round((lon - self.min_lon) / self.lon_step)
        return min(max(r, 0), self.n_rows - 1), min(max(c, 0), self.n_cols - 1)

    def to_latlon(self, cell: Cell) -> dict:
        r, c = cell
        return {"lat": float(self.lats[r]), "lon": float(self.lons[c])}

    def is_obstacle(self, cell: Cell) -> bool:
        return bool(self._obstacle[cell])

    def neighbors(self, cell: Cell):
        r, c = cell
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
                continue
            # No cutting a corner past an obstacle on a diagonal step.
            if dr != 0 and dc != 0 and (self.is_obstacle((r, nc)) or self.is_obstacle((nr, c))):
                continue
            yield (nr, nc)

    def step_cost_km(self, a: Cell, b: Cell) -> float:
        pa, pb = self.to_latlon(a), self.to_latlon(b)
        return haversine_km(pa["lat"], pa["lon"], pb["lat"], pb["lon"])

    def heuristic_km(self, cell: Cell, goal: Cell) -> float:
        p, g = self.to_latlon(cell), self.to_latlon(goal)
        return haversine_km(p["lat"], p["lon"], g["lat"], g["lon"])


def _snap_to_water(grid: _Grid, point: dict, polygons: list) -> tuple[Cell | None, float]:
    """(cell, km moved): the point's own cell if it's open water, else the
    nearest open cell within MAX_SNAP_KM whose straight connection to the
    point doesn't cross a protected area (so a destination enclosed by one
    stays unreachable instead of being "snapped" out of it). (None, 0) if
    there's no such cell."""
    lat, lon = float(point["lat"]), float(point["lon"])
    own = grid.to_cell(lat, lon)
    if not grid.is_obstacle(own):
        return own, 0.0
    free_r, free_c = np.nonzero(~grid._obstacle)
    if free_r.size == 0:
        return None, 0.0
    dlat_km = (grid.lats[free_r] - lat) * _KM_PER_DEG_LAT
    dlon_km = (grid.lons[free_c] - lon) * _km_per_deg_lon(lat)
    dist_km = np.hypot(dlat_km, dlon_km)
    for i in np.argsort(dist_km)[:_SNAP_CANDIDATES]:
        if dist_km[i] > MAX_SNAP_KM:
            break
        cell = (int(free_r[i]), int(free_c[i]))
        target = grid.to_latlon(cell)
        if not _segment_crosses(point, target, polygons, None):
            return cell, haversine_km(lat, lon, target["lat"], target["lon"])
    return None, 0.0


def _astar(grid: _Grid, start: Cell, goal: Cell) -> list[Cell] | None:
    """Standard A* over the grid's open/closed sets. Haversine distance is
    both the step cost and the heuristic — admissible, since any grid path
    between two cells is at least their great-circle distance."""
    if start == goal:
        return [start]

    open_heap: list[tuple[float, Cell]] = [(grid.heuristic_km(start, goal), start)]
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    closed: set[Cell] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct_path(came_from, current)
        closed.add(current)

        for neighbor in grid.neighbors(current):
            if grid.is_obstacle(neighbor) or neighbor in closed:
                continue
            tentative_g = g_score[current] + grid.step_cost_km(current, neighbor)
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_heap, (tentative_g + grid.heuristic_km(neighbor, goal), neighbor))
    return None


def _reconstruct_path(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _collinear_corner_indices(cell_path: list[Cell]) -> list[int]:
    """Index positions in `cell_path` worth keeping: the start, the end, and
    every point where the path's discrete step direction changes.

    A run of steps sharing the same (dr, dc) offset is exactly collinear —
    every intermediate cell in that run sits precisely on the straight line
    between the run's endpoints (constant per-step delta in both grid
    axes) — so dropping them loses no geometric information at all. This is
    why explicit direction-change detection is used here instead of a
    tolerance-based method like Ramer-Douglas-Peucker: RDP trades a small
    approximation error for a shorter list, which would need re-justifying
    against the restricted zones; collinearity-based collapsing over a
    fixed-connectivity grid path introduces no approximation to justify in
    the first place — the simplified polyline is the exact same continuous
    line as the original, just without redundant collinear points.
    """
    if len(cell_path) <= 2:
        return list(range(len(cell_path)))

    indices = [0]
    prev_dir: Cell | None = None
    for i in range(1, len(cell_path)):
        direction = (cell_path[i][0] - cell_path[i - 1][0], cell_path[i][1] - cell_path[i - 1][1])
        if prev_dir is not None and direction != prev_dir:
            indices.append(i - 1)
        prev_dir = direction
    indices.append(len(cell_path) - 1)
    return indices


def _segment_crosses(a: dict, b: dict, polygons: list, land) -> bool:
    """True if the straight line from `a` to `b` enters any protected-area
    polygon or (when given) land."""
    if a == b:
        return False
    line = LineString([(a["lon"], a["lat"]), (b["lon"], b["lat"])])
    for geom in [*polygons, *([land] if land is not None else [])]:
        try:
            if geom.intersects(line):
                return True
        except Exception:
            continue
    return False


def _simplify_path(cell_path: list[Cell], grid: _Grid, polygons: list, land) -> list[dict]:
    """Collapse straight-line runs down to their endpoints (Phase 5.2c),
    verifying each collapsed segment against protected areas and land
    before accepting it. The grid's obstacle margins mean this check should
    never fail — kept as an explicit safety net: a run that does fail keeps
    its original, uncollapsed waypoints instead of being trusted."""
    corner_indices = _collinear_corner_indices(cell_path)
    latlon_path = [grid.to_latlon(cell) for cell in cell_path]

    simplified: list[dict] = [latlon_path[corner_indices[0]]]
    for k in range(1, len(corner_indices)):
        i, j = corner_indices[k - 1], corner_indices[k]
        a, b = latlon_path[i], latlon_path[j]
        if _segment_crosses(a, b, polygons, land):
            simplified.extend(latlon_path[i + 1 : j + 1])
        else:
            simplified.append(b)
    return simplified


def _protected_area_at(point: dict, restricted_zones: list[dict]) -> str | None:
    pt = Point(float(point["lon"]), float(point["lat"]))
    for zone in restricted_zones:
        poly = zone.get("polygon")
        try:
            if poly is not None and poly.covers(pt):
                return zone.get("name") or "a protected area"
        except Exception:
            continue
    return None


def plan_route(start: dict, destination: dict, restricted_zones: list[dict], land=_DEFAULT_LAND) -> RouteResult:
    """Plan a sea route from `start` to `destination` ({"lat", "lon"} each).

    `restricted_zones` has geospatial.get_active_restricted_areas()'s shape
    ([{"name", "polygon"}, ...], each polygon the area's actual extent).
    `land` defaults to the real land mask; pass None to plan with no land
    (obstacle-logic unit tests only).

    Returns a RouteResult: `route` is the ordered waypoint list (simplified
    to the start, the destination and every direction change), or None with
    a user-facing `reason`. Never raises.
    """
    try:
        if land is _DEFAULT_LAND:
            land = get_land_mask()
            if land is None:
                return RouteResult(None, "coastline data is unavailable, so a route that stays at sea can't be guaranteed")
        polygons = [z["polygon"] for z in restricted_zones if z.get("polygon") is not None]

        for label, point in (("start", start), ("destination", destination)):
            area = _protected_area_at(point, restricted_zones)
            if area is not None:
                logger.info("navigation_agent: %s inside protected area %s — no route", label, area)
                return RouteResult(None, f"the {label} is inside {area} (protected area) — ORCA does not plan routes into or out of protected areas")

        searched_km = 0.0
        for padding_fraction, min_pad_km in ((_BBOX_PADDING_FRACTION, 0.0), (_WIDE_PADDING_FRACTION, _WIDE_PADDING_MIN_KM)):
            grid = _Grid(start, destination, polygons, land, padding_fraction, min_pad_km)
            searched_km = grid.pad_km
            start_cell, start_offset = _snap_to_water(grid, start, polygons)
            if start_cell is None:
                return RouteResult(None, f"the start is more than {MAX_SNAP_KM:.0f} km from open water that can be reached without crossing a protected area")
            goal_cell, end_offset = _snap_to_water(grid, destination, polygons)
            if goal_cell is None:
                return RouteResult(None, f"the destination is more than {MAX_SNAP_KM:.0f} km from open water that can be reached without crossing a protected area")
            cell_path = _astar(grid, start_cell, goal_cell)
            if cell_path is not None:
                return RouteResult(
                    _simplify_path(cell_path, grid, polygons, land),
                    None,
                    round(start_offset, 2),
                    round(end_offset, 2),
                )

        logger.info("navigation_agent: no sea route within %.0f km of the start/destination", searched_km)
        return RouteResult(
            None,
            f"no sea route found that avoids land and protected areas (searched up to {searched_km:.0f} km around the two points)",
        )
    except Exception as exc:  # noqa: BLE001 - must never raise across the boundary
        logger.warning("navigation_agent: route planning failed: %s", exc)
        return RouteResult(None, "route planning failed unexpectedly")


def find_route(start: dict, destination: dict, restricted_zones: list[dict], land=_DEFAULT_LAND) -> list[dict] | None:
    """plan_route's waypoints only (None when there's no route) — for
    callers that don't need the reason or snap offsets."""
    return plan_route(start, destination, restricted_zones, land).route
