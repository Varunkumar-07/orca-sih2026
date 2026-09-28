"""
Navigation Agent — Team Claude, Tier B (deterministic, no LLM call).

Plans a hazard-avoiding sea route between a start point and a chosen
destination PFZ using A* search over a grid of candidate waypoints, treating
any cell that falls inside a restricted zone (Geospatial Reasoning's real
WDPA polygon data) as impassable.

This solves a genuinely different problem than road navigation: a road
router snaps a route onto a pre-existing network ("which sequence of roads
gets me there fastest"); open water has no such network — a vessel can, in
principle, travel in any direction, constrained only by what it must avoid.
A* over a uniform obstacle grid is the direct fit for that: guaranteed
shortest for an admissible heuristic, and simple enough to verify.

Reuses geospatial.py's haversine_km (not reimplemented) for both grid-cell
distances and the A* heuristic. Point-in-polygon containment is checked
locally against the `restricted_zones` list this module receives as a
parameter (same shape as geospatial.get_active_restricted_areas()'s return —
[{"name": str, "polygon": shapely Polygon/MultiPolygon}, ...]) rather than by
importing geospatial's private module-level cache, so route planning stays
independently testable against an arbitrary zone list and decoupled from
geospatial's own caching lifecycle.

Never raises across the agent boundary — a route that cannot be found (start
or destination inside a restricted zone, or destination fully enclosed by
one) degrades to a clear None return, logged with the specific reason, never
an exception.
"""
from __future__ import annotations

import heapq
import logging
import math

from shapely.geometry import LineString, Point, Polygon

from backend.agents.deterministic.geospatial import haversine_km

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid resolution (Phase 5.2a)
# ---------------------------------------------------------------------------
# Cell spacing scales with the start->destination distance rather than using
# a fixed size: a fixed ~1km grid over a 300km route would need ~300x300
# cells; a fixed ~2km grid over a 3km route would produce only a 1-2 cell
# wide grid, too coarse to route around anything. Instead we target a
# roughly constant number of cells along the straight-line start->destination
# distance, clamped to a sane per-cell range for marine distances.
_TARGET_CELLS_ACROSS = 50
_CELL_KM_MIN = 1.0
_CELL_KM_MAX = 2.0

# Hard cap on grid dimensions (per axis) so a very long route still
# terminates quickly instead of building an unbounded grid — step size is
# rescaled up (coarser cells) rather than truncating the covered area.
_MAX_GRID_DIM = 120

# Padding around the start/destination bounding box, as a fraction of the
# straight-line start->destination distance — sized off total route length
# (not the raw lat/lon span) specifically so a due-north or due-east route
# (zero span on the other axis) still gets generous room on all sides to
# detour around an obstacle, not just a sliver matching the direct line.
_BBOX_PADDING_FRACTION = 0.30
_BBOX_PADDING_MIN_CELLS = 3

# Flat degrees-per-km approximation, consistent with geospatial.py's own
# MPA_BUFFER_DEG convention: 1 deg latitude ~= 111km, longitude within ~2% of
# that across India's 8-22N coastal range. Not valid at high latitudes.
_KM_PER_DEG_LAT = 111.0

# 8-directional grid movement (rook + diagonal moves). Chosen over
# 4-directional because it produces materially straighter, more realistic
# sea routes — a 4-connected grid forces a staircase path around any
# obstacle edge that isn't perfectly axis-aligned. Chosen over a finer
# hex/16-direction scheme because 8 directions is the simplest connectivity
# that still keeps the cost function exact: a diagonal step's true haversine
# distance is used directly (not approximated as sqrt(2) * straight-step
# cost), so admissibility doesn't depend on the connectivity pattern at all.
_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

Cell = tuple[int, int]


def _km_per_deg_lon(lat: float) -> float:
    return _KM_PER_DEG_LAT * math.cos(math.radians(lat))


def _point_in_any(lat: float, lon: float, polygons: list[Polygon]) -> bool:
    """Point-in-polygon containment check against an explicit polygon list.

    Same predicate as geospatial.py's _check_restricted (covers — inclusive
    of the boundary — falling back to intersects on a degenerate geometry),
    reimplemented here in terms of the `restricted_zones` list find_route()
    receives directly rather than by importing that private, module-cached
    helper — keeping this agent testable against an arbitrary zone list
    without depending on geospatial's own WDPA-cache lifecycle.
    """
    pt = Point(lon, lat)  # Shapely is (x=lon, y=lat)
    for poly in polygons:
        try:
            if poly.covers(pt):
                return True
        except Exception:
            try:
                if poly.intersects(pt):
                    return True
            except Exception:
                continue
    return False


class _Grid:
    """Lat/lon <-> grid-cell conversion and obstacle lookup for one
    find_route() call. Built fresh per call, never shared/cached."""

    def __init__(self, start: dict, destination: dict, restricted_zones: list[dict]):
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
        km_per_deg_lon = _km_per_deg_lon(mid_lat) or _KM_PER_DEG_LAT  # guard cos()==0, never hit for marine India routes

        pad_km = max(straight_km * _BBOX_PADDING_FRACTION, cell_km * _BBOX_PADDING_MIN_CELLS)
        pad_lat = pad_km / _KM_PER_DEG_LAT
        pad_lon = pad_km / km_per_deg_lon

        self.min_lat = min_lat - pad_lat
        self.max_lat = max_lat + pad_lat
        self.min_lon = min_lon - pad_lon
        self.max_lon = max_lon + pad_lon

        self.lat_step = cell_km / _KM_PER_DEG_LAT
        self.lon_step = cell_km / km_per_deg_lon
        self.cell_km = cell_km

        n_rows = max(2, int((self.max_lat - self.min_lat) / self.lat_step) + 1)
        n_cols = max(2, int((self.max_lon - self.min_lon) / self.lon_step) + 1)

        # Rescale (coarsen) rather than truncate if the padded bbox would
        # need more cells per axis than the cap — keeps the covered area
        # intact for a very long route, just at lower resolution.
        if n_rows > _MAX_GRID_DIM:
            self.lat_step *= n_rows / _MAX_GRID_DIM
            n_rows = _MAX_GRID_DIM
        if n_cols > _MAX_GRID_DIM:
            self.lon_step *= n_cols / _MAX_GRID_DIM
            n_cols = _MAX_GRID_DIM

        self.n_rows = n_rows
        self.n_cols = n_cols

        polygons: list[Polygon] = [
            z["polygon"] for z in restricted_zones if z.get("polygon") is not None
        ]
        self._obstacle = [
            [
                _point_in_any(self.min_lat + r * self.lat_step, self.min_lon + c * self.lon_step, polygons)
                for c in range(n_cols)
            ]
            for r in range(n_rows)
        ]

        self.start_cell = self.to_cell(lat0, lon0)
        self.goal_cell = self.to_cell(lat1, lon1)

    def to_cell(self, lat: float, lon: float) -> Cell:
        r = round((lat - self.min_lat) / self.lat_step)
        c = round((lon - self.min_lon) / self.lon_step)
        r = min(max(r, 0), self.n_rows - 1)
        c = min(max(c, 0), self.n_cols - 1)
        return r, c

    def to_latlon(self, cell: Cell) -> dict:
        r, c = cell
        return {"lat": self.min_lat + r * self.lat_step, "lon": self.min_lon + c * self.lon_step}

    def is_obstacle(self, cell: Cell) -> bool:
        r, c = cell
        return self._obstacle[r][c]

    def neighbors(self, cell: Cell):
        r, c = cell
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
                continue
            # Diagonal step: disallow cutting the corner when either
            # orthogonal flanking cell is an obstacle — otherwise the
            # path can clip straight through a restricted-zone corner
            # without ever landing on a cell flagged as an obstacle.
            if dr != 0 and dc != 0 and (self.is_obstacle((r, nc)) or self.is_obstacle((nr, c))):
                continue
            yield (nr, nc)

    def step_cost_km(self, a: Cell, b: Cell) -> float:
        pa, pb = self.to_latlon(a), self.to_latlon(b)
        return haversine_km(pa["lat"], pa["lon"], pb["lat"], pb["lon"])

    def heuristic_km(self, cell: Cell) -> float:
        p, g = self.to_latlon(cell), self.to_latlon(self.goal_cell)
        return haversine_km(p["lat"], p["lon"], g["lat"], g["lon"])


def _reconstruct_path(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _astar(grid: _Grid) -> list[Cell] | None:
    """Standard A* over the grid's open/closed sets.

    haversine distance is used as both the per-step edge cost and the
    heuristic — admissible (never overestimates) because haversine is the
    true great-circle straight-line distance between two cells, and any
    actual grid path between them is at least that long.
    """
    start, goal = grid.start_cell, grid.goal_cell

    if grid.is_obstacle(start):
        logger.warning("navigation_agent: start point falls inside a restricted zone — no route")
        return None
    if grid.is_obstacle(goal):
        logger.warning("navigation_agent: destination falls inside a restricted zone — no route")
        return None
    if start == goal:
        return [start]

    open_heap: list[tuple[float, Cell]] = [(grid.heuristic_km(start), start)]
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
                heapq.heappush(open_heap, (tentative_g + grid.heuristic_km(neighbor), neighbor))

    logger.warning(
        "navigation_agent: no path found — destination is unreachable given the restricted-zone obstacles"
    )
    return None


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


def _segment_crosses_restricted(a: dict, b: dict, polygons: list[Polygon]) -> bool:
    """True if the straight line from `a` to `b` enters any restricted
    polygon. Used to verify a simplified segment before accepting it — a
    collapsed run can span many original grid steps, so its endpoints alone
    aren't checked; the whole line between them is."""
    if a == b:
        return False
    line = LineString([(a["lon"], a["lat"]), (b["lon"], b["lat"])])
    for poly in polygons:
        try:
            if poly.intersects(line):
                return True
        except Exception:
            continue
    return False


def _simplify_path(cell_path: list[Cell], grid: _Grid, polygons: list[Polygon]) -> list[dict]:
    """Collapse straight-line runs down to their endpoints (Phase 5.2c),
    verifying each collapsed segment against `polygons` before accepting it.

    The per-run collapse is exact (see _collinear_corner_indices), so this
    check should never actually fail in practice — kept anyway as an
    explicit safety net rather than a silent assumption: any run whose
    segment does fail the check falls back to its original, uncollapsed
    waypoints instead of being dropped or accepted unverified.
    """
    corner_indices = _collinear_corner_indices(cell_path)
    latlon_path = [grid.to_latlon(cell) for cell in cell_path]

    simplified: list[dict] = [latlon_path[corner_indices[0]]]
    for k in range(1, len(corner_indices)):
        i, j = corner_indices[k - 1], corner_indices[k]
        a, b = latlon_path[i], latlon_path[j]
        if _segment_crosses_restricted(a, b, polygons):
            # Defensive fallback (see docstring) — keep every original
            # intermediate point for this run rather than trust the shortcut.
            simplified.extend(latlon_path[i + 1 : j + 1])
        else:
            simplified.append(b)
    return simplified


def find_route(
    start: dict,
    destination: dict,
    restricted_zones: list[dict],
) -> list[dict] | None:
    """Plan a hazard-avoiding sea route from `start` to `destination`.

    Args:
        start: {"lat": float, "lon": float} — the user's current position.
        destination: {"lat": float, "lon": float} — the chosen PFZ
            candidate's center (from marine_data_agent's multi-zone output).
        restricted_zones: same shape as
            geospatial.get_active_restricted_areas()'s return —
            [{"name": str, "polygon": shapely Polygon/MultiPolygon}, ...].
            Passed in explicitly rather than this module reaching into
            geospatial's own cache, so route planning stays independently
            testable against an arbitrary zone list.

    Returns:
        Ordered list of {"lat": float, "lon": float} waypoints from start to
        destination inclusive — simplified (Phase 5.2c) to the start, the
        destination, and only the points where the path's direction actually
        changes; straight-line runs between direction changes collapse to
        just their endpoints. None if no route exists — start or destination
        inside a restricted zone, or destination fully enclosed by one.
        Never raises; the specific reason is logged via this module's
        logger rather than surfaced through the return value, matching the
        rest of the agent pipeline's "degrade, don't throw" convention.
    """
    try:
        grid = _Grid(start, destination, restricted_zones)
        cell_path = _astar(grid)
        if cell_path is None:
            return None
        polygons: list[Polygon] = [
            z["polygon"] for z in restricted_zones if z.get("polygon") is not None
        ]
        return _simplify_path(cell_path, grid, polygons)
    except Exception as exc:  # noqa: BLE001 - must never raise across the boundary
        logger.warning("navigation_agent: route planning failed: %s", exc)
        return None
