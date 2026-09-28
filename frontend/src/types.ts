export interface TraceStep {
  agent_name: string;
  input_summary: string;
  output_summary: string;
  timestamp: string;
}

export interface PFZZone {
  zone_id: string;
  center: { lat: number; lon: number };
  // Backend only includes these when the live fetch actually returned a
  // value (visualization.py) — a partial fetch omits them entirely, so
  // they must be optional here too.
  distance_km?: number;
  sst_celsius?: number;
  chlorophyll_mg_m3?: number;
  advisory?: string;
}

export interface MapPayload {
  pins: ({
    lat: number;
    lon: number;
    label: string;
    type: string;
    zone_id?: string;
    center?: { lat: number; lon: number };
    distance_km?: number;
    sst_celsius?: number;
    chlorophyll_mg_m3?: number;
    advisory?: string;
  })[];
  overlays: ({
    type: string;
    name: string;
    geojson: { type: string; coordinates: unknown };
    highlighted?: boolean;
    zone_id?: string;
    center?: { lat: number; lon: number };
    distance_km?: number;
    sst_celsius?: number;
    chlorophyll_mg_m3?: number;
    advisory?: string;
  })[];
  route?: { lat: number; lon: number }[] | null;
}

export interface FinalResponse {
  answer_text: string;
  reasoning_trace: TraceStep[];
  map_payload: MapPayload;
  detected_language?: string | null;
  response_language?: string | null;
}

// Phase 2 — Zones Explorer (GET /zones)
export interface ZoneRecord {
  id: string;
  name: string;
  type: 'pfz' | 'restricted';
  near?: string;
  coordinates?: { lat: number; lon: number };
  distance_km?: number;
  sst_celsius?: number;
  chlorophyll_mg_m3?: number;
  advisory?: string;
  // type is GeoJSON's own discriminator ('Polygon' | 'MultiPolygon' seen in
  // practice — e.g. Gulf of Mannar is a MultiPolygon); coordinates' actual
  // nesting depth depends on it. Consumers must branch on `type` before
  // reading `coordinates` (see MapView.tsx's restricted-area rendering).
  geometry?: { type: string; coordinates: unknown };
}

export interface ZonesResponse {
  zones: ZoneRecord[];
  generated_at: string;
}

// Phase 3 — Weather Page (GET /weather?lat=&lon=)
export interface WeatherSnapshot {
  lat: number;
  lon: number;
  status: 'ok' | 'partial' | 'error';
  air_temperature_c: number | null;
  sst_celsius: number | null;
  wind_kmh: number | null;
  wave_height_m: number | null;
  chlorophyll_mg_m3: number | null;
  precipitation_mm: number | null;
  wind_max_kmh: number | null;
  sunrise_hour_ist: number | null;
  sunset_hour_ist: number | null;
  cyclone_alert: boolean;
  lightning_alert: boolean;
  source_timestamp: string;
}

// Weather Page — ML forecast (GET /weather/forecast?zone=)
export interface ForecastDay {
  horizon: number;
  date: string;
  wave_height_m: number;
  wind_kmh: number;
  wave_height_mae: number | null;
  wind_kmh_mae: number | null;
}

export interface ForecastResponse {
  zone: string;
  lat: number;
  lon: number;
  status: 'ok' | 'error';
  based_on_date?: string;
  model?: string;
  reason?: string;
  forecast: ForecastDay[];
  generated_at: string;
}

// Phase 4 — Route Planner (POST /route)
export interface RouteResponse {
  route: { lat: number; lon: number }[] | null;
  distance_km: number | null;
  waypoint_count: number;
  reason: string | null;
}

// Phase 5 — Alerts/Advisories (GET /alerts)
export interface AlertRecord {
  zone_id: string;
  zone_name: string;
  near?: string;
  alert_type: 'cyclone' | 'lightning';
  severity: 'high' | 'moderate';
  detail: string;
}

export interface AlertsResponse {
  alerts: AlertRecord[];
  checked_zones: number;
  generated_at: string;
}

// Phase 6 — Analytics Dashboard (GET /analytics/historical)
export interface AnalyticsSeries {
  unit: string;
  dates: string[];
  values: (number | null)[];
  status: 'ok' | 'partial' | 'error';
  /** Only present for moon_phase — display name per date, parallel to values. */
  names?: string[];
}

export interface AnalyticsResponse {
  lat: number;
  lon: number;
  start_date: string;
  end_date: string;
  series: Record<string, AnalyticsSeries>;
}

// Phase 7 — History (GET /history, GET /history/{id})
export type PageSource = 'chat' | 'zones' | 'weather' | 'route' | 'alerts' | 'analytics' | 'download';

export interface HistoryRecord {
  id: number;
  timestamp: string;
  page_source: PageSource;
  query_summary: string;
  session_id: string;
}

export interface HistoryRecordDetail extends HistoryRecord {
  // For page_source: 'download', `response` is NOT the exported file body —
  // HistoryLoggingMiddleware skips non-JSON responses and logs only
  // { content_type: string; size_bytes: number } instead. For every other
  // page_source it's the actual JSON response payload.
  payload: { request: unknown; response: unknown } | null;
}

export interface HistoryListResponse {
  items: HistoryRecord[];
  total: number;
  limit: number;
  offset: number;
}
