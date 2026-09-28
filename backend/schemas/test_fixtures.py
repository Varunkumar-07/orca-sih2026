"""
ORCA shared test fixtures — backend/schemas/test_fixtures.py

Sample EvidenceBundle instances for integration testing between Team Claude
(reasoning agents) and Team Gemini (deterministic modules + frontend).

For Phase 1, only FIXTURE_1_HAPPY_PATH's `marine` field is exercised by the
Marine Data Discovery Agent. The other fixtures, and the weather/risk fields,
are for later phases and integration testing.
"""

from backend.schemas.contracts import (
    EvidenceBundle,
    GeoPoint,
    MarineDataResult,
    RiskAssessment,
    WeatherDataResult,
)

FIXTURE_1_HAPPY_PATH = EvidenceBundle(
    query_text="Is it safe to go out tomorrow near Chennai?",
    query_location=GeoPoint(lat=13.08, lon=80.27),
    marine=MarineDataResult(
        status="ok",
        pfz_zones=[{"zone_id": "PFZ-CH-14", "center": {"lat": 13.05, "lon": 80.35}}],
        sst_celsius=28.4, chlorophyll_mg_m3=0.62,
        source_timestamp="2026-08-27T06:00:00Z",
    ),
    weather=WeatherDataResult(
        status="ok", wind_kmh=18.5, wave_height_m=1.2,
        cyclone_alert=False, lightning_alert=False,
        tide_info={"high_tide": "14:32", "low_tide": "08:10"},
        source_timestamp="2026-08-27T06:00:00Z",
    ),
    risk=RiskAssessment(status="ok", safe_to_go=True, confidence=0.88,
                        explanation="Calm seas, no active alerts."),
)

FIXTURE_2_HAZARD_PATH = EvidenceBundle(
    query_text="Is it safe to go out tomorrow near Chennai?",
    query_location=GeoPoint(lat=13.08, lon=80.27),
    marine=FIXTURE_1_HAPPY_PATH.marine,
    weather=WeatherDataResult(
        status="ok", wind_kmh=62.0, wave_height_m=4.5,
        cyclone_alert=True, lightning_alert=False,
        tide_info={"high_tide": "14:32", "low_tide": "08:10"},
        source_timestamp="2026-08-27T06:00:00Z",
    ),
    risk=RiskAssessment(status="ok", safe_to_go=False, confidence=0.95,
                        explanation="Active cyclone alert, dangerous wave height."),
)

FIXTURE_3_PARTIAL_FAILURE = EvidenceBundle(
    query_text="Is it safe to go out tomorrow near Chennai?",
    query_location=GeoPoint(lat=13.08, lon=80.27),
    marine=FIXTURE_1_HAPPY_PATH.marine,
    weather=WeatherDataResult(
        status="error", wind_kmh=None, wave_height_m=None,
        cyclone_alert=False, lightning_alert=False, tide_info=None,
        source_timestamp="2026-08-27T06:00:00Z",
        error_message="IMD feed unreachable",
    ),
    risk=RiskAssessment(status="partial", safe_to_go=None, confidence=0.4,
                        explanation="Weather data unavailable; assessment incomplete."),
)

FIXTURE_4_RESTRICTED_ZONE = EvidenceBundle(
    query_text="Can I fish near Gulf of Mannar?",
    query_location=GeoPoint(lat=9.05, lon=79.15),
    marine=FIXTURE_1_HAPPY_PATH.marine,
    weather=FIXTURE_1_HAPPY_PATH.weather,
    risk=FIXTURE_1_HAPPY_PATH.risk,
)


ALL_FIXTURES = [
    FIXTURE_1_HAPPY_PATH,
    FIXTURE_2_HAZARD_PATH,
    FIXTURE_3_PARTIAL_FAILURE,
    FIXTURE_4_RESTRICTED_ZONE,
]
