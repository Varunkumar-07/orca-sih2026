"""
Sample PFZ zones for Assistant Demo mode — backend/schemas/demo_places.py

Demo mode answers from hand-authored scenario fixtures (test_fixtures.py),
which are all set at Chennai. When a demo question names a different place,
routers/chat.py moves the scenario to that place — and these are the fishing
zones it then shows: real front-detected PFZ zones captured from the live
GET /zones catalog on 2026-09-29 (satellite pass 2026-09-28), up to 3 per
anchor city (the same anchors pfz_service.py scans), keyed by anchor name as
it appears in planning_agent._KNOWN_LOCATIONS. Sample data for an offline
demo — never presented by live mode.

Like live detection, no zone lies inside a protected area's extent: the
Gulf of Mannar set was re-derived from the same pass with those cells
masked out (the old PFZ-002 sat inside the Gulf of Mannar reserve), so its
IDs run 001-003 in rank order. tests/test_zone_protected_areas.py guards this.
"""

DEMO_SAMPLE_PFZ_BY_ANCHOR: dict[str, list[dict]] = {
    "kandla": [
        {"zone_id": "KANDLA-PFZ-001", "center": {"lat": 22.7292, "lon": 70.1875}},
        {"zone_id": "KANDLA-PFZ-002", "center": {"lat": 22.7708, "lon": 69.8542}},
        {"zone_id": "KANDLA-PFZ-003", "center": {"lat": 22.4792, "lon": 69.4792}},
    ],
    "mumbai": [
        {"zone_id": "MUMBAI-PFZ-001", "center": {"lat": 18.9792, "lon": 72.7292}},
        {"zone_id": "MUMBAI-PFZ-002", "center": {"lat": 18.5208, "lon": 72.8125}},
        {"zone_id": "MUMBAI-PFZ-003", "center": {"lat": 18.8125, "lon": 72.6875}},
    ],
    "goa": [
        {"zone_id": "GOA-PFZ-001", "center": {"lat": 14.9792, "lon": 73.4375}},
        {"zone_id": "GOA-PFZ-002", "center": {"lat": 14.8542, "lon": 73.8958}},
        {"zone_id": "GOA-PFZ-003", "center": {"lat": 15.0625, "lon": 73.5625}},
    ],
    "mangaluru": [
        {"zone_id": "MANGALURU-PFZ-001", "center": {"lat": 12.2292, "lon": 74.5625}},
        {"zone_id": "MANGALURU-PFZ-002", "center": {"lat": 13.3542, "lon": 74.2708}},
        {"zone_id": "MANGALURU-PFZ-003", "center": {"lat": 13.2292, "lon": 74.5625}},
    ],
    "kochi": [
        {"zone_id": "KOCHI-PFZ-001", "center": {"lat": 9.5208, "lon": 76.1875}},
        {"zone_id": "KOCHI-PFZ-002", "center": {"lat": 9.3542, "lon": 76.1875}},
        {"zone_id": "KOCHI-PFZ-003", "center": {"lat": 10.5208, "lon": 75.8958}},
    ],
    "chennai": [
        {"zone_id": "CHENNAI-PFZ-001", "center": {"lat": 13.0625, "lon": 80.3542}},
        {"zone_id": "CHENNAI-PFZ-002", "center": {"lat": 12.4375, "lon": 80.2292}},
        {"zone_id": "CHENNAI-PFZ-003", "center": {"lat": 13.1875, "lon": 80.4375}},
    ],
    "gulf of mannar": [
        {"zone_id": "GULF-OF-MANNAR-PFZ-001", "center": {"lat": 8.9375, "lon": 79.3542}},
        {"zone_id": "GULF-OF-MANNAR-PFZ-002", "center": {"lat": 8.8125, "lon": 79.2708}},
        {"zone_id": "GULF-OF-MANNAR-PFZ-003", "center": {"lat": 8.4375, "lon": 79.7708}},
    ],
    "visakhapatnam": [
        {"zone_id": "VISAKHAPATNAM-PFZ-001", "center": {"lat": 17.4792, "lon": 83.1875}},
        {"zone_id": "VISAKHAPATNAM-PFZ-002", "center": {"lat": 17.1042, "lon": 82.4792}},
        {"zone_id": "VISAKHAPATNAM-PFZ-003", "center": {"lat": 17.8542, "lon": 83.8125}},
    ],
    "paradip": [
        {"zone_id": "PARADIP-PFZ-001", "center": {"lat": 19.8542, "lon": 86.4375}},
        {"zone_id": "PARADIP-PFZ-002", "center": {"lat": 20.2292, "lon": 86.8958}},
        {"zone_id": "PARADIP-PFZ-003", "center": {"lat": 20.1458, "lon": 86.6875}},
    ],
    "kolkata": [
        {"zone_id": "KOLKATA-PFZ-001", "center": {"lat": 21.8542, "lon": 88.1042}},
    ],
    "port blair": [
        {"zone_id": "PORT-BLAIR-PFZ-001", "center": {"lat": 12.0625, "lon": 93.2292}},
        {"zone_id": "PORT-BLAIR-PFZ-002", "center": {"lat": 11.7708, "lon": 92.4792}},
        {"zone_id": "PORT-BLAIR-PFZ-003", "center": {"lat": 12.2708, "lon": 93.0208}},
    ],
}
