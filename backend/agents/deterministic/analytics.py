"""Ocean Analytics deterministic module — no LLM call.

Evaluates SST and chlorophyll against favorable fishing thresholds using
pandas / numpy. Never raises across the boundary; always returns a valid
AnalyticsResult and appends a TraceStep to bundle.trace.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.schemas.contracts import AnalyticsResult, EvidenceBundle, TraceStep
from backend.time_utils import now_iso as _iso_now

# Favorable fishing thresholds (tropical Indian Ocean / INCOIS PFZ guidance)
SST_OPTIMAL_MIN = 27.0
SST_OPTIMAL_MAX = 29.5
SST_ACCEPTABLE_MIN = 25.0
SST_ACCEPTABLE_MAX = 31.0
SST_EXTREME_LOW = 24.0
SST_EXTREME_HIGH = 32.0

CHL_OPTIMAL_MIN = 0.2
CHL_OPTIMAL_MAX = 1.0
CHL_LOW_THRESHOLD = 0.1
CHL_BLOOM_THRESHOLD = 5.0
CHL_EXTREME_HIGH = 10.0


def run_ocean_analytics(bundle: EvidenceBundle) -> AnalyticsResult:
    """Correlate chlorophyll + SST against known favorable fishing thresholds.

    Flags anomalies and produces a trend summary. Uses pandas/numpy
    for threshold evaluation. Always appends a TraceStep before return.
    """
    anomalies: list[str] = []
    trend_summary: str = ""
    input_summary: str = ""
    error_occurred = False

    try:
        marine = bundle.marine
        sst = marine.sst_celsius if marine is not None else None
        chl = marine.chlorophyll_mg_m3 if marine is not None else None

        # Build input summary early (for trace)
        if marine is None:
            input_summary = "marine=None"
        else:
            input_summary = (
                f"sst={sst}, chl={chl}, status={marine.status}, "
                f"pfz_zones={len(marine.pfz_zones)}"
            )

        # Use pandas DataFrame to demonstrate vectorized threshold logic
        df = pd.DataFrame([{"sst": sst, "chl": chl}])

        # Handle missing marine entirely
        if marine is None:
            anomalies.append("Marine data unavailable — cannot evaluate SST/chlorophyll")
            trend_summary = "Insufficient data: marine payload missing."
        else:
            # Check status error case still attempts to evaluate whatever values exist
            sst_val = df["sst"].iloc[0]
            chl_val = df["chl"].iloc[0]

            sst_missing = pd.isna(sst_val)
            chl_missing = pd.isna(chl_val)

            # --- SST evaluation ---
            if sst_missing:
                anomalies.append("SST data unavailable")
            else:
                sst_f = float(sst_val)
                # Use numpy for threshold checks
                if np.isnan(sst_f):
                    anomalies.append("SST data unavailable")
                elif sst_f < SST_EXTREME_LOW:
                    anomalies.append(
                        f"SST anomalously low ({sst_f:.1f}°C) — below {SST_EXTREME_LOW}°C; possible upwelling/monsoon cooling"
                    )
                elif sst_f < SST_ACCEPTABLE_MIN:
                    anomalies.append(
                        f"SST below favorable range ({sst_f:.1f}°C < {SST_ACCEPTABLE_MIN}°C)"
                    )
                elif sst_f > SST_EXTREME_HIGH:
                    anomalies.append(
                        f"SST anomalously high ({sst_f:.1f}°C) — above {SST_EXTREME_HIGH}°C; possible heatwave/bleaching risk"
                    )
                elif sst_f > SST_ACCEPTABLE_MAX:
                    anomalies.append(
                        f"SST above favorable range ({sst_f:.1f}°C > {SST_ACCEPTABLE_MAX}°C)"
                    )

            # --- Chlorophyll evaluation ---
            if chl_missing:
                anomalies.append("Chlorophyll data unavailable")
            else:
                chl_f = float(chl_val)
                if np.isnan(chl_f):
                    anomalies.append("Chlorophyll data unavailable")
                elif chl_f > CHL_EXTREME_HIGH:
                    anomalies.append(
                        f"Chlorophyll extremely high ({chl_f:.1f} mg/m³) — intense bloom, potential HAB risk"
                    )
                elif chl_f > CHL_BLOOM_THRESHOLD:
                    anomalies.append(
                        f"Chlorophyll high ({chl_f:.1f} mg/m³) — algal bloom conditions, fishing productivity may be affected"
                    )
                elif chl_f < CHL_LOW_THRESHOLD:
                    anomalies.append(
                        f"Chlorophyll very low ({chl_f:.2f} mg/m³) — low productivity water"
                    )

            # --- Trend summary ---
            if not anomalies:
                # Both within optimal?
                s_ok = not sst_missing and SST_OPTIMAL_MIN <= float(sst_val) <= SST_OPTIMAL_MAX
                c_ok = not chl_missing and CHL_OPTIMAL_MIN <= float(chl_val) <= CHL_OPTIMAL_MAX
                if s_ok and c_ok:
                    trend_summary = (
                        f"Conditions favorable: SST {float(sst_val):.1f}°C and chlorophyll "
                        f"{float(chl_val):.2f} mg/m³ within optimal fishing thresholds "
                        f"(SST {SST_OPTIMAL_MIN}-{SST_OPTIMAL_MAX}°C, chl {CHL_OPTIMAL_MIN}-{CHL_OPTIMAL_MAX} mg/m³)."
                    )
                elif s_ok or c_ok:
                    trend_summary = (
                        f"Conditions moderately favorable: one parameter optimal "
                        f"(SST {float(sst_val):.1f}°C, chl {float(chl_val):.2f} mg/m³); combined PFZ potential moderate."
                    )
                else:
                    # Within acceptable but not optimal
                    trend_summary = (
                        f"Conditions acceptable but not optimal: SST {float(sst_val):.1f}°C, "
                        f"chlorophyll {float(chl_val):.2f} mg/m³ outside optimal band but within tolerable range."
                    )
            else:
                # Build summary listing anomalies
                joined = "; ".join(anomalies)
                trend_summary = f"Anomalies detected: {joined}."

                # Add favorable note if applicable despite anomalies
                if not sst_missing and not chl_missing:
                    s_opt = SST_OPTIMAL_MIN <= float(sst_val) <= SST_OPTIMAL_MAX
                    c_opt = CHL_OPTIMAL_MIN <= float(chl_val) <= CHL_OPTIMAL_MAX
                    if s_opt or c_opt:
                        trend_summary += " Otherwise parameters appear optimal."

            # If marine status is partial/error, prepend context
            if marine.status in ("partial", "error") and marine.error_message:
                trend_summary = f"[{marine.status}: {marine.error_message}] {trend_summary}"

    except Exception as exc:  # never propagate
        error_occurred = True
        anomalies = [f"Analytics computation failed: {exc}"]
        trend_summary = f"Error during ocean analytics: {exc}"
        if not input_summary:
            input_summary = f"error capturing inputs: {exc}"

    result = AnalyticsResult(anomalies=anomalies, trend_summary=trend_summary)

    # Write back to bundle
    try:
        bundle.analytics = result
    except Exception:
        pass

    output_summary = f"anomalies={len(anomalies)}, trend='{trend_summary[:120]}'"
    if error_occurred:
        output_summary = f"error — {output_summary}"

    trace = TraceStep(
        agent_name="ocean_analytics",
        input_summary=input_summary[:500],
        output_summary=output_summary[:500],
        timestamp=_iso_now(),
    )
    bundle.trace.append(trace)

    return result
