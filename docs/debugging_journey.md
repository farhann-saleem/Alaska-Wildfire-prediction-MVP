
# The Debugging Journey: Solving the "Amazon Rainfall" Anomaly (Phase 2)

> *"In validation, we trust. In data, we verify."*

This document logs the critical debugging challenges faced during Phase 2 of the Alaska Wildfire Prediction MVP (GSoC 2026), specifically the discovery and resolution of a major data anomaly in ERA5 weather data.

---

## 🛑 The Anomaly: "Impossible Floods"

### The Observation
On December 31, 2024, during the first full analysis of 511 fire patches, our ERA5 pipeline reported a shocking statistic:
> **Mean 30-day Pre-Fire Precipitation:** 739 mm (29 inches)

### Why This Was Impossible
1.  **Context:** 739mm is monsoon-level rainfall (like the Amazon rainforest).
2.  **Contradiction:** Wildfires naturally require dry fuel. It is physically impossible for 500+ fires to ignite spontaneously in the middle of a massive flood.
3.  **Outlier Check:** We saw almost *zero variance* across hundreds of patches, suggesting a systematic error, not a local weather event.

---

## 🕵️‍♂️ The Investigation

We paused the full run and launched a forensic investigation script (`debug_precip.py`).

### Hypothesis 1: Code/Aggregation Error
*   **Theory:** Maybe our GEE script was summing the same data multiple times?
*   **Test:** We manually pulled raw hourly values for a single fire location.
*   **Result:** The raw values from `ECMWF/ERA5_LAND/HOURLY` confirmed the high totals. The bug wasn't in our Python code; the data *itself* (or how GEE aggregated it) was reporting wet conditions.

### Hypothesis 2: Reliability of Reanalysis Data
*   **Theory:** ERA5 is a *model simulation* (Reanalysis). In complex terrains like Alaska, models can drift.
*   **Test:** We cross-referenced with **NASA GPM (Global Precipitation Measurement)**, a satellite-based product (IMERG V06) that measures actual rainfall from space.

---

## ✅ The Verification (Ground Truth)

We ran a side-by-side comparison for Fire Patch #511:

| Source | Type | 30-Day Precip | Assessment |
| :--- | :--- | :--- | :--- |
| **ERA5-Land** | Model (Reanalysis) | **673.8 mm** | ❌ IMPOSSIBLE (Flood) |
| **NASA GPM** | Satellite (IMERG) | **24.7 mm** | ✅ REALISTIC (Dry) |

**Conclusion:**
The ERA5 precipitation band in GEE was producing massive artifacts for this region/timeframe. NASA GPM provided the scientifically correct "Dry" signal (~1mm/day) consistent with fire risk.

---

## 🛠 The Fix: Hybrid Data Pipeline

We re-architected `scripts/era5_analysis.py` to use a **Hybrid Approach**:

1.  **Temperature / Wind / Soil Moisture:** Kept **ERA5-Land** (High spatial resolution, reliable for state variables).
2.  **Precipitation:** Switched to **NASA GPM IMERG V06** (Reliable satellite observation for rainfall).

### The Result
After re-running the full analysis on 511 patches:
*   **New Mean Precipitation:** **30.9 mm** (Dry)
*   **Scientific Validity:** Restored. The data now correctly shows that Alaska fires occur in dry, cool conditions (Low VPD), validating our Phase 3 need for multi-modal sensing.

---

## 🧠 Lessons for GSoC

1.  **Never Blindly Trust Data:** Even "standard" datasets like ERA5 can have edge cases in extreme latitudes like Alaska.
2.  **Cross-Validation is Key:** Always have a second source (Satellite vs Model) when numbers look "weird".
3.  **Fail Fast, Debug Deep:** We stopped the 500-patch run immediately to write a dedicated debug script, saving hours of wasted computation.
