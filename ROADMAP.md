# 🗺️ Research Trajectory

## Vision

Develop a scientific understanding of wildfire drivers in Alaska through hypothesis-driven research, using satellite imagery and atmospheric data to identify the physical processes that precede ignition in boreal ecosystems.

---

## Research Philosophy

This project follows an **evidence-driven approach**, not an architecture-driven one. Each phase tests a scientific hypothesis, and the results determine the next research direction. We don't commit to specific models or data sources until empirical evidence justifies them.

---

## Completed Research

### ✅ Phase 1: Satellite-Based Detection Hypothesis

**Duration:** December 1-17, 2024  
**Research Question:** *Can deep learning detect wildfire patterns from satellite imagery despite extreme class imbalance?*

#### Hypothesis
Sentinel-2 optical imagery contains spatial signatures of pre-fire conditions that are detectable via convolutional neural networks.

#### Methodology
- **Data:** 7,000+ patches from Sentinel-2 Level-2A (June 2021)
- **Labels:** MTBS burn severity maps (Q3 2021)
- **Model:** Enhanced CNN with residual blocks
- **Challenge:** 98.3% class imbalance (1.7% burn patches)

#### Results

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Recall (Burn) | 58.6% | > 50% | ✅ Exceeded |
| Accuracy | 89.8% | > 80% | ✅ Exceeded |
| F1 Score | 16.5% | > 10% | ✅ Exceeded |

#### Scientific Findings
1. **Spatial patterns exist:** CNN successfully learned fire signatures
2. **Class imbalance solvable:** Sample weighting (10×) prevented model collapse
3. **Threshold matters:** Adjusted decision boundary (0.5 → 0.3) for safety-critical systems

**Conclusion:** ✅ Hypothesis SUPPORTED - Satellite-based detection is viable for Alaska

---

### ✅ Phase 2: Weather-Driven Fire Hypothesis

**Duration:** December 27-31, 2024  
**Research Question:** *Do traditional fire weather variables (temperature, VPD, precipitation) correlate with Alaska wildfire ignition?*

#### Hypothesis
Alaska wildfires occur under high VPD (> 1.0 kPa), low precipitation (< 50mm/month), and elevated temperatures (> 25°C), similar to temperate fire regimes.

#### Methodology
- **Data:** ERA5-Land Hourly via Google Earth Engine
- **Fire Events:** 511 burn patches from Phase 1
- **Temporal Window:** 30 days pre-fire (prevents data leakage)
- **Variables:** 
  - Temperature (2m)
  - Total Precipitation
  - Wind Speed
  - Soil Moisture (0-7cm)
  - **VPD (Vapor Pressure Deficit)** - Key fire driver

#### Results

| Variable | Mean Value | Expected for High Risk | Assessment |
|----------|------------|------------------------|------------|
| **VPD** | **0.70 kPa** | > 1.0 kPa | **LOW** |
| **Precipitation** | **739 mm** | < 50 mm/month | **WET** |
| **Temperature** | **15.7°C** | > 25°C | **MODERATE** |

#### Scientific Findings

**🚨 SURPRISING RESULT: Hypothesis REJECTED**

Alaska fires occurred in **LOW** traditional fire-risk weather conditions:
- VPD below typical fire threshold (0.70 vs. 1.0 kPa)
- Extremely high precipitation (739mm vs. 50-100mm normal)
- Moderate temperatures (not hot)

#### Implications

1. **Alaska fires are mechanistically different** from temperate/Mediterranean wildfires
2. **Weather alone is insufficient** for prediction in boreal ecosystems
3. **Alternative drivers likely:**
   - **Lightning strikes** (can ignite even in wet conditions)
   - **Temporal lag effects** (fires ignite weeks after dry period)
   - **Fuel structure** (mosses, lichens behave differently than grass/leaves)
   - **Permafrost dynamics** (affects soil moisture differently)

4. **Validates multi-modal approach:** Need satellite imagery to capture spatial fuel conditions + weather context

**Visualizations:**
- Correlation heatmap: Variable relationships
- Distribution analysis: Pre-fire weather patterns
- VPD analysis: 511 fire events

**Documentation:** [docs/phase2-weather-analysis.md](../docs/phase2-weather-analysis.md)

**Conclusion:** ❌ Hypothesis REJECTED - Weather alone does NOT explain Alaska fires  
**Pivot:** Multi-modal integration essential, not optional

---

## Proposed Research (Contingent on GSoC 2026)

### 🔬 Phase 3: Multi-Modal Fire Driver Analysis

**Status:** Proposed (contingent on GSoC acceptance)  
**Research Question:** *Can multi-modal fusion (SAR + optical + weather + lightning) improve prediction beyond any single data source?*

#### Rationale (Evidence-Based)

Phase 2 findings demand a multi-modal approach:
- **Weather:** Context, not primary signal (low correlation)
- **Satellite (Phase 1):** Spatial patterns detected (58.6% recall)
- **Missing:** All-weather imaging, temporal dynamics, lightning data

#### Proposed Architecture

```
┌─────────────────┐
│ Sentinel-2 CNN  │ ← Phase 1 (Spatial features)
│  (Optical)      │
└────────┬────────┘
         │
         ▼
    ┌─────────────────────┐
    │ Multi-Modal Fusion  │
    │  (Late Fusion)      │
    └─────────────────────┘
         ▲        ▲        ▲
         │        │        │
┌────────┴──┐ ┌──┴────┐ ┌─┴───────┐
│ Sentinel-1│ │ ERA5  │ │ NOAA    │
│    SAR    │ │Weather│ │Lightning│
│(All-weather)│(Context)│ (Trigger)│
└───────────┘ └───────┘ └─────────┘
         ↓
  Fire Risk Probability
```

#### Hypotheses to Test

1. **Sentinel-1 SAR Hypothesis:**  
   *SAR backscatter changes (VV/VH polarization) indicate vegetation moisture stress 7-14 days before ignition*
   
2. **Temporal Lag Hypothesis:**  
   *60-day weather window shows stronger correlation than 30-day (fuel drying accumulates)*
   
3. **Lightning Trigger Hypothesis:**  
   *Lightning density within 10km of patch predicts ignition better than temperature*

#### Proposed Methodology

**Weeks 1-4: Data Integration**
- Sentinel-1 GEE export and preprocessing
- NOAA Lightning Detection Network integration
- Extend ERA5 temporal window to 60 days

**Weeks 5-8: Model Development**
- Multi-input CNN architecture
- Late fusion layer (preserve modality-specific signals)
- Ablation study (each data source individually)

**Weeks 9-12: Analysis & Validation**
- Statistical hypothesis testing
- Temporal cross-validation (prevent data leakage)
- Interpretability analysis (SHAP values per modality)

#### Expected Outcomes

| Metric | Phase 1 (Optical) | Phase 2 (+ Weather) | Phase 3 Target (Multi-Modal) |
|--------|-------------------|---------------------|------------------------------|
| Recall | 58.6% | N/A | **> 75%** |
| Precision | 9.6% | N/A | **> 20%** |
| Lead Time | 0 days | N/A | **7-14 days** |

**Note:** Targets are provisional. Research is exploratory, not guaranteed.

---

## Long-Term Vision (Beyond GSoC)

### Operational Deployment (2027+)

If Phase 3 succeeds:
- Real-time ingestion pipeline (automated GEE downloads)
- Cloud deployment (AWS/GCP)
- Stakeholder integration (Alaska fire management agencies)

### Research Extensions

- **Fire severity prediction:** Multi-class output (low/medium/high)
- **Emission modeling:** CO₂ and PM2.5 from predicted fires
- **Climate scenarios:** RCP 4.5/8.5 projections for 2050/2100
- **Transfer learning:** Apply to Canada, Siberia (circumpolar regions)

---

## Success Criteria

### Phase 3 Validation

To consider Phase 3 successful, we must:
1. **Beat Phase 1 baseline:** Recall > 58.6% (multi-modal must outperform optical-only)
2. **Statistical significance:** Each modality contributes significantly (ablation study)
3. **Interpretability:** Understand *why* model makes predictions (not black box)
4. **Temporal robustness:** Works on held-out years (2019, 2020)

### Scientific Impact

- **Publications:** 1-2 peer-reviewed papers (focus on Alaska-specific fire dynamics)
- **Community:** Open-source dataset + code for circumpolar fire research
- **Collaboration:** Work with Alaska fire ecologists to validate findings

---

## Research Principles

1. **Hypothesis-driven, not architecture-driven**  
   We test scientific questions, not build predetermined models

2. **Evidence-based pivots**  
   Phase 2 rejected weather-only hypothesis → Phase 3 adapts accordingly

3. **Interpretability over performance**  
   Understanding fire drivers > achieving highest accuracy

4. **Reproducibility**  
   All code, data sources, and methods openly documented

---

## Updates & Timeline

| Date | Phase | Update |
|------|-------|--------|
| 2024-12-17 | Phase 1 | ✅ Satellite detection proven viable (58.6% recall) |
| 2024-12-31 | Phase 2 | ✅ Weather hypothesis rejected - Alaska fires differ from temperate regimes |
| 2025-01-15 | Phase 3 | 📋 GSoC proposal submitted (multi-modal approach) |
| 2026-05-27 | Phase 3 | 🚀 GSoC begins (if accepted) |

---

**Last Updated:** December 31, 2024  
**Project Lead:** Farhan Saleem  
**GSoC Organization:** [University of Alaska Anchorage](https://github.com/uaanchorage/GSoC)  
**Repository:** [Alaska-Wildfire-prediction-MVP](https://github.com/farhann-saleem/Alaska-Wildfire-prediction-MVP)

---

<p align="center">
  <i>Research is not about confirming your hypothesis - it's about discovering the truth.</i>
</p>
