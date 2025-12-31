# 🔥 Alaska Wildfire Prediction - Research Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-red.svg)](https://summerofcode.withgoogle.com/)

> **Hypothesis-driven research on wildfire drivers in Alaska using satellite imagery and weather data.**  
> Developed for [Google Summer of Code 2026](https://github.com/uaanchorage/GSoC) application with University of Alaska Anchorage.

---

## 📖 Overview

Alaska faces increasing wildfire risks due to climate change. This research project investigates wildfire drivers through hypothesis-driven analysis, testing scientific assumptions about what causes fires in boreal ecosystems. Under mentorship from [Dr. Yali Wang](https://github.com/YaliWang2019), the project shifted from architecture-driven development to evidence-based research, prioritizing scientific understanding over model performance.

---

## 🏆 Research Results

### ✅ Phase 1: Satellite-Based Detection (Complete)

**Research Question:** *Can deep learning detect wildfire patterns from satellite imagery despite extreme class imbalance?*

![Training Results](assets/training_results.png)

**Key Achievements:**
- **Recall:** 58.6% for wildfire detection (burn class)
- **Accuracy:** 89.8% overall classification
- **Model:** Enhanced CNN with residual blocks
- **Dataset:** 7,000+ patches from Alaska 2021 fire season
- **Challenge Solved:** Extreme class imbalance (1.7% positive samples)

> **Finding:** Spatial patterns in Sentinel-2 optical imagery contain detectable fire signatures. Viable for Alaska deployment.

---

### ✅ Phase 2: Weather Hypothesis Testing (Complete)

**Research Question:** *Do traditional fire weather variables (temperature, VPD, precipitation) correlate with Alaska wildfire ignition?*

**Key Finding: Alaska Fires Occur in Unexpected Conditions (The "Low Risk" Paradox)**

| Variable | Observed (Alaska Fires) | High Fire Risk Threshold | Assessment |
|----------|------------------------|-------------------------|------------|
| **VPD** | 0.70 kPa | > 1.0 kPa | **LOW** ❌ |
| **Precipitation** | **30.9 mm/month** | < 50 mm/month | **DRY** ✅ |
| **Temperature** | 15.7°C | > 25-30°C | **MODERATE** ❌ |

> **Note on Data Accuracy:** Initial ERA5 models erroneously reported 739mm of rain (flood conditions). We corrected this by switching to **NASA GPM Satellite Data** (IMERG V06), confirming the dry conditions.
>
> 📖 **Read the Full Story:** [The Debugging Journey: Solving the Amazon Rainfall Anomaly](docs/debugging_journey.md)

#### Phase 2 Visualizations (511 Fire Patches)

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/a14c7da1-ccb2-4e44-b497-eb9fa8cc176d" alt="Correlation Heatmap" width="100%">
      <br><b>Correlation Heatmap</b>
    </td>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/0bac38a1-69e1-41e0-8ddd-38a2f10e3a08" alt="Variable Distributions" width="100%">
      <br><b>Variable Distributions</b>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="https://github.com/user-attachments/assets/b12f950e-e887-45f1-ae3d-37e74347f228" alt="VPD Distribution" width="60%">
      <br><b>VPD Distribution Analysis</b>
    </td>
  </tr>
</table>

*(See [detailed analysis](docs/phase2-weather-analysis.md) for full interpretation)*

**Scientific Implications:**
1.  **Alaska fires are mechanistically different** from temperate wildfires.
2.  **Dryness alone drives fire** (even in cool/low-VPD conditions).
3.  **Moss/Lichen dynamics** likely play a key role (rapid drying).

---

## 🏗️ System Architecture

Our hybrid pipeline integrates optical imagery with multi-source climate data.

👉 [**Read Full Architecture Document**](docs/architecture.md)

> *See `docs/architecture.md` for the detailed Mermaid diagram of the System & Data Pipeline.*

---

## 🔬 Methodology

### Phase 1: CNN Baseline
- **Data:** Sentinel-2 Level-2A imagery
- **Model:** ResNet-style CNN with Focal Loss
- **Result:** 58.6% Recall (Strong Baseline)

### Phase 2: Weather Analysis
- **Data:** **Hybrid (ERA5 + NASA GPM)**
- **Innovation:** Corrected precipitation data using satellite ground truth
- **Analysis:** 511 burn patches (30-day pre-fire window)

---

## 🛠 Project Structure

```bash
wildfire-prediction-mvp/
├── docs/                   # 📚 Documentation
│   ├── architecture.md     # System Design
│   ├── debugging_journey.md# The "Amazon Rainfall" anomaly fix
│   └── phase2-weather-analysis.md
├── scripts/                # 🐍 Analysis Scripts
│   ├── era5_analysis.py    # Main Hybrid Weather Pipeline
│   └── preprocess.py       # Data Prep
├── results/                # 📊 Output Artifacts
│   └── phase2/             # Correlations & Plots
├── data/                   # 🗺️ Data (GitIgnored)
└── README.md               # You are here
```

> **Note:** Debugging scripts have been moved to `archived_scripts/` to keep the repo clean.

---

## 🚀 Quick Start

### 1. Verification
To verify the Phase 2 analysis (Precipitation Fix):
```bash
# This will run the Hybrid Pipeline (ERA5 + GPM)
python scripts/era5_analysis.py
```

### 2. View Results
Results will appear in `results/phase2/`:
- `era5_weather_correlations.csv`
- `variable_distributions.png`

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for GSoC engagement rules.

---

## 📧 Contact

**Author:** Farhan
**Project:** Alaska Wildfire Prediction MVP (GSoC 2026)
