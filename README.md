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
![Weather Correlations](results/phase2/correlation_heatmap.png)
*(See [detailed analysis](docs/phase2-weather-analysis.md) for distribution plots)*

**Scientific Implications:**
1.  **Alaska fires are mechanistically different** from temperate wildfires.
2.  **Dryness alone drives fire** (even in cool/low-VPD conditions).
3.  **Moss/Lichen dynamics** likely play a key role (rapid drying).

---

## 🏗️ System Architecture

Our hybrid pipeline integrates optical imagery with multi-source climate data.

[![Architecture](https://mermaid.ink/img/pako:eNqVkstqwzAQRX9FzCqF_AAfC9200G27KS0kXgyOrSGWjCS5hBDy7x0_bFMobTfS4M65c2RjQyljhUJS89o2vKGNw9dCl1T_Wc-V1IXWpZKaP1RXUn6yUjT8ybJ8Z6VxvLRS08_z8sVK48VK_8VK4_lK-8lK49lK-8VK49lK-8lK491K-8VKw81KO2elcLNqG979s1K4d7XhvXgrhXtXG96Lt1K4d7XhvXgrhXtXG96Lt1K4d7XhvXgrhXtXG96Lt1K4d7XhvXgrhXtXG96Lt1K4d7XhvXgrhXtXG96L91J46GrD-_FeCs9dbfj5eC-F5642_Hy8l8JzVxv-P95L4bmrDf8f76Xw3NWG_4_3Unjuanu-30vhuatt-f9et4b7Xm3L_3vdGu57tS3_73VruO_Vtvy_163hvlfP8v9et4b7Xm1X_7_XreG-V9vq__e6Ndz3alv9_163hvtebav_3-vWcN-rbfX_e90a7nu1rf5_r1vDfa-21f_vdWu479W2-v-9bg33vdqm__9et4b7Xm3T__-9bg33vdpW_7_XreG-V9vq__e6Ndz3alv9_163hvtebav_3-vWcN-rbfX_e90a7nu1rf5_r1vDfa-21f_vdWu479W2-v-9bg33vdpW_7_XreG-V9vq__e6Ndz3alv9_163hvtebav_3-vWcN-rbfX_e90a7nu1rf5_r1vDfa-21f_vdWu479W2-v-9bg33vdpW_7_XreG-V9vq__e6Ndz3alv9_163hvtebav_3-vWcN-rbfX_e90a7nu1rf5_r1vDfa-21f_vdWu479W2-v-9bg33vdpW_7_XreG-V9vq__e6Ndz3alv9_163hvtebav_3-vWcN-rbfX_e90a7nu1rf5_r1vDfa-21f_vdWu479W2-v-9bg33vdpW_7_XreG-V9vq__e6Ndz3alv9_163hvtebav_3-vWcN-rbfX_e90a7nu1rf5_r1vDfa-21f_vdWu479W2-v-9bg33vdpW_7_XreG-Vm00vN9qW_1_v9WG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91tty_9_vdWG91t)](docs/architecture.md)

👉 [Read Full Architecture Doc](docs/architecture.md)

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

## 🚀 Quick Start

**Full installation and usage guide:** [SETUP.md](SETUP.md)

```bash
# Clone repository
git clone https://github.com/farhann-saleem/Alaska-Wildfire-prediction-MVP.git
cd wildfire-prediction-mvp

# Install dependencies
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Phase 1: Train detection model
python scripts/preprocess.py
python scripts/train_model.py

# Phase 2: Weather analysis (requires GEE authentication)
python scripts/era5_analysis.py
```

---

## 📁 Project Structure

```
wildfire-prediction-mvp/
├── docs/                       # Research documentation
│   ├── phase2-weather-analysis.md    # Phase 2 empirical findings
│   ├── debugging-journey.md          # Phase 1 technical challenges
│   └── ...
├── results/phase2/             # Phase 2 outputs
├── scripts/
│   ├── preprocess.py           # Phase 1 preprocessing
│   ├── train_model.py          # Phase 1 training
│   └── era5_analysis.py        # Phase 2 weather analysis
├── src/data_pipeline/          # Utilities
├── README.md                   # This file
├── SETUP.md                    # Installation & usage guide
├── ROADMAP.md                  # Research trajectory
└── requirements.txt            # Python dependencies
```

---

## 🔧 Engineering Challenges (Phase 1)

### Softmax Collapse
**Problem:** Model predicted "No Burn" for everything (98.3% accuracy, 0% recall)  
**Solution:** Sample weighting (10×), one-hot encoding, categorical cross-entropy

### Gradient Instability
**Problem:** Aggressive class weights caused training oscillation  
**Solution:** Reduced scaling (58× → 10×), lower learning rate (0.01 → 0.0001)

**Full Technical Details:** [docs/debugging-journey.md](docs/debugging-journey.md)

---

## 🗺️ Future Research Directions

Based on Phase 2 findings, proposed Phase 3 focuses on:

**Multi-Modal Integration:**
- **Sentinel-1 SAR:** All-weather vegetation stress detection
- **Lightning Data:** NOAA strike locations (direct ignition source)
- **Temporal Modeling:** 60-day fuel accumulation analysis (CNN-LSTM)

**Architecture:** Late fusion (preserve modality-specific signals)

**Contingent on:** GSoC 2026 acceptance

📍 **Full Trajectory:** [ROADMAP.md](ROADMAP.md)

---

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Installation and usage
- **[Phase 2 Analysis](docs/phase2-weather-analysis.md)** - Weather hypothesis testing
- **[Debugging Journey](docs/debugging-journey.md)** - Phase 1 technical challenges
- **[ROADMAP.md](ROADMAP.md)** - Research trajectory

---

## 🙏 Acknowledgments

- **[Dr. Yali Wang](https://github.com/YaliWang2019)** - Research mentorship and guidance toward hypothesis-driven approach
- **[University of Alaska Anchorage](https://www.uaa.alaska.edu/)** - Project support
- **[Google Summer of Code](https://summerofcode.withgoogle.com/)** - Program framework
- **[Sentinel-2 Mission](https://sentinel.esa.int/)** - Satellite imagery (ESA)
- **[MTBS Project](https://www.mtbs.gov/)** - Burn severity data (USGS/USFS)
- **[Copernicus ERA5](https://www.ecmwf.int/)** - Weather data (ECMWF)

**Related Work:**
- Wang, Y., et al. (2023). "Toward Energy-Efficient Deep Neural Networks for Forest Fire Detection in an Image." *The Geographical Bulletin*, 64(2), Article 13.

---

## 🤝 Contributing

This research project welcomes scientific collaboration. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Developer:** Farhan Saleem  
**Repository:** [Alaska-Wildfire-prediction-MVP](https://github.com/farhann-saleem/Alaska-Wildfire-prediction-MVP)  
**GSoC Discussion:** [Alaska GSoC](https://github.com/uaanchorage/GSoC/discussions)

---

<p align="center">
  <i>🔥 Understanding wildfire drivers through hypothesis-driven research 🔥</i>
</p>
