# 🔥 Alaska Wildfire Prediction - Research Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-red.svg)](https://summerofcode.withgoogle.com/)

> **Hypothesis-driven research on wildfire drivers in Alaska using satellite imagery and weather data.**  
> Developed for [Google Summer of Code 2026](https://github.com/uaanchorage/GSoC) application with University of Alaska Anchorage.

---

## 📖 Overview

Alaska faces increasing wildfire risks due to climate change. This research project takes a hypothesis-driven approach to understanding wildfire drivers, starting with satellite-based detection and expanding to weather correlation analysis. Rather than immediately building a "hybrid model," we test scientific hypotheses about what factors drive Alaska wildfires.

### 🎯 Research Objectives

- ✅ **Phase 1 (Complete):** Prove satellite-based detection is viable
- ✅ **Phase 2 (Complete):** Test weather-driven fire hypothesis  
- 🔬 **Phase 3 (Proposed):** Multi-modal integration based on empirical findings

---

## 🏆 Research Results

### Phase 1: Baseline Detection Model

![Training Results](assets/training_results.png)

**Achievements:**
- **Accuracy:** 89.8% overall classification
- **Recall:** 58.6% for wildfire detection (burn class)
- **Model:** Enhanced CNN with residual blocks
- **Dataset:** 7,000+ patches from Alaska 2021 fire season
- **Challenge Solved:** Extreme class imbalance (1.7% positive samples)

> **Significance:** Demonstrated that deep learning can detect wildfire patterns in satellite imagery despite severe class imbalance, proving viability for Alaska deployment.

---

### Phase 2: Weather Hypothesis Testing

**Research Question:** *Do traditional fire weather variables (temperature, precipitation, VPD) correlate with Alaska wildfire ignition?*

**Methodology:**
- **Data:** ERA5-Land hourly weather via Google Earth Engine
- **Fire Events:** 511 burn patches from Phase 1
- **Temporal Window:** 30 days pre-fire (prevents data leakage)
- **Variables:** Temperature, Precipitation, Wind, Soil Moisture, VPD

**Key Finding: Alaska Fires Occur in LOW Traditional Fire-Risk Weather**

| Variable | Mean Value | Traditional Risk Threshold | Assessment |
|----------|------------|---------------------------|------------|
| **VPD** | 0.70 kPa | \> 1.0 kPa (high risk) | **LOW** |
| **Precipitation** | 739 mm/month | 50-100 mm (normal) | **WET** |
| **Temperature** | 15.7°C | \> 30°C (high risk) | **MODERATE** |

**Scientific Implications:**
1. **Alaska fires are different** from temperate/Mediterranean wildfires
2. **Weather alone is insufficient** for prediction  
3. **Validates multi-modal approach** - need satellite imagery + weather combined
4. **Suggests lightning/lag effects** - ignition may not correlate with immediate weather

**Visualizations:**
- Correlation heatmap of weather variables
- Pre-fire weather distributions  
- VPD analysis for 511 fire events

📊 **Full Analysis:** See [docs/phase2-weather-analysis.md](docs/phase2-weather-analysis.md)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM
- Optional: NVIDIA GPU with CUDA

### Installation

```bash
# Clone the repository
git clone https://github.com/farhann-saleem/Alaska-Wildfire-prediction-MVP.git
cd wildfire-prediction-mvp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Phase 1: Train Detection Model

```bash
# Extract patches from satellite imagery
python scripts/preprocess.py

# Train CNN model
python scripts/train_model.py
# OR use interactive notebook:
jupyter notebook scripts/main.ipynb
```

#### Phase 2: Weather Analysis

```bash
# Activate Phase 2 environment (includes GEE dependencies)
pip install earthengine-api pandas seaborn matplotlib

# Authenticate Google Earth Engine
python -c "import ee; ee.Authenticate()"

# Run weather correlation analysis
python scripts/era5_analysis.py
```

**Outputs:**
- `results/phase2/era5_weather_correlations.csv`
- `results/phase2/correlation_heatmap.png`
- `results/phase2/variable_distributions.png`
- `results/phase2/vpd_distribution.png`

---

## 📁 Project Structure

```
wildfire-prediction-mvp/
├── configs/
│   └── default_config.yaml        # Configuration parameters
├── data/
│   ├── raw/                       # Raw GeoTIFF imagery
│   ├── patches/                   # Extracted patches (generated)
│   └── patch_metadata.csv         # Patch metadata (generated)
├── docs/                          # Research documentation
│   ├── debugging-journey.md       # Phase 1 technical challenges
│   ├── phase2-weather-analysis.md # Phase 2 empirical findings
│   ├── architecture.md            # System design
│   └── data-pipeline.md           # Data processing workflow
├── results/
│   └── phase2/                    # Phase 2 outputs
├── scripts/
│   ├── preprocess.py              # Phase 1 preprocessing
│   ├── train_model.py             # Phase 1 training
│   ├── era5_analysis.py           # Phase 2 weather analysis
│   └── main.ipynb                 # Interactive training notebook
├── src/
│   └── data_pipeline/             # Patch extraction utilities
└── assets/                        # Visualization outputs
```

---

## 🔧 Engineering Challenges (Phase 1)

### Challenge 1: Softmax Collapse

**Problem:** Model achieved 98.3% accuracy by predicting "No Burn" for everything.

**Root Cause:** With 1.7% positive samples, the loss signal from rare fires was too weak.

**Solution:**
- Sample weighting (10× boost for minority class)
- One-hot encoding for numerical stability
- Categorical cross-entropy loss

**Result:** Model learned to detect fires (58.6% recall)

---

### Challenge 2: Gradient Instability

**Problem:** Aggressive class weights (58×) caused training instability.

**Solution:**
- Reduced weight scaling (58× → 10×)
- Lower learning rate (0.01 → 0.0001)
- Pixel value clipping
- Early stopping

---

### Challenge 3: Precision-Recall Trade-off

**Problem:** Default 0.5 threshold wasn't optimal for early warning systems.

**Solution:**
- Tuned threshold to 0.3 (prioritize catching fires over minimizing false alarms)
- Justification: For safety-critical systems, false positives are acceptable; false negatives are dangerous

**Result:**
```
Threshold 0.5: Recall 45%, Precision 15%
Threshold 0.3: Recall 58.6%, Precision 9.6%  ✓ Chosen
```

---

## 🔬 Future Research Directions

Based on Phase 2 findings, future work should focus on:

### Proposed Phase 3: Multi-Modal Integration

**Rationale:** Weather analysis revealed weak direct correlation → need complementary signals

**Recommended Approach:**
1. **Sentinel-1 SAR Integration**
   - All-weather imaging (penetrates clouds)
   - Detects vegetation stress and soil moisture changes
   - May capture drying patterns weather data misses

2. **Temporal Lag Analysis**
   - Test 60-day vs. 30-day pre-fire windows
   - Fire may ignite weeks after dry period ends
   - CNN-LSTM for sequential pattern learning

3. **Lightning Strike Data**
   - Alaska fires are predominantly lightning-caused
   - Integrate NOAA Lightning Detection Network
   - Could explain why wet-weather fires occur

**Note:** This is a research project. Phase 3 direction is contingent on GSoC acceptance and will adapt based on empirical evidence, not pre-committed architectures.

---

## 📚 Documentation

Detailed research documentation:

- **[Debugging Journey](docs/debugging-journey.md)** - Phase 1 technical challenges and solutions
- **[Phase 2 Weather Analysis](docs/phase2-weather-analysis.md)** - Empirical findings on weather-fire correlation
- **[Architecture Overview](docs/architecture.md)** - System design
- **[Data Pipeline](docs/data-pipeline.md)** - Preprocessing methodology

---

## 🤝 Contributing

This research project welcomes scientific collaboration and feedback. If you're interested in:
- Wildfire prediction research in circumpolar regions
- Multi-modal satellite imagery analysis
- Fire weather modeling in boreal forests

Feel free to:
1. Join discussions in [GitHub Discussions](https://github.com/uaanchorage/GSoC/discussions)
2. Submit issues for bugs or questions
3. Propose research collaborations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📖 Citations

If you use this work, please cite:

```bibtex
@misc{saleem2024alaska,
  author = {Saleem, Farhan},
  title = {Alaska Wildfire Prediction - Research Project},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/farhann-saleem/Alaska-Wildfire-prediction-MVP}
}
```

**Related Work:**
- Wang, Y., et al. (2023). "Toward Energy-Efficient Deep Neural Networks for Forest Fire Detection in an Image." *The Geographical Bulletin*, 64(2), Article 13.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[University of Alaska Anchorage](https://www.uaa.alaska.edu/)** - Research mentorship
- **[Dr. Yali Wang](https://github.com/YaliWang2019)** - Project guidance and hypothesis-driven approach feedback
- **[Google Summer of Code](https://summerofcode.withgoogle.com/)** - Program framework
- **[Alaska GSoC Organization](https://github.com/uaanchorage/GSoC)** - Project coordination
- **[Sentinel-2 Mission](https://sentinel.esa.int/)** - Satellite imagery (ESA)
- **[MTBS Project](https://www.mtbs.gov/)** - Burn severity data (USGS/USFS)
- **[Copernicus ERA5](https://www.ecmwf.int/)** - Weather reanalysis data (ECMWF)

---

## 📧 Contact

**Developer:** Farhan Saleem  
**Repository:** [https://github.com/farhann-saleem/Alaska-Wildfire-prediction-MVP](https://github.com/farhann-saleem/Alaska-Wildfire-prediction-MVP)  
**GSoC Discussion:** [Alaska GSoC](https://github.com/uaanchorage/GSoC/discussions)

---

<p align="center">
  <i>🔥 Understanding wildfire drivers through hypothesis-driven research 🔥</i>
</p>
