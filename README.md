# 🔥 Alaska Wildfire Prediction using Satellite Imagery & Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-red.svg)](https://summerofcode.withgoogle.com/)

> **Early-stage MVP for wildfire risk prediction in Alaska using Sentinel-2 satellite imagery and deep learning.**  
> Developed as initial contribution for [Google Summer of Code 2026](https://github.com/uaanchorage/GSoC) with University of Alaska Anchorage.

---

## 📖 Overview

Alaska faces increasing wildfire risks due to climate change, with fires threatening communities, wildlife, and infrastructure across the state. This project develops a machine learning pipeline to predict wildfire occurrence using satellite imagery, enabling early warning systems for Alaska's unique circumpolar environment.

### 🎯 Project Goals

- ✅ **Phase 1 (Completed):** Baseline CNN model using Sentinel-2 optical imagery
- 🚀 **Phase 2 (Proposed GSoC):** Multi-modal fusion (Sentinel-1 SAR + weather data)
- 🚀 **Phase 3 (Proposed GSoC):** Temporal modeling with CNN-LSTM architecture
- 🚀 **Phase 4 (Proposed GSoC):** Web dashboard for researchers and stakeholders

---

## 🏆 Current Results (Phase 1 MVP)

![Training Results](assets/training_results.png)

**Key Achievements:**
- **Accuracy:** 89.8% overall classification accuracy
- **Recall:** 58.6% for wildfire detection (burn class)
- **Model:** Enhanced CNN with residual blocks
- **Dataset:** 7,000+ patches from Alaska wildfire season (2021)
- **Challenge Solved:** Extreme class imbalance (1.7% positive samples)

> **Significance:** 58.6% recall demonstrates the model successfully learned to detect wildfire patterns despite severe class imbalance, proving the pipeline's viability for real-world deployment.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM (for in-memory patch loading)
- Optional: NVIDIA GPU with CUDA for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction.git
cd wildfire-prediction-mvp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Data Preprocessing

Extract patches from aligned GeoTIFF imagery:

```bash
python scripts/preprocess.py
```

**Output:** `data/patch_metadata.csv` with 64x64 pixel patches and burn labels

#### 2. Model Training

Train the CNN model on extracted patches:

```bash
# Option A: Run training script
python scripts/train_model.py

# Option B: Run interactive notebook
jupyter notebook scripts/main.ipynb
```

**Training Configuration:**
- Patch size: 64×64 pixels
- Batch size: 32
- Epochs: 50 (with early stopping)
- Optimizer: Adam (lr=0.0001)
- Loss: Categorical cross-entropy with sample weighting

#### 3. View Results

Training outputs are automatically saved:
- `training_results.png` - Comprehensive visualization
- Model weights saved during training

---

##  Project Structure

```
wildfire-prediction-mvp/
├── configs/
│   └── default_config.yaml        # Configuration parameters
├── data/
│   ├── raw/                       # Raw GeoTIFF imagery
│   │   ├── s2_2021_06_input_10m.tif      # Sentinel-2 input
│   │   └── burn_2021_Q3_label_10m.tif    # Burn severity labels
│   ├── patches/                   # Extracted patches (generated)
│   └── patch_metadata.csv         # Patch metadata (generated)
├── dataset/                       # Original dataset files
├── docs/                          # Project documentation
│   ├── architecture.md            # System design
│   ├── data-pipeline.md          # Data processing workflow
│   └── model-training.md         # ML methodology
├── scripts/
│   ├── export_data_gee.js        # Google Earth Engine export script
│   ├── preprocess.py             # Data preprocessing
│   ├── main.ipynb                # Training notebook
│   └── train_model.py            # Training script
├── src/
│   └── data_pipeline/
│       └── patch_extraction.py   # Patch extraction utilities
├── assets/                        # Images and media
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

##  Technical Approach

### Data Pipeline

1. **Satellite Imagery:** Sentinel-2 Level-2A (10m resolution, RGB bands)
2. **Labels:** MTBS burn severity maps (2021 Alaska fire season)
3. **Alignment:** Co-registered in Google Earth Engine
4. **Preprocessing:** 64×64 patch extraction with burn/no-burn binary labels
5. **Normalization:** Reflectance values scaled to [0, 1]

### Model Architecture

**Enhanced CNN with Residual Blocks:**
- Initial feature extraction (5×5 conv, 32 filters)
- 2× Residual blocks with skip connections
- Global pooling + dropout (0.4) for regularization
- 128-unit fully connected layer
- Softmax output (2 classes)

**Key Innovations:**
- Custom focal loss implementation (no external dependencies)
- Sample weighting (10× boost for minority class)
- Early stopping with learning rate reduction

### Class Imbalance Handling

The dataset has extreme imbalance (98.3% no-burn, 1.7% burn). Solutions implemented:
- ✅ Stratified train-test split
- ✅ Sample weighting during training
- ✅ Focal loss alternative (optional)
- ✅ Adjusted prediction threshold (0.3 instead of 0.5)

---

##  Dataset

**Source:** [Alaska Satellite Imagery Wildfire Prediction](https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction)

**Specifications:**
- **Region:** Alaska, USA
- **Time Period:** June 2021 (pre-fire imagery) → Q3 2021 (burn labels)
- **Imagery:** Sentinel-2 Level-2A Surface Reflectance
- **Labels:** MTBS (Monitoring Trends in Burn Severity)
- **Total Patches:** ~7,000
- **Spatial Resolution:** 10 meters per pixel

---

##  Roadmap

###  Phase 1: Baseline CNN (Completed)
- [x] Data pipeline for Sentinel-2 + MTBS
- [x] Patch extraction and metadata generation
- [x] Enhanced CNN with residual blocks
- [x] Class imbalance mitigation
- [x] Training visualization and metrics
- [x] Achieve >50% recall on test set

###  Phase 2: Multi-modal Fusion (GSoC 2026 Proposal)
- [ ] Integrate Sentinel-1 SAR data (cloud-penetrating radar)
- [ ] Add weather variables (temperature, humidity, wind)
- [ ] Implement fusion architecture (early/late fusion experiments)
- [ ] Reduce false positive rate

###  Phase 3: Temporal Modeling (GSoC 2026 Proposal)
- [ ] Time-series data preparation (multi-temporal Sentinel-2)
- [ ] CNN-LSTM hybrid architecture
- [ ] Temporal attention mechanisms
- [ ] Fire progression prediction

###  Phase 4: Deployment (GSoC 2026 Proposal)
- [ ] Web-based inference dashboard
- [ ] Real-time prediction API
- [ ] Interactive map visualization (Leaflet/Mapbox)
- [ ] User documentation and tutorials

---

##  Contributing

This project is being developed as part of **Google Summer of Code 2026** with the University of Alaska Anchorage. Contributions, suggestions, and feedback are welcome!

**How to Contribute:**
1. Check the [GSoC project ideas](https://github.com/uaanchorage/GSoC)
2. Join discussions in the [GitHub Discussions](https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction/discussions)
3. Submit pull requests with improvements
4. Report issues or request features

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

##  Documentation

Detailed documentation is available in the `docs/` directory:

- **[Architecture Overview](docs/architecture.md)** - System design and data flow
- **[Data Pipeline](docs/data-pipeline.md)** - Preprocessing and patch extraction
- **[Model Training](docs/model-training.md)** - ML methodology and experiments

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **[University of Alaska Anchorage](https://www.uaa.alaska.edu/)** - Project mentorship and support
- **[Google Summer of Code](https://summerofcode.withgoogle.com/)** - Program framework and funding
- **[Alaska GSoC Organization](https://github.com/uaanchorage/GSoC)** - Project coordination
- **[Sentinel-2 Mission](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)** - Satellite imagery data
- **[MTBS Project](https://www.mtbs.gov/)** - Burn severity mapping

---

##  Contact

**Developer:** [Your Name]  
**Project Repository:** [https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction](https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction)  
**GSoC Organization:** [Alaska GSoC](https://github.com/uaanchorage/GSoC)

---

<p align="center">
  <i>🔥 Protecting Alaska's communities and ecosystems through machine learning 🔥</i>
</p>
