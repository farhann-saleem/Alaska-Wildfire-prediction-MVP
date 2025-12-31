# 🛠️ Installation & Setup Guide

This guide covers installation and usage for both Phase 1 (CNN training) and Phase 2 (weather analysis).

---

## Prerequisites

- **Python:** 3.10 or higher
- **RAM:** 8GB minimum (for in-memory patch loading)
- **GPU:** Optional but recommended (NVIDIA GPU with CUDA for faster training)
- **Disk Space:** ~5GB for dataset and dependencies

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/farhann-saleem/Alaska-Wildfire-prediction-MVP.git
cd wildfire-prediction-mvp
```

### 2. Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies include:**
- Core: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- Geospatial: `rasterio`
- Deep Learning: `tensorflow`, `jupyter`
- Phase 2: `earthengine-api`, `scipy`, `xarray`, `netCDF4`

---

## Phase 1: Wildfire Detection Model

### Data Preprocessing

Extract 64×64 pixel patches from aligned GeoTIFF imagery:

```bash
python scripts/preprocess.py
```

**Output:**
- `data/patches/` - Extracted image patches
- `data/patch_metadata.csv` - Patch locations and burn labels

### Model Training

**Option A: Command Line**

```bash
python scripts/train_model.py
```

**Option B: Interactive Notebook**

```bash
jupyter notebook scripts/main.ipynb
```

**Training Configuration:**
- Patch size: 64×64 pixels
- Batch size: 32
- Epochs: 50 (with early stopping)
- Optimizer: Adam (lr=0.0001)
- Loss: Categorical crossentropy with 10× sample weighting
- Decision threshold: 0.3 (tuned for recall)

### View Results

Training automatically generates:
- `assets/training_results.png` - Comprehensive visualization
- Model checkpoints saved during training

---

## Phase 2: Weather Analysis

### Google Earth Engine Setup

1. **Authenticate GEE** (one-time):

```bash
python -c "import ee; ee.Authenticate()"
```

This will open a browser for Google account authorization.

2. **Set Project ID:**

The script uses project `alaska-dataset`. If you have a different GEE project:
- Edit line 22 in `scripts/era5_analysis.py`
- Change `GEE_PROJECT = 'alaska-dataset'` to your project ID

### Run Weather Analysis

```bash
python scripts/era5_analysis.py
```

**What it does:**
- Loads 511 burn patches from Phase 1
- Fetches ERA5-Land hourly weather for 30 days pre-fire
- Calculates VPD (Vapor Pressure Deficit)
- Generates correlation analysis and visualizations

**Expected Runtime:** 40-60 minutes (511 patches × 5-10 sec each)

**Outputs:**
- `results/phase2/era5_weather_correlations.csv` - Weather statistics
- `results/phase2/correlation_heatmap.png` - Variable correlations
- `results/phase2/variable_distributions.png` - Weather distributions
- `results/phase2/vpd_distribution.png` - VPD analysis

---

## Troubleshooting

### Google Earth Engine Errors

**Error:** `ee.Initialize: no project found`

**Solution:** Check your GEE project ID at https://code.earthengine.google.com and update `scripts/era5_analysis.py` line 22.

---

**Error:** `Not logged in to Earth Engine`

**Solution:** Run authentication again:
```bash
python -c "import ee; ee.Authenticate()"
```

### TensorFlow GPU Issues

**Error:** `Could not load dynamic library libcudart.so`

**Solution:** TensorFlow will fall back to CPU. For GPU support, install CUDA Toolkit matching your TensorFlow version.

### Memory Issues During Training

**Error:** `OOM (Out of Memory)`

**Solution:** Reduce batch size in `scripts/train_model.py`:
```python
BATCH_SIZE = 16  # Instead of 32
```

---

## Data Sources

### Phase 1 Data
- **Satellite Imagery:** Sentinel-2 Level-2A (June 2021)
- **Labels:** MTBS burn severity maps (Q3 2021)
- **Region:** Alaska wildfire zones
- **Download:** See `scripts/export_data_gee.js`

### Phase 2 Data
- **Weather:** ERA5-Land Hourly (via Google Earth Engine)
- **Source:** `ECMWF/ERA5_LAND/HOURLY` dataset
- **Access:** Free via GEE (no CDS API key needed)

---

## Next Steps

After setup:
1. **Phase 1:** Run preprocessing → train model → analyze results
2. **Phase 2:** Authenticate GEE → run weather analysis
3. **Documentation:** Read `docs/` for detailed methodology

For research collaboration or questions, see [CONTRIBUTING.md](CONTRIBUTING.md).
