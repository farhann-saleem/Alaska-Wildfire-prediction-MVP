# 🔥 Alaska Wildfire Prediction - GSOC 2026 MVP Roadmap

**Project Goal**: Predict wildfire risk in Alaska using satellite imagery and weather data  
**Timeline**: 350 hours (GSOC 2026)  
**Difficulty**: Medium/Hard  
**Status**: Phase 1 - MVP Development (In Progress)

---

## 📋 Project Overview

### What We're Building
A **hybrid deep learning model** that predicts wildfire risk in Alaska by integrating:
- **Optical Satellite Data**: Sentinel-2 (10m resolution, RGB + NIR bands)
- **SAR Data**: Sentinel-1 (10m resolution, moisture & terrain features)
- **Weather Time-Series**: Temperature, wind speed, humidity
- **Output**: Fire risk classification (High/Moderate/No Risk)

### Why This Matters
Alaska's remote wilderness makes traditional wildfire detection slow. Satellite data provides **real-time, high-resolution insights** into:
- Vegetation health & fuel dryness
- Soil moisture & burn severity
- Active fire detection (thermal anomalies)
- Pre/post-fire comparisons

---

## 🎯 MVP Scope (Phase 1) - CURRENT PHASE

### ✅ Completed
- [x] GEE data export: Sentinel-2 10m patches for Alaska study area
- [x] Patch extraction pipeline (64x64 pixels)
- [x] Data integrity checks (NaN removal)
- [x] Metadata CSV with burn labels
- [x] Unit tests (10 test cases in `test_cases.ipynb`)

### 🔴 Current Blocker
**Softmax Collapse** - Model predicts 0.503 probability for ALL samples
- Loss stuck at 0.693 (random guessing baseline)
- Extreme class imbalance (1.7% positive class)
- Class weight scaling bug (removing `* 0.3` factor)

### 📋 To-Do (This Phase)
- [ ] **Fix training pipeline** (remove weight scaling, improve loss function)
- [ ] **Validate on test set** (Recall > 0.3, F1 > 0.2 on burn class)
- [ ] **Baseline CNN model** working end-to-end
- [ ] **Create diagnostic plots** (loss curves, confusion matrix, probability distribution)
- [ ] **Document findings** (what works, what doesn't)

### ⏱️ Estimated Time: 40-50 hours

---

## 🚀 Phase 2: Multi-Modal Fusion (Post-MVP)

### Objectives
Integrate SAR and weather data for better predictions

### Deliverables
- [ ] Extract Sentinel-1 SAR features
  - Vegetation Water Content (VWC)
  - Soil moisture index
  - Terrain backscatter
  
- [ ] Integrate weather time-series
  - Load ERA5 climate reanalysis
  - Normalize temperature, humidity, wind
  - Create temporal windows (7-day, 14-day, 30-day)
  
- [ ] Build CNN-LSTM hybrid model
  - CNN processes multi-band satellite images
  - LSTM processes weather time-series
  - Concatenate features before classification
  
- [ ] 3-class classification
  - High Risk (burn probability > 0.7)
  - Moderate Risk (0.3 < prob < 0.7)
  - No Risk (prob < 0.3)

### ⏱️ Estimated Time: 80-100 hours

---

## 📊 Phase 3: Production & Visualization (Post-MVP)

### Objectives
Deploy model and create web dashboard

### Deliverables
- [ ] Model export (SavedModel format for TensorFlow Serving)
- [ ] REST API
  - Input: GeoTIFF satellite image + location
  - Output: Risk map + probabilities
  
- [ ] Web-based GIS dashboard
  - Display Alaska map with fire risk overlay
  - Interactive region selection
  - Historical fire comparison
  - Real-time predictions
  
- [ ] Performance report
  - Accuracy, Precision, Recall, F1 metrics
  - Confusion matrix & ROC curves
  - Geographic performance analysis
  - Failure case analysis

### ⏱️ Estimated Time: 80-100 hours

---

## 📂 Project Structure

```
Wildlife-prediction-try02/
├── data/
│   ├── raw/
│   │   ├── s2_2021_06_input_10m.tif          # Sentinel-2 optical data
│   │   ├── s1_2021_06_input_10m.tif          # Sentinel-1 SAR data (Phase 2)
│   │   └── weather_data.csv                   # ERA5 weather (Phase 2)
│   ├── processed/
│   │   ├── patch_metadata.csv                # Patch labels & coordinates
│   │   └── patches/                          # Extracted 64x64 patches
│   └── splits/
│       ├── train_indices.npy
│       ├── val_indices.npy
│       └── test_indices.npy
│
├── src/
│   ├── data/
│   │   ├── loader.py                        # Data loading & patch extraction
│   │   ├── preprocessing.py                 # Normalization, augmentation
│   │   └── validation.py                    # Data integrity checks
│   │
│   ├── models/
│   │   ├── cnn_baseline.py                  # Phase 1: Simple CNN
│   │   ├── cnn_advanced.py                  # Enhanced CNN with residuals
│   │   ├── cnn_lstm_hybrid.py               # Phase 2: CNN-LSTM fusion
│   │   └── losses.py                        # Custom loss functions
│   │
│   ├── training/
│   │   ├── train.py                         # Training loop
│   │   ├── callbacks.py                     # Custom callbacks
│   │   └── metrics.py                       # Evaluation metrics
│   │
│   └── inference/
│       ├── predictor.py                     # Model inference
│       └── postprocess.py                   # Risk mapping
│
├── scripts/
│   ├── export_data_gee.js                   # GEE script for data export
│   ├── preprocess.py                        # Data preprocessing
│   ├── train_model.py                       # Training script
│   ├── evaluate.py                          # Model evaluation
│   ├── main.ipynb                           # Colab training notebook (Phase 1)
│   └── test_cases.ipynb                     # 10 unit tests
│
├── configs/
│   ├── phase1_mvp.yaml                      # Phase 1 configuration
│   ├── phase2_fusion.yaml                   # Phase 2 configuration
│   └── phase3_production.yaml               # Phase 3 configuration
│
├── notebooks/
│   ├── 01_eda.ipynb                         # Exploratory data analysis
│   ├── 02_training.ipynb                    # Training experiments
│   ├── 03_evaluation.ipynb                  # Model evaluation
│   └── 04_dashboard_demo.ipynb              # Dashboard prototype (Phase 3)
│
├── outputs/
│   ├── models/
│   │   ├── phase1_baseline.h5
│   │   ├── phase2_cnn_lstm.h5
│   │   └── phase3_production.pb
│   ├── results/
│   │   ├── confusion_matrices/
│   │   ├── probability_distributions/
│   │   └── geographic_analysis/
│   └── reports/
│       ├── phase1_results.md
│       ├── phase2_results.md
│       └── phase3_final_report.md
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_inference.py
│
├── requirements.txt
├── setup.py
├── README.md
└── PROJECT_ROADMAP.md                      # This file
```

---

## 🔧 Technical Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| **Satellite Data** | Rasterio, GDAL | GeoTIFF handling |
| **ML Framework** | TensorFlow 2.x | TPU support, Keras API |
| **Data Processing** | NumPy, Pandas | Fast numerical ops |
| **GEE Export** | Google Earth Engine JS | Free satellite data |
| **Geospatial** | Geopandas, Shapely | Coordinate mapping |
| **Visualization** | Matplotlib, Folium | Maps & plots |
| **Web Framework** | Flask/FastAPI | REST API (Phase 3) |
| **Dashboard** | Streamlit/Dash | Web UI (Phase 3) |

---

## 📊 Data Specifications

### Sentinel-2 Optical (Phase 1)
```
Resolution: 10m
Bands: B2, B3, B4, B8 (RGB + NIR)
Temporal: June 2021 (pre-fire baseline)
Region: Alaska study area
Format: GeoTIFF (32-bit float)
Patch Size: 64x64 pixels
Normalization: Divide by 10,000 (Sentinel-2 SR range)
```

### Sentinel-1 SAR (Phase 2)
```
Resolution: 10m
Polarization: VV, VH (vertical-vertical, vertical-horizontal)
Temporal: Monthly composite
Extraction: VWC, backscatter ratio, moisture index
```

### Weather Data (Phase 2)
```
Source: ERA5 Climate Reanalysis
Variables: Temperature, Humidity, Wind Speed, Precipitation
Temporal: Daily (aggregated to weekly/monthly)
Spatial: 0.25° resolution (resampled to 10m)
```

### Labels
```
Source: Alaska Fire Service (AFS) historical records
Definition: Binary (Burn/No-Burn) in 2021-2022 fire season
Class Distribution: 
  - No-Burn: 29,393 patches (98.3%)
  - Burn: 511 patches (1.7%)
Challenge: EXTREME IMBALANCE
```

---

## 🎯 Phase 1 Goals & Metrics

### Primary Goal
**Get baseline CNN working on Sentinel-2 data with proper class weighting**

### Success Criteria
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Loss** | < 0.4 | 0.6954 (stuck) | 🔴 Blocked |
| **Accuracy** | > 0.95 | 0.983 (baseline) | ✅ Trivial |
| **Recall (Burn)** | > 0.30 | 0.0 | 🔴 Blocked |
| **F1 (Burn)** | > 0.25 | 0.0 | 🔴 Blocked |
| **Precision (Burn)** | > 0.20 | undefined | 🔴 Blocked |

### Key Blockers to Fix
1. **Class weight scaling** - Remove `weight_1 *= 0.3` 
2. **Loss function** - Switch to `categorical_crossentropy` or `focal_loss`
3. **Model capacity** - May need deeper architecture
4. **Data augmentation** - Rotation, flip for Burn patches
5. **Sampling strategy** - Oversampling burn class

---

## 📈 Phase 1 Implementation Plan

### Week 1: Fix & Validate (10-15 hours)
- [ ] Remove class weight scaling bug
- [ ] Implement better loss function
- [ ] Add early stopping & LR scheduling
- [ ] Run 20+ epochs and verify convergence
- [ ] Generate diagnostic plots

**Deliverable**: Working baseline with Recall > 0.3 on Burn class

---

### Week 2: Experiment & Optimize (15-20 hours)
- [ ] Try data augmentation (rotation, flips, small shifts)
- [ ] Experiment with model architectures (deeper CNN, different filters)
- [ ] Implement focal loss for better imbalance handling
- [ ] Cross-validation on hold-out test set
- [ ] Document all experiments

**Deliverable**: Best model + experiment log

---

### Week 3: Documentation & Testing (10-15 hours)
- [ ] Create comprehensive diagnostic report
- [ ] Write unit tests (data loading, preprocessing, model)
- [ ] Validate on multiple geographic regions
- [ ] Create confusion matrix & probability distribution plots
- [ ] Finalize Phase 1 submission

**Deliverable**: Complete MVP with documentation

---

## 🐛 Known Issues & Solutions

### Issue 1: Softmax Collapse (CURRENT)
**Problem**: Model outputs 0.503 for all samples  
**Root Cause**: `weight_1 *= 0.3` reduces burn class weight by 70%  
**Solution**: Remove scaling, use proper class weights + better loss  
**Status**: 🔴 Needs immediate fix

### Issue 2: Extreme Class Imbalance
**Problem**: 1.7% positive class  
**Solutions**:
- Class weights (already attempted, buggy)
- Focal loss (recommended)
- Oversampling burn patches
- Threshold tuning on probabilities

### Issue 3: NaN Patches
**Problem**: GEE data has missing values  
**Solution**: Already implemented - skip NaN patches  
**Status**: ✅ Fixed

### Issue 4: Only 100 Patches Loading
**Problem**: Loop limitation  
**Solution**: Already fixed - now loads all patches  
**Status**: ✅ Fixed

---

## 📚 Resources & References

### Satellite Data
- [Sentinel-2 Bands & Resolutions](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/radiometric)
- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [Rasterio Tutorial](https://rasterio.readthedocs.io/)

### Deep Learning for Imbalanced Data
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002) - Lin et al., 2017
- [Class Weighting in Keras](https://keras.io/api/models/model_training_apis/#fit-method)
- [Imbalanced Learning Review](https://imbalanced-learn.org/)

### Wildfire Prediction
- [LSTM for Time Series](https://www.tensorflow.org/guide/keras/rnn)
- [CNN-LSTM for Video Classification](https://arxiv.org/abs/1411.4280)
- [Satellite-Based Fire Detection](https://scholar.google.com/scholar?q=satellite+wildfire+detection+CNN)

### Tools & Libraries
- [TensorFlow 2.x](https://www.tensorflow.org/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation/)
- [Geopandas](https://geopandas.org/)
- [Folium Maps](https://python-visualization.github.io/folium/)

---

## 🤝 Team & Collaboration

### Mentors
- **Yali Wang** (ywang35@alaska.edu)
- **Arghya Kusum Das** (akdas@alaska.edu)

### Discussion Forum
https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction/discussions

### GitHub Repo
https://github.com/YaliWang2019/AK-Satellite-Imagery-Wildfire-Prediction

---

## 📋 Checklist for MVP Completion

### Phase 1: MVP (This Phase)
- [ ] Training pipeline fixed & validated
- [ ] Model achieves Recall > 0.3 on Burn class
- [ ] 10 unit tests passing
- [ ] Diagnostic plots & analysis complete
- [ ] README with results
- [ ] Code documented & clean
- [ ] Ready for mentor review

### Phase 2: Multi-Modal (Stretch Goal)
- [ ] SAR data integrated
- [ ] Weather time-series loaded
- [ ] CNN-LSTM model implemented
- [ ] 3-class classification working

### Phase 3: Production (If Time)
- [ ] Model exported & versioned
- [ ] REST API created
- [ ] Dashboard prototype
- [ ] Final report with geographic analysis

---

## 📞 Quick Reference

### To Fix Training NOW:
1. Edit `train_model.py` line 211: **Remove** `weight_1 *= 0.3`
2. Change `loss='sparse_categorical_crossentropy'` → `'categorical_crossentropy'`
3. Add one-hot encoding before `model.fit()`
4. Add early stopping & LR reduction callbacks
5. Run with validation split

### To Run MVP:
```bash
cd scripts/
python preprocess.py              # Prepare patches
python train_model.py             # Train model
```

### To Test:
```bash
cd scripts/
jupyter notebook test_cases.ipynb # Run 10 unit tests
```

---

## 🎓 Learning Goals

By completing this MVP, you'll learn:
- ✅ Satellite data processing with rasterio/GDAL
- ✅ Handling extreme class imbalance in ML
- ✅ CNN architecture design & training
- ✅ Loss functions & their trade-offs
- ✅ Model debugging & diagnostics
- ✅ Geospatial data fundamentals
- ✅ GPU/TPU optimization (Colab)

---

## 📅 Estimated Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: MVP | 40-50h | Week 1 | Week 3 |
| Phase 2: Fusion | 80-100h | Week 4 | Week 9 |
| Phase 3: Production | 80-100h | Week 10 | Week 16 |
| **Total** | **200-250h** | - | - |

**Buffer**: 100-150 hours for debugging, optimization, documentation

---

**Last Updated**: December 16, 2024  
**Status**: Phase 1 - MVP in development, awaiting training fix  
**Next Steps**: Fix softmax collapse, validate on test set, proceed to Phase 2
