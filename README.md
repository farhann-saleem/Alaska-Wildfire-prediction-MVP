# Alaska Wildfire Prediction

Project for wildfire prediction using satellite imagery and machine learning.

## Project Structure

```
Alaska-Wildfire-Prediction/
├── configs/
│   └── default_config.yaml  # Project parameters
├── data/
│   └── raw/                 # Raw downloaded GeoTIFFs
│       ├── s2_2021_06_input.tif
│       └── burn_2021_Q3_label.tif
├── scripts/
│   ├── export_data_gee.js   # Google Earth Engine script
│   └── preprocess.py        # Preprocessing script
├── src/
│   └── data_pipeline/
│       ├── __init__.py
│       └── patch_extraction.py
├── .gitignore
└── README.md
```
