# Phase 2: ERA5 Weather Data Analysis Setup

## Prerequisites

### 1. Install Required Packages

```bash
pip install cdsapi xarray netCDF4 scipy
```

### 2. Set Up Copernicus CDS API

1. **Create account**: https://cds.climate.copernicus.eu/user/register
2. **Get API key**: https://cds.climate.copernicus.eu/api-how-to
3. **Create `.cdsapirc` file** in your home directory:

```
# Windows: C:\Users\<username>\.cdsapirc
# Linux: ~/.cdsapirc

url: https://cds.climate.copernicus.eu/api/v2
key: <YOUR-UID>:<YOUR-API-KEY>
```

## Running the Analysis

```bash
cd g:/UAAS/wildfire-prediction-mvp
python scripts/era5_analysis.py
```

**Expected runtime:** ~30-60 minutes (depending on number of fire patches)

## Outputs

The script will generate:

1. **`results/phase2/era5_fire_correlations.csv`** - Statistical summary
2. **`results/phase2/correlation_matrix.png`** - Variable correlations heatmap
3. **`results/phase2/variable_distributions.png`** - Distribution plots
4. **`results/phase2/temporal_trends.png`** - Pre-fire trend analysis

## ERA5 Variables Analyzed

| Variable | Physical Meaning | Hypothesis |
|----------|------------------|------------|
| `temperature_2m` | Air temperature at 2m | Heat stress on vegetation |
| `relative_humidity` | Moisture in air | Fuel dryness indicator |
| `total_precipitation` | Rainfall | Fuel moisture (inverse) |
| `wind_speed` | 10m wind speed | Fire spread potential |
| `volumetric_soil_water_layer_1` | Soil moisture (top layer) | Vegetation stress |
| `surface_pressure` | Atmospheric pressure | Weather system indicator |
| `dewpoint_temperature` | Dewpoint | Actual moisture content |

## Next Steps

After running the analysis:

1. Review correlation plots
2. Identify top 5 variables with strongest fire associations
3. Write Phase 2 hypothesis document with empirical evidence
4. Post results to GitHub discussion

## Troubleshooting

**Issue:** CDS API timeout  
**Solution:** Reduce `PRE_FIRE_DAYS` from 30 to 14 in script

**Issue:** Missing patches in metadata  
**Solution:** Verify `data/patch_metadata.csv` exists and has `burn_label` column

**Issue:** Memory error  
**Solution:** Process patches in batches (modify `analyze_fire_correlations` function)
