"""
ERA5 Weather Analysis for Alaska Wildfire Prediction - Phase 2
Uses Google Earth Engine for fast server-side processing (vs slow CDS API)

Key Innovation: VPD (Vapor Pressure Deficit) as primary fire driver
"""

import ee
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import pearsonr

# --- CONFIGURATION ---
METADATA_PATH = "data/patch_metadata.csv"
OUTPUT_DIR = "results/phase2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Earth Engine
GEE_PROJECT = "alaska-dataset"  # Your GEE project ID

try:
    # Initialize with project ID
    ee.Initialize(project=GEE_PROJECT)
    print(f"✓ Earth Engine initialized with project: {GEE_PROJECT}")
except Exception as e:
    # If that fails, authenticate first
    print("Authenticating GEE...")
    ee.Authenticate()
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"✓ Authenticated and initialized with project: {GEE_PROJECT}")
    except Exception as e2:
        print(f"⚠ Error: {e2}")
        print(f"\nMake sure your project '{GEE_PROJECT}' exists at:")
        print("https://code.earthengine.google.com")
        exit(1)


def calculate_vpd(temp_c, dewpoint_c):
    """
    Calculate Vapor Pressure Deficit (VPD) in kPa.

    VPD = "Atmospheric Thirst" - the drying power of air.
    High VPD → Air sucks moisture from vegetation → Fuel dries → Fire risk ↑

    This is often MORE important than temperature alone because:
    - 30°C + high humidity = low fire risk (VPD low)
    - 25°C + low humidity = high fire risk (VPD high)

    Used in Canadian Fire Weather Index and many operational fire models.

    Args:
        temp_c: Air temperature (°C)
        dewpoint_c: Dewpoint temperature (°C)

    Returns:
        VPD in kPa
    """
    # Saturation Vapor Pressure (Tetens formula)
    es = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    # Actual Vapor Pressure
    ea = 0.61078 * np.exp((17.27 * dewpoint_c) / (dewpoint_c + 237.3))
    return es - ea


def check_accumulation():
    """Debug function to check if precipitation is accumulated or hourly rate"""
    print("\n--- DEBUG: CHECKING PRECIPITATION ACCUMULATION ---")
    point = ee.Geometry.Point([-150.1348, 64.7902])  # Sample location
    start = ee.Date("2021-06-01T00:00:00")
    end = start.advance(6, "hour")

    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(start, end)
        .filterBounds(point)
    )
    values = col.select("total_precipitation").getRegion(point, 1000).getInfo()

    print("Precipitation values (Meters) for 6 consecutive hours:")
    header = values[0]
    for row in values[1:]:
        print(f"  {row[0]} {pd.to_datetime(row[3], unit='ms')}: {row[4]}")

    vals = [row[4] for row in values[1:]]
    if all(x <= y for x, y in zip(vals, vals[1:])) and sum(vals) > 0:
        print(">> DATA APPEARS ACCUMULATED (Monotonic Increase)")
    else:
        print(">> DATA APPEARS DE-ACCUMULATED (Fluctuating)")
    print("--------------------------------------------------\n")


def fetch_era5_data(patches_df, days_before=30):
    """
    Fetches weather data for 30-day window BEFORE each fire.

    Hybrid Strategy:
    1. ERA5_LAND/HOURLY (11km) -> Temperature, Wind, Soil Moisture
    2. NASA GPM IMERG V06 (10km) -> Precipitation (Corrected Source)
       * ERA5 was overestimating precip by ~27x (artifacts).
       * GPM provides accurate satellite-based precip rates.

    CRITICAL: Prevents data leakage by only using pre-fire weather.
    """
    print(f"\nFetching Weather Data (ERA5-Land + GPM IMERG)...")
    print(f"Time window: {days_before} days before each fire ignition")

    # 1. ERA5 for State Variables
    era5_hourly = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")

    # 2. GPM for Precipitation (Gold Standard)
    gpm_col = ee.ImageCollection("NASA/GPM_L3/IMERG_V06")

    results = []
    failed = 0

    # Use enumerate to get a clean 0-based counter for progress
    for i, (index, row) in enumerate(patches_df.iterrows()):
        patch_id = row.get("patch_id", index)

        try:
            # --- DYNAMIC TIME WINDOW (prevents data leakage) ---
            fire_date = row["date"]
            end_date = ee.Date(fire_date)
            start_date = end_date.advance(-days_before, "day")

            # Create point geometry from fire location
            point = ee.Geometry.Point([row["longitude"], row["latitude"]])

            # --- PART A: ERA5 Variables (Temp, Wind, Soil) ---
            hourly_filtered = era5_hourly.filterDate(start_date, end_date).filterBounds(
                point
            )

            def add_wind(image):
                u = image.select("u_component_of_wind_10m")
                v = image.select("v_component_of_wind_10m")
                wind_speed = (u.pow(2).add(v.pow(2))).sqrt().rename("wind_speed")
                return image.addBands(wind_speed)

            # Reduce ERA5 -> Mean Stats
            era5_stats = (
                hourly_filtered.map(add_wind)
                .select(
                    [
                        "temperature_2m",
                        "dewpoint_temperature_2m",
                        "volumetric_soil_water_layer_1",
                        "wind_speed",
                    ]
                )
                .mean()
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=11132,
                    maxPixels=1e9,
                )
                .getInfo()
            )

            # --- PART B: GPM Precipitation ---
            # GPM is 30-min resolution. Band 'precipitationCal' is mm/hr.
            # Total mm = sum(rate * 0.5hr)
            # Efficient way: Mean Rate (mm/hr) * Total Hours (24 * 30)

            # Filter GPM
            gpm_filtered = gpm_col.filterDate(start_date, end_date).filterBounds(point)

            # Get Mean Precipitation Rate (mm/hr) over the 30 days
            gpm_stats = (
                gpm_filtered.select("precipitationCal")
                .mean()
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=11132,
                    maxPixels=1e9,
                )
                .getInfo()
            )

            if era5_stats and gpm_stats and "temperature_2m" in era5_stats:
                # Unit conversions
                temp_c = era5_stats["temperature_2m"] - 273.15
                dew_c = era5_stats["dewpoint_temperature_2m"] - 273.15

                # Precipitation Calculation
                # Mean Rate (mm/hr) * 24 hours * 30 days
                mean_precip_rate = gpm_stats.get("precipitationCal", 0)  # mm/hr
                if mean_precip_rate is None:
                    mean_precip_rate = 0
                total_precip_mm = mean_precip_rate * 24 * 30

                # Calculate VPD
                vpd = calculate_vpd(temp_c, dew_c)

                results.append(
                    {
                        "patch_id": patch_id,
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "burn_label": row["burn_label"],
                        "mean_temp_30d": temp_c,
                        "total_precip_30d": total_precip_mm,
                        "mean_wind_speed_30d": era5_stats["wind_speed"],
                        "mean_soil_moisture_30d": era5_stats[
                            "volumetric_soil_water_layer_1"
                        ],
                        "mean_vpd_30d": vpd,
                    }
                )
            else:
                print(f"  Warning on patch {patch_id}: Data missing.")
                failed += 1

        except Exception as e:
            print(f"  Error on patch {patch_id}: {e}")
            failed += 1
            if failed > 10 and len(results) == 0:
                print("  ⚠ Too many errors, checking first error detail...")
                raise e
            continue

        # Progress indicator (every 50 patches)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(patches_df)} patches...")

    print(f"\n✓ Completed: {len(results)} successful, {failed} failed")
    return pd.DataFrame(results)


def generate_visualizations(weather_df):
    """
    Generate correlation analysis and distribution plots.
    """
    print("\nGenerating visualizations...")

    weather_cols = [
        "mean_temp_30d",
        "total_precip_30d",
        "mean_wind_speed_30d",
        "mean_soil_moisture_30d",
        "mean_vpd_30d",
    ]

    # 1. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = weather_df[weather_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        square=True,
        linewidths=1,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        "ERA5 Weather Variable Correlations\n(Pre-Fire 30 Days)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: correlation_heatmap.png")

    # 2. Variable Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    col_labels = {
        "mean_temp_30d": "Mean Temperature (°C)",
        "total_precip_30d": "Total Precipitation (mm)",
        "mean_wind_speed_30d": "Mean Wind Speed (m/s)",
        "mean_soil_moisture_30d": "Mean Soil Moisture (m³/m³)",
        "mean_vpd_30d": "Mean VPD (kPa)",
    }

    for idx, col in enumerate(weather_cols):
        if idx < len(axes):
            axes[idx].hist(
                weather_df[col],
                bins=20,
                edgecolor="black",
                alpha=0.7,
                color="steelblue",
            )
            axes[idx].set_title(col_labels[col], fontweight="bold")
            axes[idx].set_xlabel("Value")
            axes[idx].set_ylabel("Frequency")
            axes[idx].grid(alpha=0.3)

    # Hide unused subplot
    if len(weather_cols) < len(axes):
        axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/variable_distributions.png", dpi=300, bbox_inches="tight"
    )
    print(f"  ✓ Saved: variable_distributions.png")

    # 3. VPD vs Fire Occurrence (if we had no-fire patches for comparison)
    # For now, just show VPD distribution for fire patches
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        weather_df["mean_vpd_30d"], bins=25, edgecolor="black", alpha=0.7, color="coral"
    )
    ax.axvline(
        weather_df["mean_vpd_30d"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {weather_df['mean_vpd_30d'].mean():.2f} kPa",
    )
    ax.set_title(
        "Vapor Pressure Deficit Distribution (Fire Patches)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("VPD (kPa)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/vpd_distribution.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: vpd_distribution.png")


def print_statistical_summary(weather_df):
    """
    Print summary statistics and correlation analysis.
    """
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    weather_cols = [
        "mean_temp_30d",
        "total_precip_30d",
        "mean_wind_speed_30d",
        "mean_soil_moisture_30d",
        "mean_vpd_30d",
    ]

    print("\nDescriptive Statistics (30-day pre-fire conditions):")
    print(weather_df[weather_cols].describe().round(3))

    print("\n" + "-" * 70)
    print("KEY FINDINGS:")
    print("-" * 70)

    # Mean VPD (primary fire driver)
    mean_vpd = weather_df["mean_vpd_30d"].mean()
    print(f"\n1. Mean VPD (Atmospheric Drying Power): {mean_vpd:.2f} kPa")
    if mean_vpd > 1.5:
        print("   → HIGH fire risk conditions (VPD > 1.5 kPa)")
    elif mean_vpd > 1.0:
        print("   → MODERATE fire risk conditions")
    else:
        print("   → LOW fire risk conditions")

    # Precipitation (inverse relationship expected)
    mean_precip = weather_df["total_precip_30d"].mean()
    print(f"\n2. Mean 30-day Precipitation: {mean_precip:.1f} mm")
    if mean_precip < 25:
        print("   → DROUGHT conditions (< 25mm/month)")
    elif mean_precip < 50:
        print("   → DRY conditions")
    else:
        print("   → NORMAL/WET conditions")

    # Temperature
    mean_temp = weather_df["mean_temp_30d"].mean()
    print(f"\n3. Mean Temperature: {mean_temp:.1f}°C")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2: ERA5 Weather Analysis (GEE-based)")
    print("=" * 70)

    # Debug: Check ERA5 accumulation logic
    # try:
    #    check_accumulation()
    # except Exception as e:
    #    print(f"Debug check failed: {e}")

    # Load Phase 1 metadata
    if not os.path.exists(METADATA_PATH):
        print(f"\n✗ Error: {METADATA_PATH} not found!")
        print("  Make sure you've run Phase 1 preprocessing first.")
        exit(1)

    print(f"\n[1/5] Loading fire patch metadata...")
    meta_df = pd.read_csv(METADATA_PATH)
    print(f"  Loaded {len(meta_df)} total patches")

    # FIX: Map Phase 1 column names to what script expects
    print(f"\n[2/5] Preparing data...")
    if "center_x" in meta_df.columns:
        meta_df.rename(
            columns={
                "center_x": "longitude",
                "center_y": "latitude",
                "fire_outcome_date": "date",
            },
            inplace=True,
        )
        print("  ✓ Mapped Phase 1 columns (center_x/center_y → longitude/latitude)")

    # FIX: Filter for BURN patches only (burn_label == 1)
    burn_patches = meta_df[meta_df["burn_label"] == 1].copy()
    print(f"  ✓ Filtered to {len(burn_patches)} burn patches (burn_label == 1)")

    if len(burn_patches) == 0:
        print("\n✗ Error: No burn patches found! Check your metadata.")
        exit(1)

    # Fetch ERA5 weather data via GEE
    print(f"\n[3/5] Fetching ERA5 weather data via Google Earth Engine...")
    print("  (This may take 2-5 minutes depending on patch count)")
    weather_df = fetch_era5_data(burn_patches)

    if len(weather_df) == 0:
        print("\n✗ Error: No weather data retrieved. Check GEE authentication.")
        exit(1)

    # Save results
    print(f"\n[4/5] Saving results...")
    weather_df.to_csv(f"{OUTPUT_DIR}/era5_weather_correlations.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR}/era5_weather_correlations.csv")

    # Generate visualizations
    print(f"\n[5/5] Generating visualizations...")
    generate_visualizations(weather_df)

    # Print statistical summary
    print_statistical_summary(weather_df)

    print("\n" + "=" * 70)
    print("✓ PHASE 2 ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("  - era5_weather_correlations.csv")
    print("  - correlation_heatmap.png")
    print("  - variable_distributions.png")
    print("  - vpd_distribution.png")
    print("\nNext: Fill in docs/phase2-weather-analysis.md with these findings!")
