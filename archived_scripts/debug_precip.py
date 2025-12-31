import pandas as pd
import ee
import os

# Init GEE
try:
    ee.Initialize(project="alaska-dataset")
except:
    ee.Authenticate()
    ee.Initialize(project="alaska-dataset")


def debug_patch(lat, lon, date_str):
    print(f"\n--- DEBUGGING PATCH: {lat}, {lon} on {date_str} ---")

    # 30 day window
    end_date = ee.Date(date_str)
    start_date = end_date.advance(-30, "day")
    point = ee.Geometry.Point([lon, lat])

    # --- 1. ERA5 (What gave us 673mm) ---
    era5_col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(start_date, end_date)
        .filterBounds(point)
    )
    era5_raw = era5_col.select("total_precipitation").getRegion(point, 11132).getInfo()

    # row[4] is total_precipitation
    if len(era5_raw) > 1:
        era5_vals_m = [row[4] for row in era5_raw[1:]]
        era5_total_mm = sum(era5_vals_m) * 1000
    else:
        era5_total_mm = 0

    print(f"\n[ERA5-Land Hourly]")
    print(f"Total Sum: {era5_total_mm:.2f} mm")

    # --- 2. NASA GPM (Comparison) ---
    # NASA/GPM_L3/IMERG_V06 (30 min resolution, units: mm/hr)
    gpm_col = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
        .filterDate(start_date, end_date)
        .filterBounds(point)
    )
    gpm_n = gpm_col.size().getInfo()
    print(f"\n[NASA GPM IMERG]")
    print(f"Images Found: {gpm_n} (Expected ~1440)")

    if gpm_n > 0:
        # GPM 'precipitationCal' is in mm/hr.
        # GPM is 30-min data.
        # Total mm = sum(rate * 0.5 hours)
        gpm_raw = gpm_col.select("precipitationCal").getRegion(point, 11132).getInfo()
        gpm_vals_rate = [row[4] for row in gpm_raw[1:] if row[4] is not None]

        gpm_total_mm = sum([r * 0.5 for r in gpm_vals_rate])

        print(f"Total Sum: {gpm_total_mm:.2f} mm")

        # Comparison
        print(f"\n>>> DIFFERENCE: {era5_total_mm - gpm_total_mm:.2f} mm")

        if gpm_total_mm < 100 and era5_total_mm > 500:
            print(">>> CONCLUSION: ERA5 is WRONG (Overestimating / Scaling Issue)")
        elif gpm_total_mm > 500:
            print(">>> CONCLUSION: IT REALLY WAS A FLOOD (Both agree)")
        else:
            print(">>> CONCLUSION: Mixed / Unclear")
    else:
        print("No GPM data found!")


def check_metadata_diversity():
    df = pd.read_csv("data/patch_metadata.csv")
    burn_patches = df[df["burn_label"] == 1]
    # Pick the first one to debug
    first = burn_patches.iloc[0]
    debug_patch(first["center_y"], first["center_x"], first["fire_outcome_date"])


if __name__ == "__main__":
    check_metadata_diversity()
