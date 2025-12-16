# Preprocessing script
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
import os

# --- Configuration 

DATA_DIR = "data/raw/"
SENTINEL_TIF = os.path.join(DATA_DIR, "s2_2021_06_input_10m.tif")
BURN_TIF = os.path.join(DATA_DIR, "burn_2021_Q3_label_10m.tif")
OUTPUT_CSV = "data/patch_metadata.csv"
PATCH_SIZE = 64  # Pixels: 64x64 patches (can be 256 or 512 for ML training)

INPUT_DATE_START = '2021-06-01'
FIRE_DATE_START = '2021-07-01'


def verify_and_extract_patches():
    """Reads aligned GeoTIFFs, verifies alignment, extracts patches, and creates metadata CSV."""
    print(f"--- Starting Patch Extraction (Patch Size: {PATCH_SIZE}x{PATCH_SIZE}) ---")
    
    try:
        sentinel_src = rasterio.open(SENTINEL_TIF)
        burn_src = rasterio.open(BURN_TIF)
    except rasterio.RasterioIOError:
        print(f"Error: Could not open one or both files. Check paths: {SENTINEL_TIF} and {BURN_TIF}")
        return

    # CRITICAL ALIGNMENT ASSURANCES [cite: 649]
    assert sentinel_src.crs == burn_src.crs, "FATAL ERROR: CRS mismatch between Sentinel and Burn mask! Alignment failed in GEE."
    assert sentinel_src.width == burn_src.width and sentinel_src.height == burn_src.height, "FATAL ERROR: Spatial dimension mismatch! Alignment failed in GEE."

    print("✅ ALIGNMENT VERIFIED: CRS and Dimensions Match Perfectly.")
    print(f"Image dimensions (W x H): {sentinel_src.width} x {sentinel_src.height}")
    print(f"Coordinate Reference System: {sentinel_src.crs}")
    
    # Checking  if a folder for storing patches is needed
    PATCH_FOLDER = "data/patches"
    os.makedirs(PATCH_FOLDER, exist_ok=True)


    # 2. Extracting  Patches and Metadata
    metadata_rows = []
    img_width, img_height = sentinel_src.width, sentinel_src.height
    patch_id = 0
    
    # Iterating 
    for y in range(0, img_height, PATCH_SIZE):
        for x in range(0, img_width, PATCH_SIZE):
            
            if x + PATCH_SIZE > img_width or y + PATCH_SIZE > img_height:
                continue 

            patch_id += 1
            window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
            
            # Reading the Sentinel image patch (all bands) 
            sentinel_patch = sentinel_src.read(window=window)
            
            # Reading  burn-mask patch (single band 1) 
            burn_patch = burn_src.read(1, window=window) 
            
            # Deriving Label: 1 if ANY pixel burned (BurnDate > 0) 
            burned = 1 if np.any(burn_patch > 0) else 0
            
            # Calculating patch center coordinates for metadata 
            col_center = x + PATCH_SIZE / 2
            row_center = y + PATCH_SIZE / 2
            # Transforming pixel index to map coordinates (x, y)
            patch_center_lon, patch_center_lat = sentinel_src.transform * (col_center, row_center)
            
            
            # 4. Store Metadata [cite: 740, 741]
            metadata_rows.append({
                "patch_id": patch_id,
                "center_x": patch_center_lon,
                "center_y": patch_center_lat,
                "input_date": INPUT_DATE_START,
                "fire_outcome_date": FIRE_DATE_START,
                "burn_label": burned
            })

    sentinel_src.close()
    burn_src.close()

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nSuccessfully extracted {len(metadata_df)} patches.")
    print(f"Metadata saved to: {OUTPUT_CSV}")
    print("--- Patch Extraction Complete ---")

if __name__ == "__main__":
    verify_and_extract_patches()