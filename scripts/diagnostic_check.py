
import pandas as pd
import numpy as np
import rasterio
     
METADATA_CSV = r"G:\UAAS\Wildlife-prediction-try02\data\patch_metadata.csv"        
BURN_TIF = r"G:\UAAS\Wildlife-prediction-try02\data\raw\burn_2021_Q3_label_10m.tif"
     
df = pd.read_csv(METADATA_CSV)
print("=== LABEL DISTRIBUTION ===")
print(df['burn_label'].value_counts())
print(f"\nTotal patches: {len(df)}")
print(f"Patches with burn_label=1: {(df['burn_label']==1).sum()}")
     
     # Check if burn TIF actually has data
with rasterio.open(BURN_TIF) as src:
    burn_array = src.read(1)
    print(f"\n=== BURN TIF STATS ===")
    print(f"Min pixel value: {burn_array.min()}")
    print(f"Max pixel value: {burn_array.max()}")
    print(f"Max pixel value: {burn_array.max()}")
    print(f"Pixels > 0: {np.sum(burn_array > 0)}")
    print(f"Unique values: {np.unique(burn_array)[:20]}")