import pandas as pd
import os

# Read all three fire detection CSV files
print("Reading fire detection data files...")

nrt_file = "data/fire_nrt_J2V-C2_686636.csv"
archive_sv_file = "data/fire_archive_SV-C2_686637.csv"
archive_j1v_file = "data/fire_archive_J1V-C2_686635.csv"

df_nrt = pd.read_csv(nrt_file)
df_sv = pd.read_csv(archive_sv_file)
df_j1v = pd.read_csv(archive_j1v_file)

print(f"NRT file: {len(df_nrt)} records")
print(f"SV Archive file: {len(df_sv)} records")
print(f"J1V Archive file: {len(df_j1v)} records")

# Add type column to NRT data if it doesn't exist (archive files have it)
if 'type' not in df_nrt.columns:
    df_nrt['type'] = None

# Combine all dataframes
print("\nCombining all datasets...")
df_combined = pd.concat([df_nrt, df_sv, df_j1v], ignore_index=True)

print(f"Total records before deduplication: {len(df_combined)}")

# Select the most useful columns for wildfire tracking
useful_columns = [
    'latitude',           # Location
    'longitude',          # Location
    'acq_date',          # Date of detection
    'acq_time',          # Time of detection
    'brightness',        # Temperature in Kelvin
    'bright_t31',        # Channel 31 brightness temperature
    'frp',               # Fire Radiative Power (MW) - intensity measure
    'confidence',        # Detection confidence (l=low, n=nominal, h=high)
    'satellite',         # Which satellite detected it
    'instrument',        # VIIRS instrument
    'daynight',          # Day or Night detection
    'scan',              # Scan pixel size
    'track',             # Track pixel size
    'type',              # Fire type (if available)
    'version'            # Data version
]

# Keep only useful columns
df_final = df_combined[useful_columns].copy()

# Sort by date and time for chronological order
print("\nSorting by date and time...")
df_final['datetime'] = pd.to_datetime(
    df_final['acq_date'] + ' ' + df_final['acq_time'].astype(str).str.zfill(4),
    format='%Y-%m-%d %H%M'
)

df_final = df_final.sort_values('datetime')

# Remove exact duplicates (same location, same time)
print("\nRemoving duplicate detections...")
before_dedup = len(df_final)
df_final = df_final.drop_duplicates(
    subset=['latitude', 'longitude', 'acq_date', 'acq_time'],
    keep='first'
)
after_dedup = len(df_final)
print(f"Removed {before_dedup - after_dedup} duplicate detections")

# Reorder columns with datetime first
column_order = ['datetime', 'latitude', 'longitude', 'acq_date', 'acq_time',
                'brightness', 'frp', 'confidence', 'satellite', 'daynight',
                'bright_t31', 'scan', 'track', 'type', 'instrument', 'version']

df_final = df_final[column_order]

# Save the combined dataset
output_file = "combined_fire_detections.csv"
df_final.to_csv(output_file, index=False)

print(f"\n✓ Combined dataset saved to: {output_file}")
print(f"✓ Total unique fire detections: {len(df_final)}")
print(f"✓ Date range: {df_final['acq_date'].min()} to {df_final['acq_date'].max()}")
print(f"\nDataset summary:")
print(f"  - Satellites: {df_final['satellite'].unique()}")
print(f"  - Confidence levels: {df_final['confidence'].unique()}")
print(f"  - Day/Night detections: Day={len(df_final[df_final['daynight']=='D'])}, Night={len(df_final[df_final['daynight']=='N'])}")
print(f"  - Average FRP (Fire Radiative Power): {df_final['frp'].mean():.2f} MW")
print(f"  - Max FRP: {df_final['frp'].max():.2f} MW")
