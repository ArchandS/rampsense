import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = Path(
    r"c:\Users\archa\Downloads\rampsense dataset\wind dataset\Penmanshiel_WT15_2017_2018_extracted.csv"
)

output_file = Path(
    r"c:\Users\archa\Downloads\rampsense dataset\wind dataset\Penmanshiel_WT15_2017_2018_cleaned.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING WIND DATA")
print("=" * 60)

df = pd.read_csv(input_file)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 3. CONVERT DATA TYPES
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["wind_speed"] = pd.to_numeric(
    df["wind_speed"],
    errors="coerce"
)

df["wind_direction"] = pd.to_numeric(
    df["wind_direction"],
    errors="coerce"
)

df["power"] = pd.to_numeric(
    df["power"],
    errors="coerce"
)


# ============================================================
# 4. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# 5. REMOVE INVALID TIMESTAMPS
# ============================================================

invalid_timestamp = df["timestamp"].isna().sum()

print("\nInvalid timestamps:", invalid_timestamp)

if invalid_timestamp > 0:
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)


# ============================================================
# 6. CHECK DUPLICATE TIMESTAMPS
# ============================================================

duplicates = df["timestamp"].duplicated().sum()

print("Duplicate timestamps:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates(
        subset="timestamp",
        keep="first"
    ).reset_index(drop=True)


# ============================================================
# 7. CHECK TIME INTERVALS
# ============================================================

time_difference = df["timestamp"].diff().dropna()

print("\nTime interval statistics:")
print(time_difference.value_counts().head(10))

expected_interval = pd.Timedelta(minutes=10)

wrong_intervals = (
    time_difference != expected_interval
).sum()

print(
    "Intervals different from 10 minutes:",
    wrong_intervals
)


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")

print(
    df[
        ["timestamp", "wind_speed",
         "wind_direction", "power"]
    ].isna().sum()
)


# ============================================================
# 9. CHECK PHYSICAL VALUES
# ============================================================

print("\nPhysical-value checks:")

negative_wind_speed = (
    df["wind_speed"] < 0
).sum()

invalid_direction = (
    (df["wind_direction"] < 0) |
    (df["wind_direction"] > 360)
).sum()

negative_power = (
    df["power"] < 0
).sum()

very_high_power = (
    df["power"] > 2500
).sum()


print("Negative wind speed:", negative_wind_speed)
print("Invalid wind direction:", invalid_direction)
print("Negative power:", negative_power)
print("Power > 2500 kW:", very_high_power)


# ============================================================
# 10. REPLACE CLEARLY INVALID SENSOR VALUES WITH NaN
# ============================================================

# Wind speed cannot physically be negative.
df.loc[
    df["wind_speed"] < 0,
    "wind_speed"
] = np.nan


# Wind direction should be between 0 and 360 degrees.
df.loc[
    (df["wind_direction"] < 0) |
    (df["wind_direction"] > 360),
    "wind_direction"
] = np.nan


# IMPORTANT:
# We do NOT automatically remove negative power.
#
# Negative values can be useful information and should be
# investigated before deciding how to handle them.
#
# We also do NOT clip power at 2050 kW.


# ============================================================
# 11. REPORT DATA AFTER BASIC CLEANING
# ============================================================

print("\nMissing values after basic validity checks:")

print(
    df[
        ["timestamp", "wind_speed",
         "wind_direction", "power"]
    ].isna().sum()
)


# ============================================================
# 12. IDENTIFY ROWS WITH MISSING CORE DATA
# ============================================================

missing_core = df[
    ["wind_speed", "wind_direction", "power"]
].isna().any(axis=1)

print(
    "\nRows with missing core measurements:",
    missing_core.sum()
)


# ============================================================
# 13. DO NOT IMPUTE YET
# ============================================================

print("\nNo interpolation/imputation has been performed.")
print("No power clipping has been performed.")
print("Negative power values have been retained.")


# ============================================================
# 14. SAVE CLEANED DATA
# ============================================================

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print(f"Final rows: {len(df):,}")
print(f"Output file:")
print(output_file)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\nFinal dataset summary:")

print(df.describe(include="all"))

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())
