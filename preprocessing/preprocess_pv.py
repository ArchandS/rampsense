import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. FILE SETTINGS
# ============================================================

INPUT_FILE = r"C:\Users\archa\Downloads\Dataset-SolarTechLab.csv"
OUTPUT_FILE = r"C:\Users\archa\Downloads\Dataset-SolarTechLab1.csv"

# Maximum consecutive missing values that we are willing
# to interpolate.
MAX_GAP = 5   # 5 minutes


# ============================================================
# 2. READ DATA
# ============================================================

print("=" * 70)
print("READING DATA")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    low_memory=False
)

print(f"Rows loaded    : {len(df):,}")
print(f"Columns loaded : {len(df.columns)}")
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nColumn names after cleaning:")
print(df.columns.tolist())


# ============================================================
# 4. PARSE TIME
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING TIMESTAMP")
print("=" * 70)

df["Time"] = pd.to_datetime(
    df["Time"],
    errors="coerce",
    dayfirst=True
)

bad_time = df["Time"].isna().sum()

print(f"Invalid timestamps: {bad_time:,}")

# Remove invalid timestamps
df = df.dropna(subset=["Time"])

# Sort chronologically
df = df.sort_values("Time")

# Remove duplicate timestamps
duplicates = df["Time"].duplicated().sum()

print(f"Duplicate timestamps: {duplicates:,}")

df = df.drop_duplicates(
    subset="Time",
    keep="first"
)

# Make timestamp the index
df = df.set_index("Time")


# ============================================================
# 5. CONVERT MEASUREMENTS TO NUMERIC
# ============================================================

numeric_columns = [
    "PV_Power",
    "T_air",
    "G_h",
    "G_tilt",
    "W_s",
    "W_d"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# 6. REPLACE OBVIOUS SENSOR ERROR VALUES
# ============================================================

print("\n" + "=" * 70)
print("REMOVING SENSOR ERROR CODES")
print("=" * 70)

# Common invalid values observed in this dataset
invalid_values = [
    -999999,
    -99999,
    -9999,
    999999,
    99999,
    9999
]

for col in numeric_columns:

    before = df[col].notna().sum()

    df[col] = df[col].replace(
        invalid_values,
        np.nan
    )

    after = df[col].notna().sum()

    print(
        f"{col:10s}: "
        f"{before-after:,} invalid values replaced"
    )


# ============================================================
# 7. CHECK PHYSICAL VALUES
# ============================================================

print("\n" + "=" * 70)
print("PHYSICAL-VALUE CHECK")
print("=" * 70)

# PV power cannot be negative
negative_pv = (df["PV_Power"] < 0).sum()

print(f"Negative PV power      : {negative_pv:,}")

df.loc[
    df["PV_Power"] < 0,
    "PV_Power"
] = np.nan


# Irradiance cannot be negative
for col in ["G_h", "G_tilt"]:

    negative = (df[col] < 0).sum()

    print(
        f"Negative {col:8s}: {negative:,}"
    )

    df.loc[
        df[col] < 0,
        col
    ] = np.nan


# Wind speed cannot be negative
negative_wind = (df["W_s"] < 0).sum()

print(f"Negative wind speed     : {negative_wind:,}")

df.loc[
    df["W_s"] < 0,
    "W_s"
] = np.nan


# ============================================================
# 8. FLAG EXTREME PV VALUES
# ============================================================

print("\n" + "=" * 70)
print("PV OUTLIER CHECK")
print("=" * 70)

print("PV power statistics BEFORE outlier filtering:")
print(df["PV_Power"].describe())

# The SolarTechLab PV system is a small system.
# Use a conservative upper physical limit.
#
# IMPORTANT:
# We flag extreme values rather than blindly deleting them.

PV_MAX = 1000  # W

extreme_pv = df["PV_Power"] > PV_MAX

print(
    f"\nPV values > {PV_MAX} W: "
    f"{extreme_pv.sum():,}"
)

df.loc[
    extreme_pv,
    "PV_Power"
] = np.nan


# ============================================================
# 9. CREATE DAY/NIGHT INDICATOR
# ============================================================

print("\n" + "=" * 70)
print("DAY/NIGHT CLASSIFICATION")
print("=" * 70)

# Consider irradiance <= 1 W/m² as nighttime.
df["is_day"] = (
    (df["G_h"] > 1) |
    (df["G_tilt"] > 1)
)

print(
    f"Daylight rows : {df['is_day'].sum():,}"
)

print(
    f"Night rows    : {(~df['is_day']).sum():,}"
)


# ============================================================
# 10. HANDLE NIGHTTIME PV MISSING VALUES
# ============================================================

# At night, PV production should be zero.
#
# We only fill PV_Power with zero when BOTH irradiance
# measurements indicate nighttime.

night = (
    (df["G_h"].fillna(0) <= 1) &
    (df["G_tilt"].fillna(0) <= 1)
)

night_missing_pv = (
    night &
    df["PV_Power"].isna()
)

print(
    f"\nNighttime missing PV values: "
    f"{night_missing_pv.sum():,}"
)

df.loc[
    night_missing_pv,
    "PV_Power"
] = 0.0


# ============================================================
# 11. INTERPOLATE SMALL GAPS ONLY
# ============================================================

print("\n" + "=" * 70)
print("INTERPOLATING SMALL GAPS")
print("=" * 70)

# We interpolate only short gaps.
# Large gaps remain NaN.
#
# This is important for ramp forecasting because
# filling long gaps artificially could create fake ramps.

interpolate_columns = [
    "PV_Power",
    "T_air",
    "G_h",
    "G_tilt",
    "W_s"
]

for col in interpolate_columns:

    missing_before = df[col].isna().sum()

    df[col] = (
        df[col]
        .interpolate(
            method="time",
            limit=MAX_GAP,
            limit_direction="both"
        )
    )

    missing_after = df[col].isna().sum()

    filled = missing_before - missing_after

    print(
        f"{col:10s}: "
        f"{filled:,} values interpolated"
    )


# Wind direction is circular, so DO NOT simply interpolate
# it as an ordinary linear variable.
#
# Instead, leave large missing values for now.


# ============================================================
# 12. REMOVE ROWS WITH MISSING TARGET
# ============================================================

print("\n" + "=" * 70)
print("FINAL TARGET CLEANING")
print("=" * 70)

before = len(df)

# PV power is our forecasting target.
#
# We cannot train a supervised forecasting model
# when the target is missing.

df = df.dropna(
    subset=["PV_Power"]
)

after = len(df)

print(
    f"Rows removed because PV_Power is missing: "
    f"{before-after:,}"
)


# ============================================================
# 13. CHECK REMAINING MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("REMAINING MISSING VALUES")
print("=" * 70)

missing_report = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percent": (
        df.isna().mean() * 100
    ).round(3)
})

print(missing_report)


# ============================================================
# 14. TIME CONTINUITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("TIME CONTINUITY CHECK")
print("=" * 70)

time_difference = df.index.to_series().diff()

one_minute = pd.Timedelta(minutes=1)

continuous = (
    time_difference == one_minute
)

print(
    f"1-minute consecutive rows: "
    f"{continuous.sum():,}"
)

print(
    f"Time gaps larger than 1 minute: "
    f"{(~continuous).sum():,}"
)


# ============================================================
# 15. CREATE USEFUL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING TIME FEATURES")
print("=" * 70)

df["hour"] = df.index.hour

df["minute"] = df.index.minute

df["day_of_year"] = df.index.dayofyear

df["month"] = df.index.month

# Cyclic encoding
df["hour_sin"] = np.sin(
    2 * np.pi * (
        df.index.hour * 60 + df.index.minute
    ) / 1440
)

df["hour_cos"] = np.cos(
    2 * np.pi * (
        df.index.hour * 60 + df.index.minute
    ) / 1440
)

df["day_sin"] = np.sin(
    2 * np.pi * df.index.dayofyear / 365.25
)

df["day_cos"] = np.cos(
    2 * np.pi * df.index.dayofyear / 365.25
)


# ============================================================
# 16. SAVE CLEAN DATASET
# ============================================================

print("\n" + "=" * 70)
print("SAVING CLEAN DATASET")
print("=" * 70)

df.to_csv(
    OUTPUT_FILE,
    index=True
)

print(
    f"\nClean dataset saved as:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nFinal rows    : {len(df):,}"
)

print(
    f"Final columns : {len(df.columns)}"
)

print("\nFinal dataset:")
print(df.head())

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
