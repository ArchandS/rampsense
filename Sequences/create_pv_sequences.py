import pandas as pd
import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = r"C:\Users\archa\Downloads\rampsense dataset\PV dataset\Dataset-SolarTechLab1_step4_clean.csv"

OUTPUT_FILE = r"C:\Users\archa\Downloads\rampsense dataset\PV dataset\Dataset-SolarTechLab1_step5_ramps.csv"

# PV system rating
RATED_POWER_W = 245.0

# Ramp threshold = 5% of rated power per minute
THRESHOLD_W_PER_MIN = 0.05 * RATED_POWER_W

# Forecast horizon
HORIZON_MIN = 10

# Tolerance window
DELTA_T_MIN = 3

# Temporal resolution
TIME_STEP_MIN = 1


# ============================================================
# LOAD DATA
# ============================================================

print("Loading PV dataset...")

df = pd.read_csv(INPUT_FILE)

df["Time"] = pd.to_datetime(df["Time"])

df = df.sort_values("Time").reset_index(drop=True)

print("Rows:", len(df))
print("Time range:", df["Time"].min(), "to", df["Time"].max())


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "PV_Power" not in df.columns:
    raise ValueError(
        "PV_Power column not found in dataset."
    )


# ============================================================
# RAMP RATE
# ============================================================

print("\nCalculating 1-minute power changes...")

# Consecutive absolute power change
df["ramp_rate_W_per_min"] = (
    df["PV_Power"].diff().abs()
    / TIME_STEP_MIN
)


# ============================================================
# FUTURE TOLERANCE WINDOW
# ============================================================

print("Calculating future maximum ramp...")

# For time t:
#
# target horizon = t + 10 min
#
# tolerance window:
#
# t + 10 - 3  = t + 7
# t + 10 + 3  = t + 13
#
# Therefore we search ramp rates from
# t+7 through t+13.
#
# A ramp rate at time s represents:
# |P(s) - P(s-1)| / Δt

future_max_ramp = np.full(
    len(df),
    np.nan,
    dtype=np.float32
)


# ============================================================
# GAP-AWARE CALCULATION
# ============================================================

# If run_id exists, calculate within each continuous run.
# This prevents ramps from being calculated across gaps.

if "run_id" in df.columns:

    print("Using run_id for gap-aware ramp calculation...")

    for run_id, group_index in df.groupby(
        "run_id",
        sort=False
    ).groups.items():

        indices = np.asarray(
            group_index,
            dtype=np.int64
        )

        ramp_values = df.loc[
            indices,
            "ramp_rate_W_per_min"
        ].to_numpy()

        n = len(indices)

        # For each starting position i,
        # future ramp positions are:
        #
        # i + 7 through i + 13

        for i in range(n):

            start = i + HORIZON_MIN - DELTA_T_MIN
            end = i + HORIZON_MIN + DELTA_T_MIN + 1

            if start >= n:
                continue

            end = min(end, n)

            values = ramp_values[start:end]

            if len(values) == 0:
                continue

            if np.all(np.isnan(values)):
                continue

            future_max_ramp[indices[i]] = np.nanmax(
                values
            )

else:

    print(
        "WARNING: run_id not found. "
        "Using global time sequence."
    )

    for i in range(len(df)):

        start = i + HORIZON_MIN - DELTA_T_MIN
        end = i + HORIZON_MIN + DELTA_T_MIN + 1

        if start >= len(df):
            continue

        end = min(end, len(df))

        values = df.loc[
            start:end - 1,
            "ramp_rate_W_per_min"
        ].to_numpy()

        if np.all(np.isnan(values)):
            continue

        future_max_ramp[i] = np.nanmax(values)


# ============================================================
# SAVE FUTURE MAX RAMP
# ============================================================

df["future_max_ramp_W_per_min"] = future_max_ramp


# ============================================================
# ORE LABEL
# ============================================================

print("\nCreating ORE labels...")

df["ORE"] = np.where(
    df["future_max_ramp_W_per_min"].notna(),
    (
        df["future_max_ramp_W_per_min"]
        >= THRESHOLD_W_PER_MIN
    ).astype(np.int8),
    np.nan
)


# ============================================================
# PRINT RESULTS
# ============================================================

total_rows = len(df)

labeled = df["ORE"].notna().sum()

ramps = (
    df["ORE"] == 1
).sum()

non_ramps = (
    df["ORE"] == 0
).sum()

unlabeled = (
    df["ORE"].isna()
).sum()


print("\n================================")
print("PV RAMP LABELING RESULTS")
print("================================")

print(
    f"Rated power       : {RATED_POWER_W:.2f} W"
)

print(
    f"Ramp threshold    : {THRESHOLD_W_PER_MIN:.2f} W/min"
)

print(
    f"Horizon            : {HORIZON_MIN} min"
)

print(
    f"Tolerance window   : ±{DELTA_T_MIN} min"
)

print(
    f"Temporal resolution: {TIME_STEP_MIN} min"
)

print(
    f"\nTotal rows         : {total_rows:,}"
)

print(
    f"Labeled rows       : {labeled:,}"
)

print(
    f"Ramp events        : {ramps:,}"
)

print(
    f"Non-ramp events    : {non_ramps:,}"
)

print(
    f"Unlabeled rows     : {unlabeled:,}"
)

if labeled > 0:

    ramp_percentage = (
        ramps / labeled
    ) * 100

    print(
        f"Ramp percentage    : {ramp_percentage:.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nDONE.")
