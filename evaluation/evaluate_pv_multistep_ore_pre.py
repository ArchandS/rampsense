import os
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Existing scaled input sequences
INPUT_X = r"c:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\X.npy"

# Original cleaned PV dataset
PV_DATA = r"c:\Users\archa\Downloads\rampsense dataset\PV dataset\Dataset-SolarTechLab1_final.csv"

# Output directory
OUTPUT_DIR = r"c:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\multistep"

# Input history
INPUT_LENGTH = 60

# Forecast horizon
FORECAST_HORIZON = 10

# Paper tolerance
TOLERANCE = 3

# Future window:
# t+7 ... t+13
FUTURE_OFFSETS = np.arange(
    FORECAST_HORIZON - TOLERANCE,
    FORECAST_HORIZON + TOLERANCE + 1
)

# 7 future values
FUTURE_LENGTH = len(FUTURE_OFFSETS)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD INPUT SEQUENCES
# ============================================================

print("=" * 60)
print("CREATING MULTI-STEP PV SEQUENCES")
print("=" * 60)

print("\nLoading X...")

X = np.load(INPUT_X, mmap_mode="r")

print("X shape:", X.shape)
print("X dtype:", X.dtype)


# ============================================================
# LOAD PV DATA
# ============================================================

print("\nLoading PV dataset...")

import pandas as pd

df = pd.read_csv(
    PV_DATA,
    usecols=["Time", "PV_Power"]
)

df["Time"] = pd.to_datetime(df["Time"])

df = df.sort_values("Time").reset_index(drop=True)

print("PV rows:", len(df))
print("First time:", df["Time"].iloc[0])
print("Last time:", df["Time"].iloc[-1])


# ============================================================
# BUILD TIME → ROW LOOKUP
# ============================================================

print("\nBuilding timestamp lookup...")

time_values = df["Time"].values
pv_values = df["PV_Power"].values.astype(np.float32)

time_to_index = {
    timestamp: i
    for i, timestamp in enumerate(time_values)
}


# ============================================================
# LOAD EXISTING BASELINE TIMES
# ============================================================

BASELINE_TIMES = r"c:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\sequence_times.npy"

print("\nLoading sequence timestamps...")

sequence_times = np.load(BASELINE_TIMES)

sequence_times = sequence_times.astype("datetime64[ns]")

print("Sequence count:", len(sequence_times))


# ============================================================
# CREATE MULTI-STEP TARGETS
# ============================================================

print("\nCreating future targets...")

y_future = np.full(
    (len(sequence_times), FUTURE_LENGTH),
    np.nan,
    dtype=np.float32
)

valid = np.ones(
    len(sequence_times),
    dtype=bool
)


for i, current_time in enumerate(sequence_times):

    # Current sequence ends at time t
    current_timestamp = pd.Timestamp(current_time)

    for j, offset in enumerate(FUTURE_OFFSETS):

        target_time = current_timestamp + pd.Timedelta(
            minutes=int(offset)
        )

        target_index = time_to_index.get(
            target_time.to_datetime64()
        )

        if target_index is None:
            valid[i] = False
            break

        value = pv_values[target_index]

        if np.isnan(value):
            valid[i] = False
            break

        y_future[i, j] = value


# ============================================================
# KEEP ONLY VALID SEQUENCES
# ============================================================

print("\nFiltering invalid sequences...")

valid_indices = np.where(valid)[0]

X_valid = X[valid_indices]
y_valid = y_future[valid_indices]
times_valid = sequence_times[valid_indices]


# ============================================================
# SAVE
# ============================================================

X_output = os.path.join(
    OUTPUT_DIR,
    "X_multistep.npy"
)

Y_output = os.path.join(
    OUTPUT_DIR,
    "y_future_7.npy"
)

TIMES_output = os.path.join(
    OUTPUT_DIR,
    "times_multistep.npy"
)

INDICES_output = os.path.join(
    OUTPUT_DIR,
    "valid_indices.npy"
)


print("\nSaving X...")

# Copy from memory-mapped source into normal writable array
X_valid_array = np.asarray(X_valid, dtype=np.float32)

np.save(
    X_output,
    X_valid_array
)

print("Saved:", X_output)


print("\nSaving future targets...")

np.save(
    Y_output,
    y_valid
)

print("Saved:", Y_output)


print("\nSaving timestamps...")

np.save(
    TIMES_output,
    times_valid
)

print("Saved:", TIMES_output)


print("\nSaving indices...")

np.save(
    INDICES_output,
    valid_indices
)

print("Saved:", INDICES_output)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MULTI-STEP SEQUENCE SUMMARY")
print("=" * 60)

print("Original sequences :", len(sequence_times))
print("Valid sequences    :", len(valid_indices))
print("Removed sequences  :", len(sequence_times) - len(valid_indices))

print("\nX shape:")
print(X_valid_array.shape)

print("\ny_future shape:")
print(y_valid.shape)

print("\nFuture offsets:")
print(FUTURE_OFFSETS)

print("\nFirst sequence:")
print("End time:", times_valid[0])

print("Future times:")

for offset in FUTURE_OFFSETS:
    print(
        "  t+%d =" % offset,
        pd.Timestamp(times_valid[0]) +
        pd.Timedelta(minutes=int(offset))
    )

print("\nFirst future target:")
print(y_valid[0])

print("\nDONE.")
