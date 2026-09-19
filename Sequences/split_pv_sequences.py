import os
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

SEQUENCE_DIR = r"C:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences"

OUTPUT_DIR = os.path.join(
    SEQUENCE_DIR,
    "splits"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD SEQUENCE METADATA
# ============================================================

print("Loading sequence timestamps...")

times = pd.to_datetime(
    np.load(
        os.path.join(
            SEQUENCE_DIR,
            "sequence_times.npy"
        ),
        allow_pickle=True
    )
)

runs = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "sequence_runs.npy"
    ),
    allow_pickle=True
)

y = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "y.npy"
    ),
    mmap_mode="r"
)


# ============================================================
# BASIC CHECK
# ============================================================

print("\n================================")
print("PV CHRONOLOGICAL SPLIT")
print("================================")

print(
    "Total sequences:",
    len(times)
)

if not (
    len(times)
    == len(runs)
    == len(y)
):

    raise ValueError(
        "Sequence metadata lengths do not match."
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

# TRAIN:
# 2017-01-01 → 2017-08-31
#
# VALIDATION:
# 2017-09-01 → 2017-10-31
#
# TEST:
# 2017-11-01 → 2017-12-31

train_mask = (
    (times >= "2017-01-01")
    &
    (times < "2017-09-01")
)

val_mask = (
    (times >= "2017-09-01")
    &
    (times < "2017-11-01")
)

test_mask = (
    (times >= "2017-11-01")
    &
    (times < "2018-01-01")
)


# ============================================================
# GET INDICES
# ============================================================

train_idx = np.where(
    train_mask
)[0]

val_idx = np.where(
    val_mask
)[0]

test_idx = np.where(
    test_mask
)[0]


# ============================================================
# CHECK THAT EVERY SEQUENCE IS ASSIGNED
# ============================================================

total_assigned = (
    len(train_idx)
    + len(val_idx)
    + len(test_idx)
)

print("\nAssigned sequences:", total_assigned)

if total_assigned != len(times):

    unassigned = (
        len(times)
        - total_assigned
    )

    print(
        "WARNING:",
        unassigned,
        "sequences were not assigned."
    )

else:

    print(
        "PASS: Every sequence belongs to a split."
    )


# ============================================================
# PRINT SPLIT INFORMATION
# ============================================================

print("\n--------------------------------")
print("TRAIN")
print("--------------------------------")

print(
    "Sequences:",
    len(train_idx)
)

if len(train_idx) > 0:

    print(
        "Start:",
        times[train_idx[0]]
    )

    print(
        "End:",
        times[train_idx[-1]]
    )


print("\n--------------------------------")
print("VALIDATION")
print("--------------------------------")

print(
    "Sequences:",
    len(val_idx)
)

if len(val_idx) > 0:

    print(
        "Start:",
        times[val_idx[0]]
    )

    print(
        "End:",
        times[val_idx[-1]]
    )


print("\n--------------------------------")
print("TEST")
print("--------------------------------")

print(
    "Sequences:",
    len(test_idx)
)

if len(test_idx) > 0:

    print(
        "Start:",
        times[test_idx[0]]
    )

    print(
        "End:",
        times[test_idx[-1]]
    )


# ============================================================
# SAVE INDICES
# ============================================================

print("\nSaving split indices...")

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_idx.npy"
    ),
    train_idx
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "val_idx.npy"
    ),
    val_idx
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_idx.npy"
    ),
    test_idx
)


# ============================================================
# SAVE TARGETS
# ============================================================

print("Saving target arrays...")

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    np.asarray(
        y[train_idx],
        dtype=np.float32
    )
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_val.npy"
    ),
    np.asarray(
        y[val_idx],
        dtype=np.float32
    )
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    np.asarray(
        y[test_idx],
        dtype=np.float32
    )
)


# ============================================================
# SAVE TIMESTAMPS
# ============================================================

print("Saving timestamps...")

np.save(
    os.path.join(
        OUTPUT_DIR,
        "times_train.npy"
    ),
    times[train_idx].values
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "times_val.npy"
    ),
    times[val_idx].values
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "times_test.npy"
    ),
    times[test_idx].values
)


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary = pd.DataFrame({

    "split": [
        "train",
        "validation",
        "test"
    ],

    "number_of_sequences": [
        len(train_idx),
        len(val_idx),
        len(test_idx)
    ],

    "start_time": [
        times[train_idx[0]],
        times[val_idx[0]],
        times[test_idx[0]]
    ],

    "end_time": [
        times[train_idx[-1]],
        times[val_idx[-1]],
        times[test_idx[-1]]
    ]
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "split_summary.csv"
    ),
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n================================")
print("SPLIT COMPLETE")
print("================================")

print(
    "Train:",
    len(train_idx)
)

print(
    "Validation:",
    len(val_idx)
)

print(
    "Test:",
    len(test_idx)
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR
)

print("\nDONE.")
