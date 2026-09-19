import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================
# PATHS
# ============================================================

SEQUENCE_DIR = r"C:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences"

SPLIT_DIR = os.path.join(
    SEQUENCE_DIR,
    "splits"
)

OUTPUT_DIR = os.path.join(
    SEQUENCE_DIR,
    "scaled"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 5000

N_FEATURES = 6

FEATURE_NAMES = [
    "PV_Power",
    "G_h",
    "G_tilt",
    "T_air",
    "W_s",
    "W_d"
]


# ============================================================
# LOAD ORIGINAL X
# ============================================================

print("Loading X.npy using memory mapping...")

X = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "X.npy"
    ),
    mmap_mode="r"
)

print("X shape:", X.shape)
print("X dtype:", X.dtype)


# ============================================================
# LOAD SPLIT INDICES
# ============================================================

train_idx = np.load(
    os.path.join(
        SPLIT_DIR,
        "train_idx.npy"
    )
)

val_idx = np.load(
    os.path.join(
        SPLIT_DIR,
        "val_idx.npy"
    )
)

test_idx = np.load(
    os.path.join(
        SPLIT_DIR,
        "test_idx.npy"
    )
)

print("\nSplit sizes:")
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))


# ============================================================
# FIT SCALER ON TRAIN ONLY
# ============================================================

print("\n================================")
print("FITTING STANDARD SCALER")
print("================================")

scaler = StandardScaler()

for start in range(
    0,
    len(train_idx),
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        len(train_idx)
    )

    indices = train_idx[
        start:end
    ]

    batch = np.asarray(
        X[indices],
        dtype=np.float32
    )

    # (samples, 60, 6)
    #        ↓
    # (samples*60, 6)

    batch_2d = batch.reshape(
        -1,
        N_FEATURES
    )

    scaler.partial_fit(
        batch_2d
    )

    print(
        f"Scaler: {end:,}/{len(train_idx):,}"
    )


# ============================================================
# PRINT SCALER PARAMETERS
# ============================================================

print("\nTraining-data means:")

for name, value in zip(
    FEATURE_NAMES,
    scaler.mean_
):

    print(
        f"{name:10s}: {value:.6f}"
    )


print("\nTraining-data standard deviations:")

for name, value in zip(
    FEATURE_NAMES,
    scaler.scale_
):

    print(
        f"{name:10s}: {value:.6f}"
    )


# ============================================================
# SAVE SCALER
# ============================================================

scaler_file = os.path.join(
    OUTPUT_DIR,
    "pv_scaler.pkl"
)

with open(
    scaler_file,
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )

print("\nScaler saved:")
print(scaler_file)


# ============================================================
# FUNCTION TO SCALE DATA
# ============================================================

def scale_and_save(
    indices,
    output_filename,
    split_name
):

    print("\n================================")
    print(f"SCALING {split_name}")
    print("================================")

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    n_samples = len(indices)

    # Create memory-mapped output file
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            n_samples,
            X.shape[1],
            X.shape[2]
        )
    )

    for start in range(
        0,
        n_samples,
        CHUNK_SIZE
    ):

        end = min(
            start + CHUNK_SIZE,
            n_samples
        )

        batch_indices = indices[
            start:end
        ]

        batch = np.asarray(
            X[batch_indices],
            dtype=np.float32
        )

        original_shape = batch.shape

        batch_2d = batch.reshape(
            -1,
            N_FEATURES
        )

        scaled = scaler.transform(
            batch_2d
        )

        scaled = scaled.reshape(
            original_shape
        )

        output[start:end] = (
            scaled.astype(
                np.float32
            )
        )

        print(
            f"{split_name}: "
            f"{end:,}/{n_samples:,}"
        )

    output.flush()

    del output

    print(
        f"{split_name} saved:"
    )

    print(
        output_path
    )


# ============================================================
# SCALE TRAIN
# ============================================================

scale_and_save(
    train_idx,
    "X_train.npy",
    "TRAIN"
)


# ============================================================
# SCALE VALIDATION
# ============================================================

scale_and_save(
    val_idx,
    "X_val.npy",
    "VALIDATION"
)


# ============================================================
# SCALE TEST
# ============================================================

scale_and_save(
    test_idx,
    "X_test.npy",
    "TEST"
)


# ============================================================
# SAVE TARGETS
# ============================================================

print("\n================================")
print("SAVING TARGETS")
print("================================")

y = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "y.npy"
    ),
    mmap_mode="r"
)

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
# FINAL CHECK
# ============================================================

print("\n================================")
print("SCALING COMPLETE")
print("================================")

print("Output folder:")
print(OUTPUT_DIR)

print("\nCreated files:")
print("X_train.npy")
print("X_val.npy")
print("X_test.npy")
print("y_train.npy")
print("y_val.npy")
print("y_test.npy")
print("pv_scaler.pkl")

print("\nScaler was fitted using TRAIN data only.")

print("\nDONE.")
