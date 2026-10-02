import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\multistep\split_scaled"

MODEL_DIR = r"C:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\models"

os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 512
HIDDEN_SIZE = 64
NUM_LAYERS = 1

LEARNING_RATE = 0.001
EPOCHS = 30
PATIENCE = 5

INPUT_SIZE = 6
OUTPUT_SIZE = 7

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "pv_multistep_lstm_gpu_best.pt"
)


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available in this PyTorch installation."
    )

device = torch.device("cuda")

print("=" * 60)
print("MULTI-STEP PV LSTM - GPU TRAINING")
print("=" * 60)

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))

print(
    "CUDA version:",
    torch.version.cuda
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_val = np.load(
    os.path.join(DATA_DIR, "X_val.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, "y_val.npy")
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ============================================================
# CONVERT TO TORCH
# ============================================================

X_train_tensor = torch.from_numpy(
    X_train
)

y_train_tensor = torch.from_numpy(
    y_train
)

X_val_tensor = torch.from_numpy(
    X_val
)

y_val_tensor = torch.from_numpy(
    y_val
)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

class PVMultiStepLSTM(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_layers=1,
        output_size=7
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        prediction = self.fc(
            last_output
        )

        return prediction


model = PVMultiStepLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE
).to(device)


print("\nModel:")
print(model)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

patience_counter = 0


print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_count = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            device,
            non_blocking=True
        )

        batch_y = batch_y.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            batch_x
        )

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        batch_size = batch_x.size(0)

        train_loss_sum += (
            loss.item() * batch_size
        )

        train_count += batch_size


    train_loss = (
        train_loss_sum / train_count
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(
                device,
                non_blocking=True
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True
            )

            predictions = model(
                batch_x
            )

            loss = criterion(
                predictions,
                batch_y
            )

            batch_size = batch_x.size(0)

            val_loss_sum += (
                loss.item() * batch_size
            )

            val_count += batch_size


    val_loss = (
        val_loss_sum / val_count
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    train_rmse = np.sqrt(train_loss)
    val_rmse = np.sqrt(val_loss)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f} | "
        f"Train RMSE: {train_rmse:.6f} | "
        f"Val RMSE: {val_rmse:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_size":
                    INPUT_SIZE,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "output_size":
                    OUTPUT_SIZE,

                "best_val_mse":
                    best_val_loss
            },
            MODEL_PATH
        )

        print("  -> Best model saved.")

    else:

        patience_counter += 1

        print(
            f"  -> No improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print("\nEarly stopping.")
        break


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("GPU TRAINING COMPLETE")
print("=" * 60)

print(
    "Best validation MSE:",
    best_val_loss
)

print(
    "Best validation RMSE:",
    np.sqrt(best_val_loss)
)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nDONE.")
