import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_DIR = r"c:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences"
SCALED_DIR = os.path.join(SEQUENCE_DIR, "scaled")

BATCH_SIZE = 256
EPOCHS = 30

LEARNING_RATE = 0.001

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0

PATIENCE = 5

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("================================")
print("PV BASELINE LSTM")
print("================================")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# DATASET
# ============================================================

class PVSequenceDataset(Dataset):

    def __init__(self, x_file, y_file):

        self.X = np.load(
            x_file,
            mmap_mode="r"
        )

        self.y = np.load(
            y_file,
            mmap_mode="r"
        )

        print("Loaded:")
        print("X:", self.X.shape)
        print("y:", self.y.shape)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        x = np.asarray(
            self.X[index],
            dtype=np.float32
        )

        y = np.float32(
            self.y[index]
        )

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32)
        )


# ============================================================
# FILES
# ============================================================

X_train_file = os.path.join(
    SCALED_DIR,
    "X_train.npy"
)

y_train_file = os.path.join(
    SCALED_DIR,
    "y_train.npy"
)

X_val_file = os.path.join(
    SCALED_DIR,
    "X_val.npy"
)

y_val_file = os.path.join(
    SCALED_DIR,
    "y_val.npy"
)

# ============================================================
# DATASETS
# ============================================================

print("\nLoading training dataset...")

train_dataset = PVSequenceDataset(
    X_train_file,
    y_train_file
)

print("\nLoading validation dataset...")

val_dataset = PVSequenceDataset(
    X_val_file,
    y_val_file
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# MODEL
# ============================================================

class PVLSTM(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_layers=1,
        dropout=0.0
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        # x shape:
        # (batch, 60, 6)

        output, (hidden, cell) = self.lstm(x)

        # Last timestep
        last_hidden = output[:, -1, :]

        last_hidden = self.dropout(
            last_hidden
        )

        prediction = self.fc(
            last_hidden
        )

        # (batch, 1) -> (batch,)
        return prediction.squeeze(1)


# ============================================================
# CREATE MODEL
# ============================================================

model = PVLSTM(
    input_size=6,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

model = model.to(device)

print("\nModel:")
print(model)

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

MODEL_DIR = os.path.join(
    SEQUENCE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

best_model_file = os.path.join(
    MODEL_DIR,
    "pv_baseline_lstm_best.pt"
)

# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

epochs_without_improvement = 0

print("\n================================")
print("STARTING TRAINING")
print("================================")

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_samples = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            device,
            non_blocking=True
        )

        batch_y = batch_y.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

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

        train_samples += batch_size

    train_loss = (
        train_loss_sum /
        train_samples
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_samples = 0

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

            val_samples += batch_size

    val_loss = (
        val_loss_sum /
        val_samples
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    train_rmse = np.sqrt(
        train_loss
    )

    val_rmse = np.sqrt(
        val_loss
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train MSE : {train_loss:.4f}"
    )

    print(
        f"Train RMSE: {train_rmse:.4f} W"
    )

    print(
        f"Val MSE   : {val_loss:.4f}"
    )

    print(
        f"Val RMSE  : {val_rmse:.4f} W"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "hidden_size": HIDDEN_SIZE,
                "input_size": 6
            },
            best_model_file
        )

        print(
            "✓ Best model saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# FINISHED
# ============================================================

print("\n================================")
print("TRAINING COMPLETE")
print("================================")

print(
    "Best validation MSE:",
    best_val_loss
)

print(
    "Best validation RMSE:",
    np.sqrt(best_val_loss),
    "W"
)

print(
    "\nModel saved to:"
)

print(
    best_model_file
)
