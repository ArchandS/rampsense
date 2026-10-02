import pandas as pd
import numpy as np

FILE = r"C:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\evaluation\pv_multistep_ore_pre_predictions.csv"

THRESHOLD = 12.25

df = pd.read_csv(FILE)

actual = df["actual_max_ramp_W_per_min"].values
predicted = df["predicted_max_ramp_W_per_min"].values

print("=" * 70)
print("MULTI-STEP RAMP DIAGNOSTIC")
print("=" * 70)

print("\nACTUAL MAX RAMP")
print("----------------")
print("Minimum :", np.min(actual))
print("Median  :", np.median(actual))
print("Mean    :", np.mean(actual))
print("95th %  :", np.percentile(actual, 95))
print("99th %  :", np.percentile(actual, 99))
print("Maximum :", np.max(actual))

print("\nPREDICTED MAX RAMP")
print("------------------")
print("Minimum :", np.min(predicted))
print("Median  :", np.median(predicted))
print("Mean    :", np.mean(predicted))
print("95th % :", np.percentile(predicted, 95))
print("99th % :", np.percentile(predicted, 99))
print("Maximum :", np.max(predicted))

print("\nTHRESHOLD")
print("---------")
print("Threshold:", THRESHOLD)

print(
    "Actual windows >= threshold:",
    np.sum(actual >= THRESHOLD)
)

print(
    "Predicted windows >= threshold:",
    np.sum(predicted >= THRESHOLD)
)

print("\nTOP 20 PREDICTED RAMPS")
print("----------------------")

top_indices = np.argsort(predicted)[-20:][::-1]

for i in top_indices:

    print(
        f"{df['time'].iloc[i]} | "
        f"actual={actual[i]:.2f} W/min | "
        f"predicted={predicted[i]:.2f} W/min"
    )

print("\nTOP 20 ACTUAL RAMPS")
print("-------------------")

top_actual = np.argsort(actual)[-20:][::-1]

for i in top_actual:

    print(
        f"{df['time'].iloc[i]} | "
        f"actual={actual[i]:.2f} W/min | "
        f"predicted={predicted[i]:.2f} W/min"
    )

print("\nDONE.")
