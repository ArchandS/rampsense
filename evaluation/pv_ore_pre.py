import numpy as np


# ============================================================
# PV ORE / PRE PARAMETERS
# ============================================================

FORECAST_HORIZON = 10       # minutes
TOLERANCE = 3               # ±3 minutes
TIME_RESOLUTION = 1         # 1 minute
RAMP_THRESHOLD = 12.25      # W/min = 5% of 245 W


# ============================================================
# CALCULATE MAXIMUM RAMP IN THE PAPER'S WINDOW
# ============================================================

def max_ramp_in_window(power_values):
    """
    Calculate the maximum absolute consecutive power change.

    power_values:
        Consecutive PV power values at 1-minute resolution.

    Returns:
        Maximum ramp magnitude in W/min.
    """

    power_values = np.asarray(power_values, dtype=np.float32)

    if len(power_values) < 2:
        return np.nan

    differences = np.abs(np.diff(power_values))

    ramp_rates = differences / TIME_RESOLUTION

    return float(np.max(ramp_rates))


# ============================================================
# CONVERT RAMP MAGNITUDE TO ORE LABEL
# ============================================================

def ramp_label(power_values):
    """
    Return 1 if the window contains a critical ramp,
    otherwise 0.
    """

    max_ramp = max_ramp_in_window(power_values)

    if np.isnan(max_ramp):
        return np.nan

    return int(max_ramp >= RAMP_THRESHOLD)


# ============================================================
# CALCULATE TP / FP / FN / TN
# ============================================================

def calculate_confusion_matrix(actual_labels, predicted_labels):

    actual_labels = np.asarray(actual_labels)
    predicted_labels = np.asarray(predicted_labels)

    valid = (
        ~np.isnan(actual_labels)
        & ~np.isnan(predicted_labels)
    )

    actual = actual_labels[valid].astype(int)
    predicted = predicted_labels[valid].astype(int)

    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    tn = np.sum((actual == 0) & (predicted == 0))

    return tp, fp, fn, tn


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

def calculate_metrics(tp, fp, fn, tn):

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    detection = recall

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection": detection
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Example: no significant ramp
    normal_power = [
        100,
        101,
        102,
        103,
        104
    ]

    # Example: significant ramp
    ramp_power = [
        100,
        101,
        102,
        120,
        121
    ]

    print("PV ORE/PRE utility test")
    print("========================")

    print(
        "Normal window ramp:",
        max_ramp_in_window(normal_power),
        "W/min"
    )

    print(
        "Ramp window ramp:",
        max_ramp_in_window(ramp_power),
        "W/min"
    )

    print(
        "Normal ORE:",
        ramp_label(normal_power)
    )

    print(
        "Ramp ORE:",
        ramp_label(ramp_power)
    )

    actual = np.array([1, 1, 0, 0, 1])
    predicted = np.array([1, 0, 0, 1, 0])

    tp, fp, fn, tn = calculate_confusion_matrix(
        actual,
        predicted
    )

    metrics = calculate_metrics(
        tp, fp, fn, tn
    )

    print("\nConfusion matrix")
    print("----------------")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)

    print("\nMetrics")
    print("-------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
