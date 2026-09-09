
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

DATA_FOLDER = Path(
    r"C:\Users\archa\Downloads\rampsense dataset\wind dataset"
)

ZIP_FILES = [
    DATA_FOLDER / "Penmanshiel_SCADA_2017_WT11-15_3115.zip",
    DATA_FOLDER / "Penmanshiel_SCADA_2018_WT11-15_3116.zip",
]

OUTPUT_FILE = DATA_FOLDER / "Penmanshiel_WT15_2017_2018_extracted.csv"


# ============================================================
# FIND WT15 CSV
# ============================================================

def find_wt15_csv(zip_path):

    print("\n" + "=" * 70)
    print("OPENING:", zip_path.name)
    print("=" * 70)

    with zipfile.ZipFile(zip_path, "r") as z:

        files = z.namelist()

        candidates = [
            f for f in files
            if (
                "Turbine_Data" in f
                and "Penmanshiel_15" in f
                and f.lower().endswith(".csv")
            )
        ]

        if not candidates:

            print("WT15 CSV not found.")

            print("\nAvailable CSV files:")

            for f in files:

                if f.lower().endswith(".csv"):
                    print(f)

            raise FileNotFoundError(
                "WT15 CSV was not found."
            )

        print("\nWT15 CSV found:")
        print(candidates[0])

        return candidates[0]


# ============================================================
# FIND COMMENTED HEADER
# ============================================================

def find_header(zip_path, csv_name):

    print("\nSearching for the real header...")

    with zipfile.ZipFile(zip_path, "r") as z:

        with z.open(csv_name) as f:

            for line_number in range(100):

                raw = f.readline()

                if not raw:
                    break

                line = raw.decode(
                    "utf-8",
                    errors="replace"
                ).strip()

                # The actual header starts with:
                # # Date and time,...

                if line.startswith(
                    "# Date and time"
                ):

                    print(
                        "\nHeader found at line:",
                        line_number + 1
                    )

                    # Remove "# " from beginning
                    header = line[1:].strip()

                    print(
                        "\nHeader preview:"
                    )

                    print(
                        header[:500]
                    )

                    return line_number, header

    raise ValueError(
        "Could not find '# Date and time' header."
    )


# ============================================================
# READ WT15
# ============================================================

def read_wt15(zip_path):

    csv_name = find_wt15_csv(
        zip_path
    )

    header_line, header_text = find_header(
        zip_path,
        csv_name
    )

    # --------------------------------------------------------
    # Parse the commented header correctly
    # --------------------------------------------------------

    from io import StringIO

    header_df = pd.read_csv(
        StringIO(header_text),
        sep=","
    )

    columns = header_df.columns.tolist()

    print(
        "\nNumber of columns:",
        len(columns)
    )

    # --------------------------------------------------------
    # Find required columns
    # --------------------------------------------------------

    timestamp_candidates = [
        c for c in columns
        if "date and time" in c.lower()
    ]

    wind_speed_candidates = [
        c for c in columns
        if c.lower().strip()
        == "wind speed (m/s)"
    ]

    wind_direction_candidates = [
        c for c in columns
        if "wind direction" in c.lower()
        and "sensor" not in c.lower()
    ]

    power_candidates = [
        c for c in columns
        if "active power" in c.lower()
    ]

    # --------------------------------------------------------
    # Print candidates
    # --------------------------------------------------------

    print("\nTimestamp candidates:")

    for c in timestamp_candidates:
        print(" ", c)

    print("\nWind speed candidates:")

    for c in wind_speed_candidates:
        print(" ", c)

    print("\nWind direction candidates:")

    for c in wind_direction_candidates:
        print(" ", c)

    print("\nActive power candidates:")

    for c in power_candidates:
        print(" ", c)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not timestamp_candidates:
        raise ValueError(
            "Timestamp column not found."
        )

    if not wind_speed_candidates:
        raise ValueError(
            "Wind speed column not found."
        )

    if not wind_direction_candidates:
        raise ValueError(
            "Wind direction column not found."
        )

    if not power_candidates:
        raise ValueError(
            "Active power column not found."
        )

    # --------------------------------------------------------
    # Select columns
    # --------------------------------------------------------

    timestamp_col = timestamp_candidates[0]

    wind_speed_col = wind_speed_candidates[0]

    wind_direction_col = wind_direction_candidates[0]

    power_col = power_candidates[0]

    print("\n" + "=" * 70)
    print("SELECTED COLUMNS")
    print("=" * 70)

    print(
        "Timestamp      :",
        timestamp_col
    )

    print(
        "Wind speed     :",
        wind_speed_col
    )

    print(
        "Wind direction :",
        wind_direction_col
    )

    print(
        "Active power   :",
        power_col
    )

    # --------------------------------------------------------
    # Read data
    #
    # header_line is zero-indexed.
    #
    # Example:
    # header line = 9
    #
    # Data starts after line 10.
    # --------------------------------------------------------

    with zipfile.ZipFile(zip_path, "r") as z:

        with z.open(csv_name) as f:

            df = pd.read_csv(
                f,
                skiprows=header_line + 1,
                names=columns,
                usecols=[
                    timestamp_col,
                    wind_speed_col,
                    wind_direction_col,
                    power_col
                ],
                sep=",",
                na_values=[
                    "NaN",
                    "nan",
                    "",
                    "NA"
                ],
                low_memory=False
            )

    # --------------------------------------------------------
    # Rename
    # --------------------------------------------------------

    df = df.rename(
        columns={
            timestamp_col: "timestamp",
            wind_speed_col: "wind_speed",
            wind_direction_col: "wind_direction",
            power_col: "power"
        }
    )

    print(
        "\nRows loaded:",
        len(df)
    )

    return df


# ============================================================
# QUALITY CHECK
# ============================================================

def quality_check(df, year):

    print("\n" + "=" * 70)
    print(f"QUALITY CHECK - {year}")
    print("=" * 70)

    print(
        "Rows:",
        len(df)
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_time = df["timestamp"].isna().sum()

    print(
        "Invalid timestamps:",
        invalid_time
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    # Sort
    df = df.sort_values(
        "timestamp"
    )

    # Duplicate timestamps
    duplicates = df[
        "timestamp"
    ].duplicated().sum()

    print(
        "Duplicate timestamps:",
        duplicates
    )

    df = df.drop_duplicates(
        subset="timestamp",
        keep="first"
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in [
        "wind_speed",
        "wind_direction",
        "power"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    for col in [
        "wind_speed",
        "wind_direction",
        "power"
    ]:

        print(
            f"{col:18s}:",
            df[col].isna().sum()
        )

    # --------------------------------------------------------
    # Physical checks
    # --------------------------------------------------------

    negative_wind = (
        df["wind_speed"] < 0
    ).sum()

    print(
        "\nNegative wind speed:",
        negative_wind
    )

    invalid_direction = (
        (df["wind_direction"] < 0)
        |
        (df["wind_direction"] > 360)
    ).sum()

    print(
        "Invalid wind direction:",
        invalid_direction
    )

    # --------------------------------------------------------
    # Time interval
    # --------------------------------------------------------

    time_difference = (
        df["timestamp"].diff()
    )

    ten_minutes = pd.Timedelta(
        minutes=10
    )

    correct_intervals = (
        time_difference == ten_minutes
    ).sum()

    incorrect_intervals = (
        (
            time_difference.notna()
        )
        &
        (
            time_difference != ten_minutes
        )
    ).sum()

    print(
        "\nCorrect 10-minute intervals:",
        correct_intervals
    )

    print(
        "Incorrect intervals:",
        incorrect_intervals
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nWind speed statistics:")

    print(
        df["wind_speed"].describe()
    )

    print("\nWind direction statistics:")

    print(
        df["wind_direction"].describe()
    )

    print("\nPower statistics:")

    print(
        df["power"].describe()
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PENMANSHIEL WT15 PREPROCESSING")
    print("=" * 70)

    yearly_data = []

    # --------------------------------------------------------
    # Process each ZIP
    # --------------------------------------------------------

    for zip_file in ZIP_FILES:

        if not zip_file.exists():

            raise FileNotFoundError(
                f"\nZIP file not found:\n{zip_file}"
            )

        df = read_wt15(
            zip_file
        )

        if "2017" in zip_file.name:
            year = 2017

        elif "2018" in zip_file.name:
            year = 2018

        else:
            year = "unknown"

        df = quality_check(
            df,
            year
        )

        yearly_data.append(
            df
        )

    # --------------------------------------------------------
    # Combine years
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINING 2017 + 2018")
    print("=" * 70)

    combined = pd.concat(
        yearly_data,
        ignore_index=True
    )

    combined = combined.sort_values(
        "timestamp"
    )

    combined = combined.drop_duplicates(
        subset="timestamp",
        keep="first"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\nFinal number of rows:",
        len(combined)
    )

    print(
        "\nStart:",
        combined["timestamp"].min()
    )

    print(
        "End:",
        combined["timestamp"].max()
    )

    print("\nFinal missing values:")

    print(
        combined[
            [
                "wind_speed",
                "wind_direction",
                "power"
            ]
        ].isna().sum()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nOriginal ZIP files were NOT modified."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
