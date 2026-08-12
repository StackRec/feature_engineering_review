from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "bank.csv"

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/222/"
    "bank+marketing.zip"
)

SAMPLE_SIZE = 6000
RANDOM_STATE = 42


# -------------------------------------------------------------------
# Download UCI Bank Marketing dataset
# -------------------------------------------------------------------

print("Downloading UCI Bank Marketing dataset...")

with urlopen(UCI_URL) as response:
    outer_zip_bytes = response.read()

print(
    f"Downloaded "
    f"{len(outer_zip_bytes) / 1024 / 1024:.2f} MB"
)


# -------------------------------------------------------------------
# Open outer ZIP
#
# The UCI archive currently contains:
#
#   bank.zip
#   bank-additional.zip
#
# We want bank.zip because it contains bank-full.csv.
# -------------------------------------------------------------------

with ZipFile(BytesIO(outer_zip_bytes)) as outer_zip:

    outer_files = outer_zip.namelist()

    print()
    print("Files found in outer archive:")

    for filename in outer_files:
        print(f"   {filename}")

    bank_zip_files = [
        filename
        for filename in outer_files
        if filename.lower().endswith("/bank.zip")
        or filename.lower() == "bank.zip"
    ]

    if not bank_zip_files:
        raise FileNotFoundError(
            "Could not find bank.zip in the UCI archive."
        )

    bank_zip_name = bank_zip_files[0]

    print()
    print(f"Opening nested archive: {bank_zip_name}")

    bank_zip_bytes = outer_zip.read(bank_zip_name)


# -------------------------------------------------------------------
# Open nested bank.zip
# -------------------------------------------------------------------

with ZipFile(BytesIO(bank_zip_bytes)) as bank_zip:

    bank_files = bank_zip.namelist()

    print()
    print("Files found in bank.zip:")

    for filename in bank_files:
        print(f"   {filename}")

    csv_files = [
        filename
        for filename in bank_files
        if filename.lower().endswith("bank-full.csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            "Could not find bank-full.csv inside bank.zip."
        )

    csv_path = csv_files[0]

    print()
    print(f"Reading dataset: {csv_path}")

    with bank_zip.open(csv_path) as file:

        df = pd.read_csv(
            file,
            sep=";",
            quotechar='"'
        )


# -------------------------------------------------------------------
# Validate the dataset
# -------------------------------------------------------------------

print()
print(f"Original dataset shape: {df.shape}")

if "y" not in df.columns:
    raise ValueError(
        "Target column 'y' was not found in the dataset."
    )


# -------------------------------------------------------------------
# Convert target variable
#
# Original UCI values:
#     yes -> 1
#     no  -> 0
# -------------------------------------------------------------------

df["y"] = (
    df["y"]
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)


# Make sure target conversion worked
if df["y"].isna().any():

    invalid_values = (
        df.loc[df["y"].isna(), "y"]
        .unique()
    )

    raise ValueError(
        "Unexpected values were found in target column."
    )


df["y"] = df["y"].astype(int)


# -------------------------------------------------------------------
# Show original target distribution
# -------------------------------------------------------------------

print()
print("Original target distribution:")

print(
    df["y"]
    .value_counts()
    .sort_index()
)

print()
print("Original target proportions:")

print(
    df["y"]
    .value_counts(normalize=True)
    .sort_index()
)


# -------------------------------------------------------------------
# Create stratified sample
#
# This preserves approximately the same class distribution as the
# original dataset while reducing the dataset to 6,000 observations.
# -------------------------------------------------------------------

if len(df) > SAMPLE_SIZE:

    sampled_indices, _ = train_test_split(
        df.index,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["y"]
    )

    df = df.loc[sampled_indices].copy()

    # Shuffle rows so that positive/negative observations aren't
    # grouped together.
    df = df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

else:

    print()
    print(
        f"Dataset contains {len(df)} rows, "
        f"so no sampling was required."
    )


# -------------------------------------------------------------------
# Validate final dataset
# -------------------------------------------------------------------

if df["y"].nunique() < 2:

    raise ValueError(
        "Final dataset contains only one target class. "
        "Both classes are required for model training."
    )


# -------------------------------------------------------------------
# Save prepared dataset
# -------------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -------------------------------------------------------------------
# Final summary
# -------------------------------------------------------------------

print()
print("=" * 60)
print("Dataset preparation complete")
print("=" * 60)

print()
print(f"Final dataset shape: {df.shape}")

print()
print(f"Saved to:")
print(OUTPUT_FILE)

print()
print("Final target distribution:")

print(
    df["y"]
    .value_counts()
    .sort_index()
)

print()
print("Final target proportions:")

print(
    df["y"]
    .value_counts(normalize=True)
    .sort_index()
)

print()
print("Columns:")

for column in df.columns:
    print(f"  - {column}")

print()
print("Dataset preview:")

print(df.head())