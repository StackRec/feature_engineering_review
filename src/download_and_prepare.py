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
# Download outer archive
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
# -------------------------------------------------------------------

with ZipFile(BytesIO(outer_zip_bytes)) as outer_zip:

    files = outer_zip.namelist()

    print("Files found in outer archive:")

    for name in files:
        print(f"   {name}")

    # Find nested bank.zip
    bank_zip_files = [
        name
        for name in files
        if name.lower().endswith("bank.zip")
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

    files = bank_zip.namelist()

    print()
    print("Files found in bank.zip:")

    for name in files:
        print(f"   {name}")

    # Find bank-full.csv
    csv_files = [
        name
        for name in files
        if name.lower().endswith("bank-full.csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            "Could not find bank-full.csv inside bank.zip."
        )

    csv_path = csv_files[0]

    print()
    print(f"Reading dataset: {csv_path}")

    with bank_zip.open(csv_path) as f:

        df = pd.read_csv(
            f,
            sep=";",
            quotechar='"'
        )


# -------------------------------------------------------------------
# Basic preparation
# -------------------------------------------------------------------

print()
print(f"Original dataset shape: {df.shape}")


# Convert target:
#
# yes -> 1
# no  -> 0
#
df["y"] = (
    df["y"]
    .map({
        "yes": 1,
        "no": 0
    })
    .astype(int)
)


# -------------------------------------------------------------------
# Create a smaller stratified sample
# -------------------------------------------------------------------

if len(df) > SAMPLE_SIZE:

    df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["y"]
    )

    df = df.reset_index(drop=True)


# -------------------------------------------------------------------
# Save prepared dataset
# -------------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

print()
print("Dataset preparation complete.")
print(f"Final dataset shape: {df.shape}")
print(f"Saved to: {OUTPUT_FILE}")

print()
print("Target distribution:")

print(
    df["y"]
    .value_counts(normalize=True)
    .rename("proportion")
)

print()
print("Columns:")

for column in df.columns:
    print(f"  - {column}")