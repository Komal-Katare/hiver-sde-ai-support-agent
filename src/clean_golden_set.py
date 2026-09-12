import pandas as pd
from pathlib import Path


INPUT_PATH = Path("evaluation/golden_set.csv")
OUTPUT_PATH = Path("evaluation/golden_set_labelled.csv")


# Load golden set
df = pd.read_csv(INPUT_PATH)

# Keep only manually labelled examples
labelled = df[df["intent"].notna()].copy()

# Clean intent values
labelled["intent"] = labelled["intent"].astype(str).str.strip()

# Remove accidental duplicate customer messages
labelled = labelled.drop_duplicates(
    subset=["customer_message"],
    keep="first"
).reset_index(drop=True)


# Validate
print("=" * 70)
print("CLEAN GOLDEN SET")
print("=" * 70)

print(f"\nOriginal rows       : {len(df)}")
print(f"Labelled rows       : {len(df[df['intent'].notna()])}")
print(f"Final labelled rows: {len(labelled)}")
print(f"Duplicate messages removed: {len(df[df['intent'].notna()]) - len(labelled)}")

print("\nIntent distribution:")
print(labelled["intent"].value_counts().to_string())


# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
labelled.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:")
print(OUTPUT_PATH)