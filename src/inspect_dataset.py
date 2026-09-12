import pandas as pd
from collections import Counter

DATASET_PATH = "data/raw/twcs/twcs.csv"

print("=" * 70)
print("HIVER SDE ASSIGNMENT - DATASET INSPECTION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Inspect columns
# ---------------------------------------------------------

print("\n[1] Reading dataset columns...")

sample = pd.read_csv(
    DATASET_PATH,
    nrows=5
)

print("\nColumns:")
for column in sample.columns:
    print(f"  - {column}")

print("\nSample rows:")
print(sample.to_string(index=False))


# ---------------------------------------------------------
# 2. Count rows and inspect brands
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("[2] Scanning dataset in chunks")
print("=" * 70)

chunk_size = 100_000

total_rows = 0
inbound_count = 0
outbound_count = 0

author_counts = Counter()

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATASET_PATH,
        chunksize=chunk_size
    ),
    start=1
):

    total_rows += len(chunk)

    # Count inbound/outbound tweets
    if "inbound" in chunk.columns:
        inbound_count += int(chunk["inbound"].sum())
        outbound_count += int((~chunk["inbound"]).sum())

    # Count authors
    if "author_id" in chunk.columns:
        author_counts.update(
            chunk["author_id"].dropna().astype(str)
        )

    if chunk_number % 10 == 0:
        print(
            f"Processed {total_rows:,} rows..."
        )


# ---------------------------------------------------------
# 3. Dataset statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("[3] DATASET STATISTICS")
print("=" * 70)

print(f"\nTotal rows: {total_rows:,}")

if "inbound" in sample.columns:
    print(f"Inbound tweets:  {inbound_count:,}")
    print(f"Outbound tweets: {outbound_count:,}")

print("\nTop 30 authors/brands:")

for author, count in author_counts.most_common(30):
    print(f"{author:30s} {count:,}")


# ---------------------------------------------------------
# 4. Check AppleSupport
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("[4] APPLESUPPORT CHECK")
print("=" * 70)

apple_variants = [
    "AppleSupport",
    "@AppleSupport",
    "applesupport",
    "@applesupport"
]

for variant in apple_variants:
    count = author_counts.get(variant, 0)

    if count:
        print(
            f"FOUND: {variant} -> {count:,} tweets"
        )

print("\nInspection complete.")