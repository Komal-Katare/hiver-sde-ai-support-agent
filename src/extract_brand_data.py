import os
import pandas as pd

DATASET_PATH = "data/raw/twcs/twcs.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "applesupport_tweets.csv"
)

BRAND = "AppleSupport"

CHUNK_SIZE = 100_000


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("APPLE SUPPORT DATA EXTRACTION")
    print("=" * 70)

    selected_chunks = []
    total_rows = 0

    print("\nReading dataset in chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATASET_PATH,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        total_rows += len(chunk)

        brand_rows = chunk[
            chunk["author_id"].astype(str) == BRAND
        ].copy()

        if not brand_rows.empty:
            selected_chunks.append(brand_rows)

        if chunk_number % 5 == 0:
            print(
                f"Processed {total_rows:,} rows | "
                f"AppleSupport rows collected: "
                f"{sum(len(x) for x in selected_chunks):,}"
            )

    if not selected_chunks:
        raise RuntimeError(
            f"No rows found for brand: {BRAND}"
        )

    result = pd.concat(
        selected_chunks,
        ignore_index=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"\nBrand: {BRAND}")
    print(f"Rows extracted: {len(result):,}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nInbound / outbound:")
    print(result["inbound"].value_counts())

    print("\nSample AppleSupport tweets:")
    print(
        result[
            ["tweet_id", "inbound", "text"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()