import os
import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_PATH = "data/processed/applesupport_clean.csv"
GOLDEN_PATH = Path("evaluation/golden_set_labelled.csv")

VECTORIZER_PATH = "results/retrieval_vectorizer.joblib"
MATRIX_PATH = "results/retrieval_matrix.joblib"
DATA_PATH = "results/retrieval_data.joblib"


def main():

    print("=" * 70)
    print("BUILDING HISTORICAL RETRIEVAL INDEX")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nHistorical conversations: {len(df):,}")

    # ---------------------------------------------------------
    # IMPORTANT: Remove golden-set examples from retrieval
    # to prevent evaluation leakage.
    #
    # We use customer_message because tweet IDs in the golden
    # set were not preserved consistently.
    # ---------------------------------------------------------
    golden = pd.read_csv(GOLDEN_PATH)

    golden_messages = set(
        golden.loc[
            golden["intent"].notna(),
            "customer_message"
        ].astype(str).str.strip()
    )

    before_exclusion = len(df)

    df = df[
        ~df["customer_message"]
        .astype(str)
        .str.strip()
        .isin(golden_messages)
    ].copy()

    excluded = before_exclusion - len(df)

    print(f"\nGolden-set examples excluded: {excluded:,}")
    print(f"Retrieval pool after exclusion: {len(df):,}")

    # ---------------------------------------------------------
    # Keep only fields required for retrieval
    # ---------------------------------------------------------
    retrieval_df = df[
        [
            "customer_tweet_id",
            "customer_message",
            "agent_response"
        ]
    ].copy()

    # ---------------------------------------------------------
    # TF-IDF representation
    # ---------------------------------------------------------
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        max_features=100000
    )

    print("\nBuilding TF-IDF matrix...")

    matrix = vectorizer.fit_transform(
        retrieval_df["customer_message"]
    )

    os.makedirs("results", exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(matrix, MATRIX_PATH)
    joblib.dump(retrieval_df, DATA_PATH)

    print("\n" + "=" * 70)
    print("RETRIEVAL INDEX COMPLETE")
    print("=" * 70)

    print(f"\nMatrix shape: {matrix.shape}")
    print(f"Vectorizer: {VECTORIZER_PATH}")
    print(f"Matrix: {MATRIX_PATH}")
    print(f"Data: {DATA_PATH}")


if __name__ == "__main__":
    main()