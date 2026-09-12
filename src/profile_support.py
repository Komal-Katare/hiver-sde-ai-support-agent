import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

INPUT_PATH = "data/processed/applesupport_clean.csv"


def main():
    print("=" * 70)
    print("APPLE SUPPORT DATA PROFILE")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nTotal conversations: {len(df):,}")

    # Message length
    df["msg_len"] = df["customer_message"].str.len()

    print("\nCustomer message length:")
    print(df["msg_len"].describe().round(1).to_string())

    # Most common words / phrases
    print("\nExtracting common customer phrases...")

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=100,
        max_features=100
    )

    X = vectorizer.fit_transform(df["customer_message"])

    counts = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    top = sorted(
        zip(terms, counts),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop customer terms/phrases:")
    for term, count in top[:60]:
        print(f"{term:35} {count:,}")

    # Random examples
    print("\n" + "=" * 70)
    print("RANDOM CUSTOMER EXAMPLES")
    print("=" * 70)

    sample = df.sample(100, random_state=42)

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"\n{i}. {row['customer_message']}")

    # Save sample for manual inspection
    output = "data/processed/intent_review_sample.csv"

    review = sample[
        [
            "customer_tweet_id",
            "customer_message",
            "agent_response"
        ]
    ].copy()

    review["intent"] = ""
    review["notes"] = ""

    review.to_csv(output, index=False)

    print("\n" + "=" * 70)
    print(f"Saved manual review sample: {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()