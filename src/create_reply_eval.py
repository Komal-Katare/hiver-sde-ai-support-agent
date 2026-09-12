import pandas as pd

INPUT_PATH = "evaluation/golden_set.csv"
OUTPUT_PATH = "evaluation/reply_eval_set.csv"


def main():

    print("=" * 70)
    print("CREATING REPLY EVALUATION SET")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    # Only human-labelled examples
    df = df[
        df["intent"].notna() &
        (df["intent"].astype(str).str.strip() != "")
    ].copy()

    # 30 reproducible examples
    sample = df.sample(
        n=30,
        random_state=2026
    ).copy()

    sample = sample[
        [
            "customer_tweet_id",
            "customer_message",
            "agent_response",
            "intent"
        ]
    ]

    sample["generated_reply"] = ""
    sample["retrieval_similarity"] = ""
    sample["classifier_confidence"] = ""
    sample["decision"] = ""
    sample["decision_reason"] = ""

    sample.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nGolden examples available: {len(df)}")
    print(f"Reply evaluation examples: {len(sample)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()