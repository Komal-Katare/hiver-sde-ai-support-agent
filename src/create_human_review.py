import os
import pandas as pd


INPUT_PATH = "evaluation/reply_eval_outputs.csv"
OUTPUT_PATH = "evaluation/human_review.csv"


def main():

    print("=" * 70)
    print("CREATING HUMAN REVIEW FILE")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    review = df[
        [
            "customer_tweet_id",
            "customer_message",
            "gold_intent",
            "predicted_intent",
            "classifier_confidence",
            "top_similarity",
            "decision",
            "decision_reason",
            "historical_agent_response",
            "generated_reply"
        ]
    ].copy()

    # Human evaluation scores: 1-5
    review["human_groundedness"] = ""
    review["human_relevance"] = ""
    review["human_helpfulness"] = ""
    review["human_tone"] = ""
    review["human_no_unsupported_claims"] = ""

    # Calculated later
    review["human_overall"] = ""

    os.makedirs("evaluation", exist_ok=True)

    review.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nReview cases: {len(review)}")

    print("\nHuman scoring columns:")
    print(" - human_groundedness")
    print(" - human_relevance")
    print(" - human_helpfulness")
    print(" - human_tone")
    print(" - human_no_unsupported_claims")

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("HUMAN REVIEW FILE CREATED")
    print("=" * 70)


if __name__ == "__main__":
    main()