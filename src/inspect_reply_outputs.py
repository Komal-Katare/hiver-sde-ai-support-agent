import pandas as pd


INPUT_PATH = "evaluation/reply_eval_outputs.csv"


def main():

    print("=" * 70)
    print("INSPECTING GENERATED REPLIES")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nEvaluation cases: {len(df)}")

    print("\nDecision distribution:")
    print(df["decision"].value_counts().to_string())

    print("\nPredicted intent distribution:")
    print(df["predicted_intent"].value_counts().to_string())

    print("\nAverage classifier confidence:")
    print(
        df["classifier_confidence"]
        .astype(float)
        .mean()
    )

    print("\nAverage retrieval similarity:")
    print(
        df["top_similarity"]
        .astype(float)
        .mean()
    )

    print("\n" + "=" * 70)
    print("GENERATED RESPONSES")
    print("=" * 70)

    for i, row in df.iterrows():

        print("\n" + "-" * 70)
        print(f"CASE {i + 1}")

        print("\nCUSTOMER:")
        print(row["customer_message"])

        print("\nGOLD INTENT:")
        print(row["gold_intent"])

        print("\nPREDICTED INTENT:")
        print(row["predicted_intent"])

        print(
            f"\nCONFIDENCE: "
            f"{float(row['classifier_confidence']):.3f}"
        )

        print(
            f"RETRIEVAL SIMILARITY: "
            f"{float(row['top_similarity']):.3f}"
        )

        print("\nDECISION:")
        print(row["decision"])

        print("\nGENERATED REPLY:")
        print(row["generated_reply"])

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()