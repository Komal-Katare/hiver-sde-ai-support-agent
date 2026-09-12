import pandas as pd
from pathlib import Path


INPUT_PATH = Path("evaluation/reply_eval_outputs.csv")
OUTPUT_PATH = Path("results/failure_analysis.txt")


df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("FAILURE MODE ANALYSIS")
print("=" * 70)

print(f"\nTotal cases: {len(df)}")


# ---------------------------------------------------------
# 1. Intent classification failures
# ---------------------------------------------------------

intent_failures = df[
    df["gold_intent"].astype(str).str.strip()
    != df["predicted_intent"].astype(str).str.strip()
].copy()

print("\n" + "-" * 70)
print("INTENT CLASSIFICATION FAILURES")
print("-" * 70)

print(f"Failures: {len(intent_failures)} / {len(df)}")

for _, row in intent_failures.iterrows():
    print("\nCase:", row["customer_tweet_id"])
    print("Customer:", row["customer_message"])
    print("Gold intent:", row["gold_intent"])
    print("Predicted:", row["predicted_intent"])
    print("Confidence:", round(float(row["classifier_confidence"]), 3))


# ---------------------------------------------------------
# 2. Low retrieval similarity
# ---------------------------------------------------------

low_retrieval = df[
    df["top_similarity"].astype(float) < 0.35
].copy()

print("\n" + "-" * 70)
print("LOW RETRIEVAL SIMILARITY")
print("-" * 70)

print(f"Cases with similarity < 0.35: {len(low_retrieval)}")

for _, row in low_retrieval.iterrows():
    print("\nCase:", row["customer_tweet_id"])
    print("Customer:", row["customer_message"])
    print("Similarity:", round(float(row["top_similarity"]), 3))
    print("Decision:", row["decision"])


# ---------------------------------------------------------
# 3. Escalations
# ---------------------------------------------------------

escalations = df[
    df["decision"].astype(str).str.upper() == "ESCALATE"
].copy()

print("\n" + "-" * 70)
print("ESCALATIONS")
print("-" * 70)

print(f"Escalated: {len(escalations)} / {len(df)}")

for _, row in escalations.iterrows():
    print("\nCase:", row["customer_tweet_id"])
    print("Intent:", row["gold_intent"], "->", row["predicted_intent"])
    print("Confidence:", round(float(row["classifier_confidence"]), 3))
    print("Similarity:", round(float(row["top_similarity"]), 3))
    print("Reason:", row["decision_reason"])


# ---------------------------------------------------------
# 4. Potentially weak retrieval
# ---------------------------------------------------------

weak_retrieval = df[
    df["top_similarity"].astype(float) < 0.45
].copy()

print("\n" + "-" * 70)
print("WEAK RETRIEVAL (< 0.45)")
print("-" * 70)

print(f"Cases: {len(weak_retrieval)}")


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    f.write("FAILURE MODE ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Total cases: {len(df)}\n\n")

    f.write("INTENT CLASSIFICATION FAILURES\n")
    f.write("-" * 70 + "\n")
    f.write(f"Failures: {len(intent_failures)} / {len(df)}\n\n")

    for _, row in intent_failures.iterrows():
        f.write(f"Case: {row['customer_tweet_id']}\n")
        f.write(f"Customer: {row['customer_message']}\n")
        f.write(f"Gold intent: {row['gold_intent']}\n")
        f.write(f"Predicted: {row['predicted_intent']}\n")
        f.write(
            f"Confidence: "
            f"{float(row['classifier_confidence']):.3f}\n\n"
        )

    f.write("\nLOW RETRIEVAL SIMILARITY\n")
    f.write("-" * 70 + "\n")
    f.write(f"Cases with similarity < 0.35: {len(low_retrieval)}\n\n")

    for _, row in low_retrieval.iterrows():
        f.write(f"Case: {row['customer_tweet_id']}\n")
        f.write(f"Customer: {row['customer_message']}\n")
        f.write(
            f"Similarity: "
            f"{float(row['top_similarity']):.3f}\n"
        )
        f.write(f"Decision: {row['decision']}\n\n")

    f.write("\nESCALATIONS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Escalated: {len(escalations)} / {len(df)}\n\n")

    for _, row in escalations.iterrows():
        f.write(f"Case: {row['customer_tweet_id']}\n")
        f.write(
            f"Intent: {row['gold_intent']} -> "
            f"{row['predicted_intent']}\n"
        )
        f.write(
            f"Confidence: "
            f"{float(row['classifier_confidence']):.3f}\n"
        )
        f.write(
            f"Similarity: "
            f"{float(row['top_similarity']):.3f}\n"
        )
        f.write(f"Reason: {row['decision_reason']}\n\n")

print("\nSaved:")
print(OUTPUT_PATH)