import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

GOLDEN_PATH = Path("evaluation/golden_set.csv")
OUTPUT_PATH = Path("results/baseline_results.txt")


# ---------------------------------------------------------
# Load golden set
# ---------------------------------------------------------

golden = pd.read_csv(GOLDEN_PATH)

# Keep only manually labelled examples
golden = golden[golden["intent"].notna()].copy()

golden["intent"] = golden["intent"].astype(str).str.strip()

print("=" * 70)
print("BASELINE EVALUATION")
print("=" * 70)

print(f"\nGolden examples: {len(golden)}")


# ---------------------------------------------------------
# Baseline 1: Majority Class
# ---------------------------------------------------------

majority_class = golden["intent"].value_counts().idxmax()

y_true = golden["intent"]
y_pred = [majority_class] * len(golden)

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "-" * 70)
print("BASELINE 1: MAJORITY CLASS")
print("-" * 70)

print(f"Majority intent : {majority_class}")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Macro Precision : {precision:.4f}")
print(f"Macro Recall    : {recall:.4f}")
print(f"Macro F1        : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    f.write("BASELINE EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Golden examples: {len(golden)}\n\n")

    f.write("BASELINE 1: MAJORITY CLASS\n")
    f.write("-" * 70 + "\n")

    f.write(f"Majority intent : {majority_class}\n")
    f.write(f"Accuracy        : {accuracy:.4f}\n")
    f.write(f"Macro Precision : {precision:.4f}\n")
    f.write(f"Macro Recall    : {recall:.4f}\n")
    f.write(f"Macro F1        : {f1:.4f}\n\n")

    f.write("Classification Report:\n")

    f.write(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

print("\nResults saved to:")
print(OUTPUT_PATH)