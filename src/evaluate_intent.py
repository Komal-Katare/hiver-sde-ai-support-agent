import os
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


MODEL_PATH = "results/intent_classifier.joblib"
GOLDEN_PATH = Path("evaluation/golden_set_labelled.csv")

RESULTS_PATH = "results/intent_predictions.csv"
METRICS_PATH = "results/intent_metrics.csv"


def main():

    print("=" * 70)
    print("EVALUATING INTENT CLASSIFIER")
    print("=" * 70)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load golden set
    df = pd.read_csv(GOLDEN_PATH)

    # Keep only human-labelled rows
    df = df[
        df["intent"].notna() &
        (df["intent"].astype(str).str.strip() != "")
    ].copy()

    print(f"\nGolden examples: {len(df):,}")

    X = df["customer_message"]
    y_true = df["intent"].astype(str).str.strip()

    # Predict
    y_pred = model.predict(X)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("HEADLINE METRICS")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1        : {f1:.4f}")

    print("\n" + "=" * 70)
    print("PER-INTENT PERFORMANCE")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    # Confusion matrix
    labels = sorted(y_true.unique())

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print(cm_df.to_string())

    # Save predictions
    output = df[
        [
            "customer_tweet_id",
            "customer_message",
            "intent"
        ]
    ].copy()

    output["predicted_intent"] = y_pred
    output["correct"] = output["intent"] == output["predicted_intent"]

    os.makedirs("results", exist_ok=True)

    output.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Save metrics
    metrics = pd.DataFrame({
        "metric": [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1"
        ],
        "value": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    metrics.to_csv(
        METRICS_PATH,
        index=False
    )

    print("\nResults saved:")
    print(RESULTS_PATH)
    print(METRICS_PATH)


if __name__ == "__main__":
    main()
