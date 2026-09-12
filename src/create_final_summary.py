from pathlib import Path
import pandas as pd


OUTPUT_PATH = Path("results/final_evaluation_summary.txt")


# ---------------------------------------------------------
# Classifier metrics
# ---------------------------------------------------------

intent_metrics = pd.read_csv(
    "results/intent_metrics.csv"
)

metrics = dict(
    zip(
        intent_metrics["metric"],
        intent_metrics["value"]
    )
)

accuracy = float(metrics["accuracy"])
macro_precision = float(metrics["macro_precision"])
macro_recall = float(metrics["macro_recall"])
macro_f1 = float(metrics["macro_f1"])


# ---------------------------------------------------------
# Majority baseline
# ---------------------------------------------------------

baseline_text = Path(
    "results/baseline_results.txt"
).read_text(encoding="utf-8")


# ---------------------------------------------------------
# Reply evaluation
# ---------------------------------------------------------

reply_eval = pd.read_csv(
    "evaluation/reply_eval_outputs.csv"
)

total_reply_cases = len(reply_eval)

escalated = (
    reply_eval["decision"]
    .astype(str)
    .str.upper()
    .eq("ESCALATE")
    .sum()
)

auto_handled = total_reply_cases - escalated

avg_confidence = reply_eval[
    "classifier_confidence"
].mean()

avg_similarity = reply_eval[
    "top_similarity"
].mean()


# ---------------------------------------------------------
# LLM judge
# ---------------------------------------------------------

judge = pd.read_csv(
    "evaluation/llm_judge_results.csv"
)

judge_columns = {
    "Groundedness": "judge_groundedness",
    "Relevance": "judge_relevance",
    "Helpfulness": "judge_helpfulness",
    "Tone": "judge_tone",
    "No unsupported claims": "judge_no_unsupported_claims",
    "Overall": "judge_overall"
}

judge_averages = {}

for name, column in judge_columns.items():
    judge_averages[name] = judge[column].mean()


# ---------------------------------------------------------
# Judge agreement
# ---------------------------------------------------------

agreement_text = Path(
    "evaluation/judge_agreement_results.txt"
).read_text(encoding="utf-8")


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("=" * 70)
print("FINAL EVALUATION SUMMARY")
print("=" * 70)


print("\nCLASSIFIER")
print("-" * 70)

print(f"Accuracy        : {accuracy:.4f}")
print(f"Macro Precision : {macro_precision:.4f}")
print(f"Macro Recall    : {macro_recall:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")


print("\nMAJORITY BASELINE")
print("-" * 70)
print(baseline_text)


print("\nREPLY SYSTEM")
print("-" * 70)

print(f"Reply cases     : {total_reply_cases}")
print(f"Auto-handled    : {auto_handled}")
print(f"Escalated       : {escalated}")
print(
    f"Escalation rate : "
    f"{escalated / total_reply_cases:.4f}"
)
print(f"Avg confidence  : {avg_confidence:.4f}")
print(f"Avg similarity  : {avg_similarity:.4f}")


print("\nLLM JUDGE")
print("-" * 70)

for name, value in judge_averages.items():
    print(f"{name:25s}: {value:.2f}/5")


print("\nJUDGE AGREEMENT")
print("-" * 70)

print(agreement_text)


# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("FINAL EVALUATION SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("CLASSIFIER\n")
    f.write("-" * 70 + "\n")

    f.write(f"Accuracy        : {accuracy:.4f}\n")
    f.write(f"Macro Precision : {macro_precision:.4f}\n")
    f.write(f"Macro Recall    : {macro_recall:.4f}\n")
    f.write(f"Macro F1        : {macro_f1:.4f}\n\n")

    f.write("MAJORITY BASELINE\n")
    f.write("-" * 70 + "\n")
    f.write(baseline_text)
    f.write("\n\n")

    f.write("REPLY SYSTEM\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Reply cases     : {total_reply_cases}\n"
    )
    f.write(
        f"Auto-handled    : {auto_handled}\n"
    )
    f.write(
        f"Escalated       : {escalated}\n"
    )
    f.write(
        f"Escalation rate : "
        f"{escalated / total_reply_cases:.4f}\n"
    )
    f.write(
        f"Avg confidence  : "
        f"{avg_confidence:.4f}\n"
    )
    f.write(
        f"Avg similarity  : "
        f"{avg_similarity:.4f}\n\n"
    )

    f.write("LLM JUDGE\n")
    f.write("-" * 70 + "\n")

    for name, value in judge_averages.items():
        f.write(
            f"{name:25s}: {value:.2f}/5\n"
        )

    f.write("\nJUDGE AGREEMENT\n")
    f.write("-" * 70 + "\n")

    f.write(agreement_text)


print("\nSaved:")
print(OUTPUT_PATH)