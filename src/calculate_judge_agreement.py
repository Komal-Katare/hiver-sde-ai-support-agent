import os
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


HUMAN_PATH = "evaluation/human_review.csv"
JUDGE_PATH = "evaluation/llm_judge_results.csv"
OUTPUT_PATH = "evaluation/judge_agreement_results.txt"


HUMAN_COLUMNS = [
    "human_groundedness",
    "human_relevance",
    "human_helpfulness",
    "human_tone",
    "human_no_unsupported_claims"
]

JUDGE_COLUMNS = [
    "judge_groundedness",
    "judge_relevance",
    "judge_helpfulness",
    "judge_tone",
    "judge_no_unsupported_claims"
]


def main():

    print("=" * 70)
    print("CALCULATING HUMAN vs LLM JUDGE AGREEMENT")
    print("=" * 70)

    human = pd.read_csv(HUMAN_PATH)
    judge = pd.read_csv(JUDGE_PATH)

    print(f"\nHuman cases: {len(human)}")
    print(f"Judge cases: {len(judge)}")

    results = []

    for human_col, judge_col in zip(
        HUMAN_COLUMNS,
        JUDGE_COLUMNS
    ):

        h = pd.to_numeric(
            human[human_col],
            errors="coerce"
        )

        j = pd.to_numeric(
            judge[judge_col],
            errors="coerce"
        )

        valid = h.notna() & j.notna()

        h = h[valid]
        j = j[valid]

        # Spearman correlation
        correlation, p_value = spearmanr(h, j)

        # Exact agreement
        exact_agreement = (h.values == j.values).mean()

        # Pass/fail agreement
        # Score >= 4 = pass
        human_pass = (h >= 4).astype(int)
        judge_pass = (j >= 4).astype(int)

        pass_agreement = (
            human_pass.values == judge_pass.values
        ).mean()

        kappa = cohen_kappa_score(
            human_pass,
            judge_pass
        )

        results.append({
            "criterion": human_col.replace("human_", ""),
            "n": len(h),
            "spearman": correlation,
            "spearman_p": p_value,
            "exact_score_agreement": exact_agreement,
            "pass_fail_agreement": pass_agreement,
            "cohen_kappa": kappa
        })

        print("\n" + "-" * 70)
        print(human_col.replace("human_", "").upper())
        print("-" * 70)

        print(f"Valid cases       : {len(h)}")
        print(f"Spearman          : {correlation:.3f}")
        print(f"Spearman p-value  : {p_value:.4f}")
        print(f"Exact agreement   : {exact_agreement:.3f}")
        print(f"Pass/fail         : {pass_agreement:.3f}")
        print(f"Cohen's kappa     : {kappa:.3f}")

    results_df = pd.DataFrame(results)

    # Overall human score
    human_overall = human[HUMAN_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    ).mean(axis=1)

    # Overall judge score
    judge_overall = judge[JUDGE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    ).mean(axis=1)

    valid = human_overall.notna() & judge_overall.notna()

    overall_corr, overall_p = spearmanr(
        human_overall[valid],
        judge_overall[valid]
    )

    overall_exact = (
        round(human_overall[valid], 1).values ==
        round(judge_overall[valid], 1).values
    ).mean()

    human_overall_pass = (
        human_overall[valid] >= 4
    ).astype(int)

    judge_overall_pass = (
        judge_overall[valid] >= 4
    ).astype(int)

    overall_pass_agreement = (
        human_overall_pass.values ==
        judge_overall_pass.values
    ).mean()

    overall_kappa = cohen_kappa_score(
        human_overall_pass,
        judge_overall_pass
    )

    print("\n" + "=" * 70)
    print("OVERALL AGREEMENT")
    print("=" * 70)

    print(f"\nSpearman correlation : {overall_corr:.3f}")
    print(f"Spearman p-value    : {overall_p:.4f}")
    print(f"Exact agreement     : {overall_exact:.3f}")
    print(f"Pass/fail agreement  : {overall_pass_agreement:.3f}")
    print(f"Cohen's kappa       : {overall_kappa:.3f}")

    os.makedirs("evaluation", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        f.write(
            "HUMAN vs LLM JUDGE AGREEMENT\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write(
            results_df.to_string(index=False)
        )

        f.write("\n\n")
        f.write("OVERALL AGREEMENT\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"Spearman correlation: {overall_corr:.3f}\n"
        )

        f.write(
            f"Spearman p-value: {overall_p:.4f}\n"
        )

        f.write(
            f"Exact agreement: {overall_exact:.3f}\n"
        )

        f.write(
            f"Pass/fail agreement: "
            f"{overall_pass_agreement:.3f}\n"
        )

        f.write(
            f"Cohen's kappa: {overall_kappa:.3f}\n"
        )

    results_df.to_csv(
        "evaluation/judge_agreement_metrics.csv",
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print(
        "evaluation/judge_agreement_metrics.csv"
    )


if __name__ == "__main__":
    main()