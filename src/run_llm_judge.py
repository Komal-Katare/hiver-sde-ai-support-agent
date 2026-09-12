import json
import re
import time
import pandas as pd
import ollama


INPUT_PATH = "evaluation/human_review.csv"
OUTPUT_PATH = "evaluation/llm_judge_results.csv"

LLM_MODEL = "llama3.2:3b"


def extract_json(text):
    """
    Extract the first JSON object from the model response.
    """
    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # JSON inside markdown/code/text
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return None


def clamp_score(value):
    try:
        value = int(value)
        return max(1, min(5, value))
    except:
        return None


def judge_reply(customer_message, historical_response, generated_reply):

    prompt = f"""
You are evaluating an AI customer-support reply.

Evaluate ONLY the generated reply against the customer message and
the historical support response provided as evidence.

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL SUPPORT RESPONSE:
{historical_response}

GENERATED AI REPLY:
{generated_reply}

Score the generated reply on these five dimensions from 1 to 5.

1. groundedness
5 = clearly supported by the historical support evidence
4 = mostly supported, with minor extrapolation
3 = partially grounded
2 = weakly grounded
1 = unsupported or hallucinated

2. relevance
5 = directly addresses the customer's problem
4 = mostly relevant
3 = somewhat relevant
2 = mostly off-target
1 = does not address the problem

3. helpfulness
5 = gives a useful next step or excellent clarification
4 = helpful but could be more specific
3 = somewhat useful
2 = little practical value
1 = not useful

4. tone
5 = professional, empathetic and natural support tone
4 = good tone with minor awkwardness
3 = acceptable but generic
2 = noticeably poor tone
1 = inappropriate or unprofessional

5. no_unsupported_claims
5 = no unsupported factual, policy, troubleshooting or guarantee claims
4 = very minor questionable inference
3 = some questionable claim
2 = significant unsupported claim
1 = clear hallucination or unsupported claim

Return ONLY valid JSON in exactly this format:

{{
  "groundedness": 1,
  "relevance": 1,
  "helpfulness": 1,
  "tone": 1,
  "no_unsupported_claims": 1
}}
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        content = response["message"]["content"]

        result = extract_json(content)

        if result is None:
            print("WARNING: Could not parse JSON:")
            print(content)
            return {
                "groundedness": None,
                "relevance": None,
                "helpfulness": None,
                "tone": None,
                "no_unsupported_claims": None
            }

        return {
            "groundedness": clamp_score(result.get("groundedness")),
            "relevance": clamp_score(result.get("relevance")),
            "helpfulness": clamp_score(result.get("helpfulness")),
            "tone": clamp_score(result.get("tone")),
            "no_unsupported_claims": clamp_score(
                result.get("no_unsupported_claims")
            )
        }

    except Exception as e:

        print(f"ERROR: {e}")

        return {
            "groundedness": None,
            "relevance": None,
            "helpfulness": None,
            "tone": None,
            "no_unsupported_claims": None
        }


def main():

    print("=" * 70)
    print("RUNNING LLM-AS-JUDGE")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nEvaluation cases: {len(df)}")
    print(f"Judge model: {LLM_MODEL}")

    results = []

    for i, row in df.iterrows():

        print(f"\n[{i + 1}/{len(df)}] Judging response...")

        scores = judge_reply(
            customer_message=str(row["customer_message"]),
            historical_response=str(row["historical_agent_response"]),
            generated_reply=str(row["generated_reply"])
        )

        results.append(scores)

        # Small pause to keep local inference stable
        time.sleep(0.2)

    scores_df = pd.DataFrame(results)

    output = df[
        [
            "customer_tweet_id",
            "customer_message",
            "gold_intent",
            "predicted_intent",
            "classifier_confidence",
            "top_similarity",
            "decision",
            "generated_reply"
        ]
    ].copy()

    output["judge_groundedness"] = scores_df["groundedness"]
    output["judge_relevance"] = scores_df["relevance"]
    output["judge_helpfulness"] = scores_df["helpfulness"]
    output["judge_tone"] = scores_df["tone"]
    output["judge_no_unsupported_claims"] = (
        scores_df["no_unsupported_claims"]
    )

    output["judge_overall"] = output[
        [
            "judge_groundedness",
            "judge_relevance",
            "judge_helpfulness",
            "judge_tone",
            "judge_no_unsupported_claims"
        ]
    ].mean(axis=1)

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("LLM-AS-JUDGE COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nAverage judge scores:")

    score_columns = [
        "judge_groundedness",
        "judge_relevance",
        "judge_helpfulness",
        "judge_tone",
        "judge_no_unsupported_claims",
        "judge_overall"
    ]

    for column in score_columns:
        print(
            f"{column}: "
            f"{output[column].mean():.2f}"
        )


if __name__ == "__main__":
    main()