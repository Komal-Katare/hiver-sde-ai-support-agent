import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from escalation import decide_handling


INPUT_PATH = "evaluation/reply_eval_set.csv"
OUTPUT_PATH = "evaluation/reply_eval_outputs.csv"

MODEL_PATH = "results/intent_classifier.joblib"
VECTORIZER_PATH = "results/retrieval_vectorizer.joblib"
MATRIX_PATH = "results/retrieval_matrix.joblib"
DATA_PATH = "results/retrieval_data.joblib"

LLM_MODEL = "llama3.2:3b"


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

classifier = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
retrieval_matrix = joblib.load(MATRIX_PATH)
retrieval_data = joblib.load(DATA_PATH)


# ---------------------------------------------------------
# Classification
# ---------------------------------------------------------

def classify_intent(message):

    prediction = classifier.predict([message])[0]

    probabilities = classifier.predict_proba([message])[0]

    confidence = probabilities.max()

    return prediction, confidence


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_examples(message, top_k=5):

    query_vector = vectorizer.transform([message])

    scores = cosine_similarity(
        query_vector,
        retrieval_matrix
    ).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    results = retrieval_data.iloc[top_indices].copy()

    results["similarity"] = scores[top_indices]

    return results


# ---------------------------------------------------------
# LLM generation
# ---------------------------------------------------------

def generate_reply(customer_message, intent, confidence, examples):

    evidence = ""

    for i, (_, row) in enumerate(examples.iterrows(), 1):

        evidence += f"""
HISTORICAL EXAMPLE {i}

Customer:
{row["customer_message"]}

Historical Apple Support response:
{row["agent_response"]}

Similarity:
{row["similarity"]:.4f}

"""

    prompt = f"""
You are an AI customer-support drafting assistant.

Draft a concise and professional response to the customer.

Use the historical Apple Support examples below as your
primary evidence.

Do NOT invent:
- refunds
- warranty policies
- company policies
- guarantees
- timelines
- unsupported troubleshooting steps

If the historical examples do not provide enough information
to safely solve the issue, ask a useful clarifying question
or suggest continuing the conversation through Direct Message.

Customer message:
{customer_message}

Predicted intent:
{intent}

Classifier confidence:
{confidence:.3f}

Historical examples:
{evidence}

Write ONLY the customer-facing support response.

Do not mention:
- classifier
- historical examples
- similarity scores
- AI
"""

    import ollama

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("RUNNING REPLY EVALUATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nEvaluation cases: {len(df)}")

    outputs = []

    for i, row in df.iterrows():

        message = row["customer_message"]

        print(
            f"\n[{i + 1}/{len(df)}] "
            f"Generating response..."
        )

        # Classify
        intent, confidence = classify_intent(message)

        # Retrieve
        examples = retrieve_examples(
            message,
            top_k=5
        )

        top_similarity = float(
            examples.iloc[0]["similarity"]
        )

        # Decision
        decision, reason = decide_handling(
            intent,
            confidence,
            top_similarity
        )

        # Generate
        reply = generate_reply(
            message,
            intent,
            confidence,
            examples
        )

        outputs.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_message": message,
            "gold_intent": row["intent"],
            "predicted_intent": intent,
            "classifier_confidence": confidence,
            "top_similarity": top_similarity,
            "decision": decision,
            "decision_reason": reason,
            "historical_agent_response": row["agent_response"],
            "generated_reply": reply
        })

    output_df = pd.DataFrame(outputs)

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("REPLY EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()