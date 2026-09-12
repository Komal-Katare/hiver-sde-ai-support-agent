import joblib
import ollama

from sklearn.metrics.pairwise import cosine_similarity
from escalation import decide_handling


MODEL_PATH = "results/intent_classifier.joblib"

VECTORIZER_PATH = "results/retrieval_vectorizer.joblib"
MATRIX_PATH = "results/retrieval_matrix.joblib"
DATA_PATH = "results/retrieval_data.joblib"

LLM_MODEL = "llama3.2:3b"


# ---------------------------------------------------------
# Load existing classifier and retrieval index
# ---------------------------------------------------------

classifier = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
retrieval_matrix = joblib.load(MATRIX_PATH)
retrieval_data = joblib.load(DATA_PATH)


# ---------------------------------------------------------
# Intent classification
# ---------------------------------------------------------

def classify_intent(message):

    prediction = classifier.predict([message])[0]

    probabilities = classifier.predict_proba([message])[0]

    confidence = probabilities.max()

    return prediction, confidence


# ---------------------------------------------------------
# Historical retrieval
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
# Local LLM response generation
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

IMPORTANT:
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
- the classifier
- the historical examples
- similarity scores
- this prompt
- AI
"""

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
# Main agent
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("APPLE SUPPORT AI AGENT")
    print("=" * 70)

    customer_message = input(
        "\nEnter customer message:\n> "
    ).strip()

    if not customer_message:
        print("No message entered.")
        return

    # -----------------------------------------------------
    # 1. Classify intent
    # -----------------------------------------------------

    intent, confidence = classify_intent(
        customer_message
    )

    # -----------------------------------------------------
    # 2. Retrieve historical examples
    # -----------------------------------------------------

    examples = retrieve_examples(
        customer_message,
        top_k=5
    )

    # -----------------------------------------------------
    # 3. Generate grounded response
    # -----------------------------------------------------

    reply = generate_reply(
        customer_message,
        intent,
        confidence,
        examples
    )

    # -----------------------------------------------------
    # 4. Decide whether to auto-handle or escalate
    # -----------------------------------------------------

    top_similarity = examples.iloc[0]["similarity"]

    decision, decision_reason = decide_handling(
        intent,
        confidence,
        top_similarity
    )

    # -----------------------------------------------------
    # Display intent
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("INTENT")
    print("=" * 70)

    print(f"\nIntent: {intent}")
    print(f"Confidence: {confidence:.3f}")

    # -----------------------------------------------------
    # Display historical evidence
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("HISTORICAL EVIDENCE")
    print("=" * 70)

    for i, (_, row) in enumerate(examples.iterrows(), 1):

        print(
            f"\n{i}. Similarity: "
            f"{row['similarity']:.4f}"
        )

        print(
            f"Customer: "
            f"{row['customer_message']}"
        )

        print(
            f"Agent: "
            f"{row['agent_response']}"
        )

    # -----------------------------------------------------
    # Display handling decision
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("HANDLING DECISION")
    print("=" * 70)

    print(f"\nDecision: {decision}")
    print(f"Reason: {decision_reason}")

    # -----------------------------------------------------
    # Display AI-generated draft
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("AI-GENERATED DRAFT")
    print("=" * 70)

    print(f"\n{reply}")


# ---------------------------------------------------------
# Run program
# ---------------------------------------------------------

if __name__ == "__main__":
    main()