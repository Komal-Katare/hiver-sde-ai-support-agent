import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


VECTORIZER_PATH = "results/retrieval_vectorizer.joblib"
MATRIX_PATH = "results/retrieval_matrix.joblib"
DATA_PATH = "results/retrieval_data.joblib"


def retrieve(query, top_k=5):

    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)
    df = joblib.load(DATA_PATH)

    # Convert new customer message into TF-IDF
    query_vector = vectorizer.transform([query])

    # Compare against historical customer messages
    scores = cosine_similarity(query_vector, matrix).flatten()

    # Get highest scoring conversations
    top_indices = scores.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]

    return results


def main():

    print("=" * 70)
    print("HISTORICAL SUPPORT RETRIEVER")
    print("=" * 70)

    query = input("\nEnter customer message:\n> ").strip()

    if not query:
        print("No message entered.")
        return

    results = retrieve(query, top_k=5)

    print("\n" + "=" * 70)
    print("TOP HISTORICAL MATCHES")
    print("=" * 70)

    for i, (_, row) in enumerate(results.iterrows(), 1):

        print(f"\n{'-' * 70}")
        print(f"MATCH #{i}")
        print(f"Similarity: {row['similarity']:.4f}")

        print("\nCUSTOMER:")
        print(row["customer_message"])

        print("\nAPPLE SUPPORT:")
        print(row["agent_response"])


if __name__ == "__main__":
    main()