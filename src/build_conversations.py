import os
import pandas as pd


DATASET_PATH = "data/raw/twcs/twcs.csv"
APPLE_PATH = "data/processed/applesupport_tweets.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "applesupport_conversations.csv"
)

CHUNK_SIZE = 100_000


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("BUILDING APPLESUPPORT CUSTOMER → AGENT CONVERSATIONS")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load AppleSupport tweets
    # ---------------------------------------------------------

    print("\n[1] Loading AppleSupport tweets...")

    apple = pd.read_csv(APPLE_PATH)

    print(f"AppleSupport tweets: {len(apple):,}")

    # We need the IDs of tweets that AppleSupport replied to
    parent_ids = (
        apple["in_response_to_tweet_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    print(
        f"Unique parent tweet IDs: {len(parent_ids):,}"
    )

    parent_id_set = set(parent_ids)

    # ---------------------------------------------------------
    # 2. Scan original dataset for customer tweets
    # ---------------------------------------------------------

    print("\n[2] Finding customer tweets...")

    customer_chunks = []

    total_rows = 0
    matched_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATASET_PATH,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        total_rows += len(chunk)

        # Customer tweets that AppleSupport replied to
        matched = chunk[
            (chunk["tweet_id"].isin(parent_id_set))
            & (chunk["inbound"] == True)
        ].copy()

        if not matched.empty:
            customer_chunks.append(matched)
            matched_rows += len(matched)

        if chunk_number % 5 == 0:
            print(
                f"Processed {total_rows:,} rows | "
                f"Matched customer tweets: "
                f"{matched_rows:,}"
            )

    if not customer_chunks:
        raise RuntimeError(
            "No customer tweets matched AppleSupport responses."
        )

    customers = pd.concat(
        customer_chunks,
        ignore_index=True
    )

    print(
        f"\nCustomer tweets found: {len(customers):,}"
    )

    # ---------------------------------------------------------
    # 3. Prepare AppleSupport replies
    # ---------------------------------------------------------

    print("\n[3] Joining customer messages with replies...")

    apple_replies = apple[
    [
        "tweet_id",
        "text",
        "created_at",
        "in_response_to_tweet_id"
    ]
].copy()

    apple_replies = apple_replies.rename(
        columns={
            "tweet_id": "agent_tweet_id",
            "text": "agent_response",
            "created_at": "agent_created_at"
        }
    )

    # ---------------------------------------------------------
    # 4. Join
    # ---------------------------------------------------------

    conversations = customers.merge(
        apple_replies,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        how="inner"
    )

    # ---------------------------------------------------------
    # 5. Rename important columns
    # ---------------------------------------------------------

    conversations = conversations.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_id",
            "text": "customer_message",
            "created_at": "customer_created_at"
        }
    )

    # Keep useful columns only
    conversations = conversations[
        [
            "customer_tweet_id",
            "customer_id",
            "customer_message",
            "customer_created_at",
            "agent_tweet_id",
            "agent_response",
            "agent_created_at"
        ]
    ]

    # ---------------------------------------------------------
    # 6. Basic cleaning
    # ---------------------------------------------------------

    conversations = conversations.dropna(
        subset=[
            "customer_message",
            "agent_response"
        ]
    )

    conversations["customer_message"] = (
        conversations["customer_message"]
        .astype(str)
        .str.strip()
    )

    conversations["agent_response"] = (
        conversations["agent_response"]
        .astype(str)
        .str.strip()
    )

    conversations = conversations[
        (conversations["customer_message"].str.len() > 0)
        &
        (conversations["agent_response"].str.len() > 0)
    ]

    # Remove duplicate pairs
    conversations = conversations.drop_duplicates(
        subset=[
            "customer_tweet_id",
            "agent_tweet_id"
        ]
    )

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------

    conversations.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # 8. Report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONVERSATION BUILD COMPLETE")
    print("=" * 70)

    print(
        f"\nConversation pairs: {len(conversations):,}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print("\nColumns:")
    for column in conversations.columns:
        print(f"  - {column}")

    print("\nSample conversations:\n")

    print(
        conversations[
            [
                "customer_message",
                "agent_response"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()