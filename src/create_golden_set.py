import os
import pandas as pd

INPUT_PATH = "data/processed/applesupport_clean.csv"
OUTPUT_PATH = "evaluation/golden_set.csv"

INTENTS = [
    "ios_update_bug",
    "battery_power",
    "device_hardware",
    "app_issue",
    "connectivity",
    "account_security",
    "data_storage",
    "settings_howto",
    "orders_repair_warranty",
    "other"
]


def main():
    print("=" * 70)
    print("CREATING GOLDEN SET")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    # Reproducible random sample
    sample = df.sample(n=200, random_state=2026).copy()

    # Keep only fields needed for annotation
    golden = sample[
        [
            "customer_tweet_id",
            "customer_message",
            "agent_response"
        ]
    ].copy()

    # Human annotation fields
    golden["intent"] = ""
    golden["annotation_notes"] = ""

    os.makedirs("evaluation", exist_ok=True)

    golden.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSource conversations: {len(df):,}")
    print(f"Golden examples created: {len(golden):,}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nIntents to use:")
    for i, intent in enumerate(INTENTS, 1):
        print(f"{i:2}. {intent}")

    print("\nIMPORTANT:")
    print("Open evaluation/golden_set.csv in Excel.")
    print("Fill the 'intent' column manually.")
    print("Do NOT change customer_message or agent_response.")
    print("\nGolden set creation complete.")


if __name__ == "__main__":
    main()