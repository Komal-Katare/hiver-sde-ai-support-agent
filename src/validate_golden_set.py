import pandas as pd

PATH = "evaluation/golden_set.csv"

VALID_INTENTS = {
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
}


def main():
    df = pd.read_csv(PATH)

    print("=" * 70)
    print("GOLDEN SET VALIDATION")
    print("=" * 70)

    print(f"\nTotal rows in file: {len(df):,}")

    labelled = df[df["intent"].notna() & (df["intent"].astype(str).str.strip() != "")].copy()

    print(f"Labelled rows: {len(labelled):,}")
    print(f"Unlabelled rows: {len(df) - len(labelled):,}")

    # Check invalid labels
    invalid = labelled[
        ~labelled["intent"].astype(str).str.strip().isin(VALID_INTENTS)
    ]

    print(f"\nInvalid labels: {len(invalid):,}")

    if len(invalid) > 0:
        print("\nInvalid labels found:")
        print(invalid["intent"].value_counts().to_string())

    print("\nIntent distribution:")
    print(labelled["intent"].value_counts().to_string())

    print("\nMissing customer messages:")
    print(labelled["customer_message"].isna().sum())

    print("\nMissing agent responses:")
    print(labelled["agent_response"].isna().sum())

    if len(labelled) == 200 and len(invalid) == 0:
        print("\n" + "=" * 70)
        print("✓ GOLDEN SET VALIDATION PASSED")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠ PLEASE CHECK THE GOLDEN SET")
        print("=" * 70)


if __name__ == "__main__":
    main()