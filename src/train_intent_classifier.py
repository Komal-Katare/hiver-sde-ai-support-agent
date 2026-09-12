import os
import re
import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


INPUT_PATH = "data/processed/applesupport_clean.csv"
GOLDEN_PATH = Path("evaluation/golden_set_labelled.csv")

MODEL_DIR = "results"
MODEL_PATH = "results/intent_classifier.joblib"


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


def weak_label(text):
    text = str(text).lower()

    # Account / security
    if any(x in text for x in [
        "activation lock",
        "apple id",
        "appleid",
        "password",
        "phishing",
        "suspicious email",
        "account locked",
        "login"
    ]):
        return "account_security"

    # Battery / power
    if any(x in text for x in [
        "battery",
        "battery life",
        "battery drain",
        "draining",
        "overheating",
        "overheat",
        "gets hot",
        "shuts down",
        "shutdown"
    ]):
        return "battery_power"

    # Connectivity
    if any(x in text for x in [
        "wifi",
        "wi-fi",
        "bluetooth",
        "internet",
        "network",
        "cellular",
        "mobile data",
        "connection"
    ]):
        return "connectivity"

    # Data / storage
    if any(x in text for x in [
        "icloud",
        "storage",
        "memory",
        "photos disappeared",
        "photos missing",
        "pictures missing",
        "messages disappeared",
        "data lost",
        "deleted photos"
    ]):
        return "data_storage"

    # Hardware
    if any(x in text for x in [
        "screen",
        "touch screen",
        "touchscreen",
        "fingerprint",
        "finger print",
        "home button",
        "camera",
        "speaker",
        "microphone",
        "charging port",
        "not charging",
        "broken iphone",
        "broken ipad"
    ]):
        return "device_hardware"

    # App-related
    if any(x in text for x in [
        "app store",
        "app won't",
        "app wont",
        "apps won't",
        "apps wont",
        "application",
        "safari",
        "itunes",
        "imessage",
        "apple music"
    ]):
        return "app_issue"

    # Orders / repair / warranty
    if any(x in text for x in [
        "order",
        "ordered",
        "delivery",
        "delivered",
        "shipping",
        "shipped",
        "warranty",
        "repair",
        "replacement",
        "replace my iphone",
        "pre-order",
        "preorder"
    ]):
        return "orders_repair_warranty"

    # How-to
    if any(x in text for x in [
        "how do i",
        "how can i",
        "how to",
        "can i turn",
        "where can i",
        "how do you",
        "is there a way",
        "turn off",
        "turn on",
        "enable",
        "disable",
        "set up"
    ]):
        return "settings_howto"

    # iOS/update bugs
    if any(x in text for x in [
        "ios",
        "ios 11",
        "ios11",
        "update",
        "updated",
        "updating",
        "upgrade",
        "upgraded",
        "latest update",
        "after updating",
        "since updating",
        "macos"
    ]):
        return "ios_update_bug"

    return "other"


def main():

    print("=" * 70)
    print("TRAINING INTENT CLASSIFIER")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nHistorical conversations: {len(df):,}")

    # ---------------------------------------------------------
    # IMPORTANT: Remove golden-set examples from training
    # to prevent evaluation leakage.
    # ---------------------------------------------------------
    golden = pd.read_csv(GOLDEN_PATH)

# Use the actual customer messages to identify golden examples.
# The tweet IDs in golden_set.csv were not preserved consistently.
    golden_messages = set(
    golden.loc[
        golden["intent"].notna(),
        "customer_message"
    ].astype(str).str.strip()
)

    before_exclusion = len(df)

    df = df[
    ~df["customer_message"].astype(str).str.strip().isin(golden_messages)
    ].copy()

    excluded = before_exclusion - len(df)

    print(f"\nGolden-set examples excluded: {excluded:,}")
    print(f"Training pool after exclusion: {len(df):,}")

    # ---------------------------------------------------------
    # Generate weak labels ONLY on the non-golden data
    # ---------------------------------------------------------
    df["weak_intent"] = df["customer_message"].apply(weak_label)

    print("\nWeak-label distribution:")
    print(df["weak_intent"].value_counts().to_string())

    # Keep only known intent categories
    train_df = df[
        df["weak_intent"].isin(INTENTS)
    ].copy()

    X = train_df["customer_message"]
    y = train_df["weak_intent"]

    print(f"\nTraining examples: {len(train_df):,}")

    # ---------------------------------------------------------
    # TF-IDF + Logistic Regression
    # ---------------------------------------------------------
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                sublinear_tf=True,
                max_features=100000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    print("\nTraining model...")

    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nModel saved to:")
    print(MODEL_PATH)

    print("\nClasses:")
    for c in model.classes_:
        print(f" - {c}")


if __name__ == "__main__":
    main()