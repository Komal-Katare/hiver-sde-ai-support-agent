import os
import re
import html
import pandas as pd


INPUT_PATH = "data/processed/applesupport_conversations.csv"
OUTPUT_PATH = "data/processed/applesupport_clean.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities such as &gt; and &amp;
    text = html.unescape(text)

    # Replace URLs with a stable token
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " <URL> ",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def main():

    print("=" * 70)
    print("CLEANING APPLESUPPORT CONVERSATIONS")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nOriginal rows: {len(df):,}")

    # Preserve original text
    df["customer_message_original"] = df["customer_message"]
    df["agent_response_original"] = df["agent_response"]

    # Clean both sides
    df["customer_message"] = df["customer_message"].apply(clean_text)
    df["agent_response"] = df["agent_response"].apply(clean_text)

    # Remove empty messages
    before = len(df)

    df = df[
        (df["customer_message"].str.len() > 0)
        &
        (df["agent_response"].str.len() > 0)
    ].copy()

    print(
        f"Removed empty rows: {before - len(df):,}"
    )

    # Remove exact duplicate customer/response pairs
    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "customer_message",
            "agent_response"
        ]
    ).copy()

    print(
        f"Removed duplicate pairs: {before - len(df):,}"
    )

    # Save
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(f"\nFinal rows: {len(df):,}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nSample cleaned conversations:\n")

    print(
        df[
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