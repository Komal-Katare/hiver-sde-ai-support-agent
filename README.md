# Hiver SDE Intern — AI Support Agent

## 1. Problem Framing

This project builds a prototype AI customer-support agent for **AppleSupport** using the Customer Support on Twitter dataset.

Given an incoming customer message, the system:

1. Classifies the message into a small set of support intents.
2. Retrieves similar historical AppleSupport customer/agent interactions.
3. Drafts a response grounded in those historical resolutions.
4. Decides whether the response can be auto-handled or should be escalated to a human.

The goal is not to build a production support system, but to test whether a lightweight, reproducible architecture can combine intent classification, historical retrieval, response generation, and risk-aware escalation.

---

# 2. Dataset and Preprocessing

The project uses the Kaggle Customer Support on Twitter dataset.

The original dataset contains approximately 2.8M tweets. The selected brand is **AppleSupport**.

AppleSupport contains approximately 106K direct support interactions after reconstructing customer-to-agent response pairs.

The preprocessing pipeline:

1. Filter tweets belonging to AppleSupport.
2. Identify inbound customer messages.
3. Use `in_response_to_tweet_id` to reconstruct direct customer → AppleSupport response pairs.
4. Normalize URLs and whitespace.
5. Remove empty and duplicate pairs.
6. Exclude golden-set examples from both classifier training and historical retrieval.

Final AppleSupport historical corpus:

**106,333 customer/agent pairs**

The evaluation golden set contains:

**200 manually labelled customer messages**

---

# 3. Intent Taxonomy

The system uses 10 intents derived from recurring support themes in the AppleSupport data.

| Intent | Description |
|---|---|
| `ios_update_bug` | iOS/software update bugs, crashes, freezing or slow performance |
| `battery_power` | Battery drain, overheating, shutdowns and battery life |
| `device_hardware` | Screen, touch, charging and physical hardware problems |
| `app_issue` | Individual app failures and App Store issues |
| `connectivity` | Wi-Fi, Bluetooth, cellular and Internet connectivity |
| `account_security` | Apple ID, password, activation lock and security |
| `data_storage` | Missing data, photos, messages, storage and iCloud |
| `settings_howto` | Settings, configuration and how-to questions |
| `orders_repair_warranty` | Orders, delivery, repairs, replacements and warranty |
| `other` | Messages outside the above categories |

For multi-issue messages, a single primary issue is selected.

---

# 4. System Architecture

```text
                    Incoming Customer Message
                               |
                               v
                    +----------------------+
                    | TF-IDF + Logistic    |
                    | Regression Classifier|
                    +----------+-----------+
                               |
                    Intent + Confidence
                               |
                +--------------+--------------+
                |                             |
                v                             v
       TF-IDF Historical               Escalation Policy
          Retrieval                    - confidence
                |                      - similarity
                v                      - high-risk intent
       Top historical pairs                    |
                |                             |
                +-------------+---------------+
                              |
                              v
                    +----------------------+
                    | Ollama / Llama 3.2 3B|
                    | Grounded Reply Draft |
                    +----------+-----------+
                               |
                               v
                    AUTO-HANDLE / ESCALATE