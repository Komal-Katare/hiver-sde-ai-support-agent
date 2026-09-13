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
---

## 5. Baselines

Two baselines were used for intent classification.

### Baseline 1 - Majority Class

Every message is assigned to the most frequent class in the golden set.

Results:

| Metric | Result |
|---|---:|
| Accuracy | 36.0% |
| Macro Precision | 3.6% |
| Macro Recall | 10.0% |
| Macro F1 | 5.3% |

This establishes a simple lower bound and shows how much performance comes from learning intent distinctions instead of always predicting the dominant class.

### Baseline 2 - TF-IDF + Logistic Regression

The main classifier uses TF-IDF features with Logistic Regression and balanced class weights.

This was selected as the primary approach because it is:

- Lightweight.
- Fast to train.
- Interpretable.
- Easy to reproduce.
- Appropriate for a take-home prototype.

No large transformer classifier was required for the headline result.

---

## 6. Intent Classification Results

Evaluation was performed on the 200-message manually labelled golden set.

| Metric | Result |
|---|---:|
| Accuracy | 78.5% |
| Macro Precision | 81.0% |
| Macro Recall | 72.2% |
| Macro F1 | 71.2% |

### Per-Intent Results

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `account_security` | 1.00 | 0.43 | 0.60 | 7 |
| `app_issue` | 0.50 | 0.04 | 0.07 | 28 |
| `battery_power` | 1.00 | 1.00 | 1.00 | 17 |
| `connectivity` | 1.00 | 0.86 | 0.92 | 7 |
| `data_storage` | 1.00 | 0.50 | 0.67 | 6 |
| `device_hardware` | 1.00 | 0.77 | 0.87 | 26 |
| `ios_update_bug` | 0.67 | 0.97 | 0.79 | 30 |
| `orders_repair_warranty` | 0.67 | 0.67 | 0.67 | 3 |
| `other` | 0.76 | 1.00 | 0.86 | 72 |
| `settings_howto` | 0.50 | 1.00 | 0.67 | 4 |

The model performs strongly on intents such as `battery_power`, `connectivity`, and `device_hardware`.

The largest weakness is `app_issue`, which is frequently confused with `other` and `ios_update_bug`.

This is consistent with the overlap between app failures, operating-system problems, and general troubleshooting language.

The evaluation is intentionally small and manually labelled, so these numbers should be treated as an initial benchmark rather than a production-quality estimate.

---

## 7. Reply Generation and Escalation Evaluation

A separate 30-example evaluation set was used to inspect the complete support-agent pipeline.

For each message the system:

1. Predicts an intent.
2. Retrieves historical examples.
3. Generates a grounded reply with the local LLM.
4. Calculates escalation based on intent risk, classifier confidence, and retrieval similarity.

### Results

| Metric | Result |
|---|---:|
| Evaluation examples | 30 |
| Auto-handled | 14 |
| Escalated | 16 |
| Escalation rate | 53.3% |
| Average classifier confidence | 0.734 |
| Average retrieval similarity | 0.431 |

The relatively high escalation rate is intentional.

The system is designed to prefer human review when the classifier is uncertain, historical evidence is weak, or the intent is considered high-risk.

---

## 8. LLM-as-Judge Evaluation

The generated replies were evaluated using a local LLM judge.

Each response was scored from 1 to 5 on:

- Groundedness
- Relevance
- Helpfulness
- Tone
- Avoidance of unsupported claims
- Overall quality

### Average Judge Scores

| Criterion | Average |
|---|---:|
| Groundedness | 2.70 |
| Relevance | 2.67 |
| Helpfulness | 2.70 |
| Tone | 2.80 |
| No unsupported claims | 2.40 |
| Overall | 2.65 |

The judge is used as a diagnostic evaluation tool rather than being treated as ground truth.

### Human Validation of the Judge

30 responses were also scored using a human review rubric.

Agreement was measured using Spearman correlation, exact agreement, pass/fail agreement, and Cohen's kappa.

For the overall score:

- Spearman correlation: `0.338`
- Pass/fail agreement: `40.0%`
- Cohen's kappa: `0.106`

This indicates limited agreement between the local LLM judge and human scoring.

Therefore, the LLM judge should not be interpreted as a replacement for human evaluation.

### Agreement by Criterion

| Criterion | Spearman | Exact Agreement | Pass/Fail Agreement | Cohen's Kappa |
|---|---:|---:|---:|---:|
| Groundedness | 0.316 | 23.3% | 46.7% | 0.111 |
| Relevance | 0.260 | 20.0% | 33.3% | 0.051 |
| Helpfulness | 0.374 | 20.0% | 40.0% | 0.106 |
| Tone | 0.101 | 16.7% | 30.0% | 0.000 |
| No Unsupported Claims | -0.046 | 13.3% | 36.7% | 0.059 |
| Overall | 0.338 | 10.0% | 40.0% | 0.106 |

The limited agreement is itself an important evaluation result and is treated as a limitation of the automated judge.

---

## 9. Top 5 Failure Modes

### 1. App Issues Are Frequently Confused with `other` or `ios_update_bug`

Example:

> Customer reports an app problem involving AirPlay.

Prediction:

- Actual: `app_issue`
- Predicted: `other`
- Confidence: `0.348`
- Retrieval similarity: `0.335`

**Hypothesis:** App failures often contain generic words such as "not working", "update", "phone", and "fix". These words overlap with both software-update and miscellaneous support messages.

### 2. Low Retrieval Similarity Produces Weak Grounding

Several cases had retrieval similarity below `0.35`.

Example:

- Shattered-screen/hardware issue
- Retrieval similarity: approximately `0.229`

**Hypothesis:** TF-IDF retrieval is effective when terminology overlaps with historical cases but struggles with unusual wording or issues that are underrepresented in the historical corpus.

### 3. The LLM Can Produce Unsupported Claims

Even when historical examples are supplied, a small local language model can add plausible-sounding information that is not explicitly supported by the retrieved evidence.

**Hypothesis:** The generation model has general language knowledge that can override or extend beyond the narrow evidence supplied in the prompt.

**Mitigation:** The prompt explicitly prohibits invented policies, refunds, warranties, guarantees, timelines, and unsupported troubleshooting instructions.

### 4. Taxonomy Boundaries Are Sometimes Ambiguous

Some customer messages naturally contain multiple problems.

For example, an update may cause an application to stop working, making the message simultaneously relevant to `ios_update_bug` and `app_issue`.

**Hypothesis:** A single-label taxonomy loses information when support cases contain multiple related issues.

**Current decision:** Use the primary issue so that classification remains simple and deterministic.

### 5. Escalation Thresholds Are Heuristic

The `0.70` classifier-confidence and `0.35` retrieval-similarity thresholds were chosen conservatively rather than learned from a large calibration set.

**Hypothesis:** Better threshold calibration could reduce unnecessary escalation while maintaining safety.

**Next step:** Evaluate precision/recall trade-offs for escalation on a larger human-labelled set.

---

## 10. What Is Misleading About My Headline Number?

The headline **78.5% accuracy** is useful but potentially misleading.

First, the golden set contains only 200 examples and is imbalanced. The `other` class contains 72 examples, while some intents contain only 3-7 examples.

Second, the classifier was trained using weak heuristic labels rather than a large fully human-labelled training set. Therefore, the model can partially learn the biases of the labelling rules.

Third, the golden set was randomly sampled rather than temporally held out. This means the evaluation does not measure robustness to future changes in customer language.

Finally, accuracy hides class-specific weaknesses. For example, `app_issue` has very poor recall despite the overall accuracy looking strong.

For these reasons, the **71.2% macro-F1 and per-intent results are more informative than accuracy alone**.

The result should therefore be treated as an early benchmark rather than a production-readiness claim.

---

## 11. What I Would Do Next Week

If given another week, I would focus on evaluation quality and failure reduction rather than adding unnecessary infrastructure.

### Priority 1 - Improve Labelled Data

Expand the manually labelled set and ensure every intent has enough examples, especially:

- `app_issue`
- `orders_repair_warranty`
- `settings_howto`
- `data_storage`
- `account_security`

### Priority 2 - Improve Intent Classification

Compare the current TF-IDF classifier against a lightweight sentence-embedding or transformer baseline.

I would specifically measure whether semantic representations reduce the `app_issue` / `ios_update_bug` / `other` confusion.

### Priority 3 - Improve Retrieval

Experiment with hybrid retrieval:

- Lexical TF-IDF similarity
- Semantic similarity
- Metadata filtering by intent

The goal would be to increase retrieval quality before changing the generation model.

### Priority 4 - Calibrate Escalation

Use a larger human-labelled validation set to tune:

- Confidence threshold
- Retrieval threshold
- High-risk intent policy

Measure escalation precision, unnecessary escalations, and missed escalations.

### Priority 5 - Improve Response Evaluation

Create a larger human-reviewed response set and use the LLM judge only as a scalable secondary evaluator after establishing stronger human agreement.

---

## 12. Reproducibility

### Environment

Python 3.11 was used.

Install dependencies:

```text
pip install -r requirements.txt

The response generator requires Ollama and the `llama3.2:3b` model. 
```

### Data Preparation

The repository includes the preprocessing and training scripts needed to reproduce the headline results without requiring the full 3M-row dataset run.

1. Download the Kaggle dataset:
   `thoughtvector/customer-support-on-twitter`

2. Place the extracted `twcs.csv` under:
   `data/raw/twcs/twcs.csv`

3. Run the preprocessing pipeline:

```bash
python src/data_prepare.py
```


4. Train the intent classifier:

```bash
python src/train_intent.py
```

5. Build the historical retrieval index:

```bash
python src/build_retrieval.py
```

6. Run the intent evaluation:

```bash
python src/evaluate_intent.py
```

7. Run the reply-generation and escalation evaluation:

```bash
python src/evaluate_replies.py
```

8. Run the LLM-as-judge evaluation:

```bash
python src/llm_judge.py
```

The generated evaluation outputs are stored under results/.

### Golden Set

The final manually labelled golden set is included at:
evaluation/golden_set_labelled.csv


Validate it with:

```bash
python src/validate_golden_set.py
```

The golden set is already manually labelled and should be treated as the reported evaluation set.

The generation and cleaning scripts are included for reproducibility, but should not be run when reproducing the reported final metrics unless intentionally regenerating the golden set.

### Train Classifier

python src/train_intent_classifier.py
python src/evaluate_intent.py


### Build and Test Retrieval


python src/build_retriever.py
python src/test_retriever.py


### Run the Support Agent

After building the classifier and retrieval artifacts:


python src/support_agent.py


The support agent loads the trained classifier and retrieval artifacts and uses Ollama for local response generation.

### Run Reply Evaluation

Make sure Ollama is running and the model is available:


ollama pull llama3.2:3b


Then:


python src/create_reply_eval.py
python src/run_reply_eval.py
python src/run_llm_judge.py
python src/calculate_judge_agreement.py
python src/analyze_failures.py
python src/create_final_summary.py

Generated model and retrieval artifacts are intentionally excluded from Git because they are large derived files. They can be rebuilt using the commands above.

---

## 13. Repository Structure


hiver-sde-ai-support-agent/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- src/
| |-- inspect_dataset.py
| |-- extract_brand_data.py
| |-- build_conversations.py
| |-- clean_conversations.py
| |-- profile_support.py
| |-- create_golden_set.py
| |-- clean_golden_set.py
| |-- validate_golden_set.py
| |-- train_intent_classifier.py
| |-- evaluate_intent.py
| |-- build_retriever.py
| |-- test_retriever.py
| |-- support_agent.py
| |-- escalation.py
| |-- create_reply_eval.py
| |-- run_reply_eval.py
| |-- run_llm_judge.py
| |-- calculate_judge_agreement.py
| |-- analyze_failures.py
| |-- create_final_summary.py
| |-- run_baselines.py
| -- inspect_reply_outputs.py | |-- evaluation/ | |-- golden_set_labelled.csv | |-- human_review.csv | |-- reply_eval_set.csv | |-- reply_eval_outputs.csv | |-- llm_judge_results.csv | |-- judge_agreement_metrics.csv | -- judge_agreement_results.txt
|
|-- results/
| |-- baseline_results.txt
| |-- failure_analysis.txt
| -- final_evaluation_summary.txt | -- report/
`-- decision_log.md


---

## 14. Decision Log

The major design decisions are recorded in:


report/decision_log.md


The decision log contains 15 decisions covering:

1.  Brand selection 
2.  Customer-agent pair reconstruction 
3.  Intent taxonomy 
4.  Primary-issue labelling 
5.  TF-IDF + Logistic Regression 
6.  Majority baseline 
7.  Weak heuristic labels 
8.  Golden-set exclusion from classifier training 
9.  Golden-set exclusion from retrieval 
10.  TF-IDF retrieval instead of a vector database 
11.  Local Ollama model 
12.  Historical grounding and unsupported-claim restrictions 
13.  Escalation thresholds 
14.  High-risk escalation intents 
15.  Human validation of the LLM judge 

---

## 15. Limitations

This is a take-home prototype rather than a production support system.

Important limitations include:

-  Small manually labelled evaluation set 
-  Imbalanced intent distribution 
-  Weakly labelled classifier training data 
-  Random rather than temporal evaluation split 
-  TF-IDF retrieval depends on lexical overlap 
-  Small local LLM can generate unsupported statements 
-  Escalation thresholds are heuristic 
-  LLM-as-judge agreement with human review is limited 
-  No live customer-support integration 
-  No production monitoring or feedback loop 

The system is therefore best viewed as an auditable prototype demonstrating the core support-agent workflow and an evaluation-first approach.

