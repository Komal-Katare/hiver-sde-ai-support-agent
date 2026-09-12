\# Decision Log



\## 1. Brand selection — AppleSupport

\*\*Decision:\*\* Use AppleSupport as the single brand.



\*\*Reason:\*\* AppleSupport has a large number of customer-support interactions in the dataset and provides a coherent product-support domain for building a focused prototype.



\---



\## 2. Use direct customer → agent pairs

\*\*Decision:\*\* Reconstruct direct customer messages and their immediate AppleSupport responses.



\*\*Reason:\*\* This provides a simple historical resolution corpus for retrieval without requiring full multi-turn conversation modelling.



\---



\## 3. Use a small intent taxonomy

\*\*Decision:\*\* Use 10 support intents.



\*\*Intents:\*\*

\- `ios\_update\_bug`

\- `battery\_power`

\- `device\_hardware`

\- `app\_issue`

\- `connectivity`

\- `account\_security`

\- `data\_storage`

\- `settings\_howto`

\- `orders\_repair\_warranty`

\- `other`



\*\*Reason:\*\* A small taxonomy makes the prototype easier to evaluate and keeps categories tied to recurring support themes in the selected brand data.



\---



\## 4. Primary-issue labelling

\*\*Decision:\*\* Assign one primary intent to each customer message.



\*\*Reason:\*\* Customer messages can contain multiple symptoms. A single primary issue keeps the classifier task well-defined for this prototype.



\---



\## 5. TF-IDF + Logistic Regression

\*\*Decision:\*\* Use TF-IDF features with Logistic Regression for intent classification.



\*\*Reason:\*\* It is lightweight, fast to train, interpretable, and easy to reproduce without requiring GPU infrastructure or model fine-tuning.



\---



\## 6. Majority-class baseline

\*\*Decision:\*\* Compare the classifier against a majority-class predictor.



\*\*Reason:\*\* This establishes a simple lower-bound baseline and tests whether the learned classifier provides meaningful improvement.



\---



\## 7. Weak heuristic labels for training

\*\*Decision:\*\* Generate training labels using deterministic keyword/heuristic rules rather than manually labelling the full historical corpus.



\*\*Reason:\*\* Manually labelling more than 100K historical conversations is impractical for the assignment. The manually labelled golden set is reserved for evaluation.



\---



\## 8. Golden-set exclusion from classifier training

\*\*Decision:\*\* Remove golden-set customer messages from the classifier training pool.



\*\*Reason:\*\* Prevent direct evaluation leakage and ensure the evaluation examples are not used as training examples.



\---



\## 9. Golden-set exclusion from retrieval

\*\*Decision:\*\* Remove golden-set customer messages from the historical retrieval corpus.



\*\*Reason:\*\* Prevent the retriever from returning the exact evaluation conversation and artificially inflating retrieval similarity.



\---



\## 10. TF-IDF retrieval instead of a vector database

\*\*Decision:\*\* Use TF-IDF cosine similarity for historical-response retrieval.



\*\*Reason:\*\* The assignment prioritizes a runnable prototype. TF-IDF retrieval is simple, local, fast, and avoids unnecessary infrastructure such as a vector database.



\---



\## 11. Local LLM through Ollama

\*\*Decision:\*\* Use Ollama with `llama3.2:3b` for response generation.



\*\*Reason:\*\* Local inference avoids external API dependency and allows the prototype to run without requiring paid API credits.



\---



\## 12. Ground responses in historical evidence

\*\*Decision:\*\* Give the LLM retrieved historical customer/agent examples and explicitly restrict unsupported claims.



\*\*Reason:\*\* The agent should draft responses based on observed support resolutions rather than inventing policies, refunds, guarantees, timelines, or troubleshooting steps.



\---



\## 13. Confidence and retrieval thresholds for escalation

\*\*Decision:\*\* Escalate when classifier confidence is below 0.70 or retrieval similarity is below 0.35.



\*\*Reason:\*\* Low classification confidence or weak historical evidence indicates that the system does not have sufficient confidence to safely auto-handle the request.



\---



\## 14. High-risk intent escalation

\*\*Decision:\*\* Automatically escalate `account\_security`, `orders\_repair\_warranty`, and `data\_storage`.



\*\*Reason:\*\* Security, account access, potential data loss, orders, repairs, and warranty-related requests can require case-specific handling and should receive additional human oversight.



\---



\## 15. Human validation of the LLM judge

\*\*Decision:\*\* Compare LLM-judge ratings against human ratings on 30 reply cases.



\*\*Reason:\*\* An automated judge should not be treated as ground truth without checking whether its scores align with human evaluation.



\*\*Result:\*\* Overall Spearman correlation was 0.338 and Cohen's kappa was 0.106, indicating limited agreement. The LLM judge is therefore treated as a secondary diagnostic rather than definitive ground truth.

