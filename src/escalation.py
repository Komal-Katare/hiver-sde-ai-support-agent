HIGH_RISK_INTENTS = {
    "account_security",
    "orders_repair_warranty",
    "data_storage"
}


def decide_handling(intent, confidence, top_similarity):

    reasons = []

    # High-risk cases should go to a human
    if intent in HIGH_RISK_INTENTS:
        reasons.append(
            f"{intent} is classified as a high-risk intent"
        )

    # Low classifier confidence
    if confidence < 0.70:
        reasons.append(
            f"classifier confidence {confidence:.3f} is below 0.70"
        )

    # Weak historical evidence
    if top_similarity < 0.35:
        reasons.append(
            f"historical similarity {top_similarity:.3f} is below 0.35"
        )

    if reasons:
        return "ESCALATE", "; ".join(reasons)

    return (
        "AUTO-HANDLE",
        "high classifier confidence and sufficiently "
        "similar historical support evidence"
    )