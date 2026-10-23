"""Example usage for RAG Faithfulness and Hallucination Scorer."""
from client import RAGFaithfulnessScorer


def main():
    retrieved_contexts = [
        "GenPark delivers enterprise agentic workflows with 100% Python Standard Library implementations.",
        "The pricing tier starts at $49/month for starter teams and includes 50,000 monthly execution credits.",
        "Security compliance includes SOC2 Type II certification and append-only cryptographic audit logging.",
    ]

    print("=== 1. Evaluating Faithful Response ===")
    faithful_answer = (
        "GenPark offers enterprise agentic workflows using pure Python Standard Library. "
        "The starter tier costs $49/month with 50,000 monthly execution credits. "
        "It supports SOC2 Type II compliance and cryptographic audit logging."
    )
    res_pass = RAGFaithfulnessScorer.score_faithfulness(faithful_answer, retrieved_contexts)
    print("Verdict:", res_pass["verdict"])
    print("Faithfulness Score:", res_pass["faithfulness_score"])
    print("Hallucinations Count:", res_pass["hallucinations_count"])

    print("\n=== 2. Evaluating Hallucinatory Response ===")
    hallucinatory_answer = (
        "GenPark includes an autonomous quantum computing compiler. "
        "The starter plan is free forever with unlimited compute. "
        "It supports SOC2 Type II certification."
    )
    res_fail = RAGFaithfulnessScorer.score_faithfulness(hallucinatory_answer, retrieved_contexts)
    print("Verdict:", res_fail["verdict"])
    print("Faithfulness Score:", res_fail["faithfulness_score"])
    print("Hallucinations Count:", res_fail["hallucinations_count"])
    for h in res_fail["hallucinations"]:
        print(f"  [HALLUCINATION] '{h['claim']}' (score: {h['grounding_score']})")


if __name__ == "__main__":
    main()
