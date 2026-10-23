"""RAG Faithfulness and Hallucination Scorer.
100% Python Standard Library.
"""

import re
from typing import List, Dict, Any, Optional


class RAGFaithfulnessScorer:
    """Evaluates RAG generation faithfulness by decomposing answers into atomic claims,
    measuring context grounding, and flagging unverified factual hallucinations."""

    @classmethod
    def decompose_into_claims(cls, answer: str) -> List[str]:
        """Splits answer text into discrete factual assertions across sentence and clause boundaries."""
        raw_sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        claims = []
        for s in raw_sentences:
            cleaned = s.strip()
            if not cleaned or len(cleaned) < 8:
                continue
            # Also split compound sentences with semicolons
            clauses = re.split(r';\s*', cleaned)
            for c in clauses:
                c_clean = c.strip().rstrip('.')
                if len(c_clean) >= 10:
                    claims.append(c_clean)
        return claims

    @classmethod
    def evaluate_claim_grounding(
        cls,
        claim: str,
        context_passages: List[str],
        grounding_threshold: float = 0.65,
    ) -> Dict[str, Any]:
        """Evaluates whether an individual claim is grounded in the provided context passages."""
        # Extract meaningful claim keywords (ignore common stop words)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at",
            "to", "for", "with", "by", "from", "of", "it", "this", "that", "these", "those",
            "as", "be", "has", "have", "had", "can", "will", "would", "should", "not"
        }
        claim_tokens = [
            t.lower() for t in re.findall(r'\b[a-zA-Z0-9_\-\$]+\b', claim)
            if t.lower() not in stopwords and len(t) > 2
        ]

        if not claim_tokens:
            return {"grounded": True, "score": 1.0, "matched_context": ""}

        best_score = 0.0
        best_passage = ""

        combined_contexts = context_passages if context_passages else [""]
        for passage in combined_contexts:
            p_lower = passage.lower()
            matched_count = 0
            for token in claim_tokens:
                if token in p_lower:
                    matched_count += 1

            overlap_ratio = matched_count / len(claim_tokens)
            if overlap_ratio > best_score:
                best_score = overlap_ratio
                best_passage = passage

        is_grounded = best_score >= grounding_threshold

        return {
            "claim": claim,
            "grounded": is_grounded,
            "grounding_score": round(best_score, 3),
            "matched_snippet": best_passage[:150] if is_grounded else "",
            "key_tokens": claim_tokens,
        }

    @classmethod
    def score_faithfulness(
        cls,
        answer: str,
        context_passages: List[str],
        min_faithfulness_threshold: float = 0.80,
    ) -> Dict[str, Any]:
        """Computes aggregate faithfulness score across all claims in the generated response."""
        claims = cls.decompose_into_claims(answer)
        if not claims:
            return {
                "faithfulness_score": 1.0,
                "hallucination_rate": 0.0,
                "total_claims": 0,
                "grounded_claims_count": 0,
                "hallucinations_count": 0,
                "is_faithful": True,
                "hallucinations": [],
                "claim_evaluations": [],
            }

        evaluations = []
        grounded_count = 0
        hallucinations = []

        for c in claims:
            res = cls.evaluate_claim_grounding(c, context_passages)
            evaluations.append(res)
            if res["grounded"]:
                grounded_count += 1
            else:
                hallucinations.append({
                    "claim": c,
                    "grounding_score": res["grounding_score"],
                    "reason": "Insufficient factual overlap with reference context.",
                })

        faithfulness_score = round(grounded_count / len(claims), 3)
        hallucination_rate = round(1.0 - faithfulness_score, 3)
        is_faithful = faithfulness_score >= min_faithfulness_threshold

        return {
            "faithfulness_score": faithfulness_score,
            "hallucination_rate": hallucination_rate,
            "total_claims": len(claims),
            "grounded_claims_count": grounded_count,
            "hallucinations_count": len(hallucinations),
            "is_faithful": is_faithful,
            "verdict": "FAITHFUL_PASS" if is_faithful else "HALLUCINATION_DETECTED",
            "hallucinations": hallucinations,
            "claim_evaluations": evaluations,
        }
