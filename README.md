# genpark-llm-eval-rag-faithfulness-and-hallucination-scorer-skill

RAG context faithfulness evaluator, atomic claim decomposition engine, and hallucination risk scorer.

100% Python Standard Library implementation with zero external dependencies.

## Key Capabilities

- **Atomic Factual Claim Extraction**: Splits LLM generated narrative into testable, discrete assertions.
- **Context Grounding Overlap Evaluation**: Measures token containment and semantic overlap against retrieved reference passages.
- **Hallucination Detection & Risk Indexing**: Pinpoints exact unsupported claims, calculating aggregate faithfulness scores against compliance thresholds.
- **Evaluation Gate Verdicts**: Emits deterministic `FAITHFUL_PASS` or `HALLUCINATION_DETECTED` verdicts to prevent serving unverified outputs to users.

## Architecture

```
[Generated Model Output + Retrieved Context Passages]
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              RAGFaithfulnessScorer                  │
│  - Atomic Sentence & Clause Claim Decomposer        │
│  - Reference Context Token Matcher & Overlap Scorer │
│  - Aggregate Faithfulness & Hallucination Counter   │
│  - SLA Compliance Gate & Verdict Decider            │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
[Faithfulness Score / Flagged Hallucinations / Gate Verdict]
```

## MCP Server Integration

Run standard MCP stdio server:
```bash
python mcp_server.py
```

Exposed Tools:
- `score_faithfulness`: Decomposes RAG generated response into atomic claims, measures grounding overlap, and calculates hallucination score.
- `decompose_into_claims`: Extracts atomic factual assertions from model generated prose.

## Quickstart (Python API)

```python
from client import RAGFaithfulnessScorer

context = ["Python 3.12 was released in October 2023 with performance enhancements."]
answer = "Python 3.12 came out in October 2023."
result = RAGFaithfulnessScorer.score_faithfulness(answer, context)
print("Faithful:", result["is_faithful"])
print("Score:", result["faithfulness_score"])
```

---

## GenPark Ecosystem

- **Official Website**: [https://www.genpark.ai](https://www.genpark.ai)
- **LinkedIn Page**: [https://www.linkedin.com/company/92955550](https://www.linkedin.com/company/92955550)
