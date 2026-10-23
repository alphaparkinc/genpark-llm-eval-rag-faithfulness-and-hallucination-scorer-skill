"""MCP server for RAG Faithfulness and Hallucination Scorer."""
import sys
import json
from client import RAGFaithfulnessScorer


def handle_request(req):
    method = req.get("method")
    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "score_faithfulness",
                    "description": "Decomposes RAG generated response into atomic claims, measures grounding overlap, and calculates hallucination score",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "context_passages": {"type": "array", "items": {"type": "string"}},
                            "min_faithfulness_threshold": {"type": "number", "default": 0.80},
                        },
                        "required": ["answer", "context_passages"],
                    },
                },
                {
                    "name": "decompose_into_claims",
                    "description": "Extracts atomic factual assertions from model generated prose",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                        },
                        "required": ["answer"],
                    },
                },
            ]
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name == "score_faithfulness":
            res = RAGFaithfulnessScorer.score_faithfulness(
                answer=args.get("answer", ""),
                context_passages=args.get("context_passages", []),
                min_faithfulness_threshold=float(args.get("min_faithfulness_threshold", 0.80)),
            )
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
        elif tool_name == "decompose_into_claims":
            res = RAGFaithfulnessScorer.decompose_into_claims(args.get("answer", ""))
            return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
    return {"error": "Method not found"}


if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(handle_request(json.loads(line))))
            sys.stdout.flush()
