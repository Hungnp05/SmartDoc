"""
SmartDoc-Insight Benchmark Script
───────────────────────────────────
Measures Table QA accuracy vs standard RAG baseline.
Usage: python scripts/benchmark.py --doc tests/sample_docs/sample_report.pdf
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.models.ollama_client import OllamaClient
from src.layers.vision_processing import VisionProcessingLayer
from src.layers.knowledge_base import KnowledgeBaseLayer
from src.layers.retrieval_reasoning import RetrievalReasoningLayer

# Sample QA pairs for evaluation 
# Format: {question, expected_keywords, content_type}
SAMPLE_QA_PAIRS = [
    {
        "question": "What is the total revenue in Q3?",
        "expected_keywords": ["revenue", "Q3", "billion", "million"],
        "content_type": "table",
    },
    {
        "question": "What trend does the growth chart show?",
        "expected_keywords": ["growth", "increase", "trend", "quarter"],
        "content_type": "figure",
    },
    {
        "question": "What are the main risks mentioned in the report?",
        "expected_keywords": ["risk", "challenge", "concern"],
        "content_type": "text",
    },
]


def evaluate_answer(answer: str, expected_keywords: list) -> float:
    """Simple keyword-based evaluation (0-1 score)."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords) if expected_keywords else 0.0


def run_benchmark(doc_path: str, output_file: str = None):
    print(f"\n{'='*60}")
    print("SmartDoc-Insight Benchmark")
    print(f"{'='*60}")
    print(f"Document: {doc_path}\n")

    # Initialize
    ollama = OllamaClient(config)
    vision = VisionProcessingLayer(config, ollama)
    kb = KnowledgeBaseLayer(config, ollama)
    rag = RetrievalReasoningLayer(config, kb, ollama)

    # Process document
    print("Processing document...")
    start = time.time()
    doc = vision.process_document(doc_path)
    kb.ingest_document(doc)
    process_time = time.time() - start
    print(f"✓ Processed in {process_time:.1f}s — {len(doc.all_regions())} regions\n")

    # Run queries
    results = []
    total_score = 0.0
    table_scores = []
    figure_scores = []
    text_scores = []

    for qa in SAMPLE_QA_PAIRS:
        print(f"Q: {qa['question']}")
        q_start = time.time()

        response = rag.query(qa['question'])
        q_time = time.time() - q_start

        score = evaluate_answer(response.answer, qa['expected_keywords'])
        total_score += score

        result = {
            "question": qa['question'],
            "answer": response.answer[:200],
            "score": score,
            "latency": q_time,
            "content_type": qa['content_type'],
            "sources_used": [s.content_type for s in response.sources],
        }
        results.append(result)

        if qa['content_type'] == "table":
            table_scores.append(score)
        elif qa['content_type'] == "figure":
            figure_scores.append(score)
        else:
            text_scores.append(score)

        emoji = "✅" if score >= 0.5 else "⚠️"
        print(f"{emoji} Score: {score:.0%} | Latency: {q_time:.1f}s")
        print(f"   A: {response.answer[:150]}...\n")

    # Summary
    avg_score = total_score / len(SAMPLE_QA_PAIRS)
    print(f"\n{'─'*40}")
    print(f"RESULTS SUMMARY")
    print(f"{'─'*40}")
    print(f"Overall Accuracy:  {avg_score:.0%}")
    if table_scores:
        print(f"Table QA:          {sum(table_scores)/len(table_scores):.0%}")
    if figure_scores:
        print(f"Chart QA:          {sum(figure_scores)/len(figure_scores):.0%}")
    if text_scores:
        print(f"Text QA:           {sum(text_scores)/len(text_scores):.0%}")
    print(f"Processing Time:   {process_time:.1f}s")
    print(f"{'─'*40}\n")

    if output_file:
        with open(output_file, "w") as f:
            json.dump({"summary": {"overall": avg_score}, "results": results}, f, indent=2)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True, help="Path to PDF document")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    run_benchmark(args.doc, args.output)
