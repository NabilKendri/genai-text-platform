import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.append("src")
from search import search
from graph import llm, Extraction

CHUNKS = Path("data/processed/chunks.jsonl")
random.seed(42)


def load_sample(n):
    with CHUNKS.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return random.sample(rows, n)


def eval_retrieval(rows, k=5):
    search("warm up")  # load model once so it doesn't skew latency
    precisions, latencies = [], []
    for r in rows:
        start = time.perf_counter()
        results = search(r["text"][:80], k=k)
        latencies.append((time.perf_counter() - start) * 1000)
        same = sum(d.metadata["label"] == r["label"] for d, _ in results)
        precisions.append(same / k)
    return statistics.mean(precisions), statistics.mean(latencies)


def eval_extraction(rows):
    extractor = llm.with_structured_output(Extraction)
    correct, failures, latencies = 0, 0, []
    for i, r in enumerate(rows, 1):
        start = time.perf_counter()
        try:
            out = extractor.invoke(f"Extract information from this news article:\n\n{r['text']}")
            correct += out.category == r["label"]
        except Exception:
            failures += 1
        latencies.append((time.perf_counter() - start) * 1000)
        print(f"Extraction {i}/{len(rows)}", flush=True)
    return correct / len(rows), failures, statistics.mean(latencies)


if __name__ == "__main__":
    precision, ret_ms = eval_retrieval(load_sample(100))
    acc, fails, ext_ms = eval_extraction(load_sample(20))
    print("\n| Metric | Value |")
    print("|---|---|")
    print(f"| Retrieval label precision@5 (100 queries) | {precision:.1%} |")
    print(f"| Avg search latency | {ret_ms:.0f} ms |")
    print(f"| Extraction category accuracy (20 docs) | {acc:.1%} |")
    print(f"| Extraction failures | {fails} |")
    print(f"| Avg extraction latency | {ext_ms:.0f} ms |")