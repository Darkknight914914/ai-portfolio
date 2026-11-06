"""Measure retrieval on Python's language reference, with labelled questions.

The corpus is real, is not written for this project, and ships with Python
(pydoc_data.topics). Settings are chosen on the dev half. The numbers to quote
come from the test half, which played no part in choosing them.
"""
from __future__ import annotations

import hashlib
import json
import platform
from itertools import product
from pathlib import Path

import numpy as np
from pydoc_data.topics import topics

from rag import Retriever, chunk_text

HERE = Path(__file__).parent
TOP_K = 10


def build_chunks(size: int, overlap: int):
    chunks = []
    for name in sorted(topics):
        chunks.extend(chunk_text(name, topics[name], size, overlap))
    return chunks


def split(items, key=lambda x: x):
    dev, test = [], []
    for item in items:
        bucket = int(hashlib.sha256(key(item).encode()).hexdigest(), 16) % 2
        (dev if bucket == 0 else test).append(item)
    return dev, test


def rank_of_first_hit(retriever: Retriever, q: dict, mode: str) -> int | None:
    gold = set(q["gold"])
    for rank, (chunk, _) in enumerate(retriever.search(q["q"], TOP_K, mode), 1):
        if chunk.source in gold:
            return rank
    return None


def score(ranks: list[int | None]) -> dict:
    r = np.array([x if x else 10**6 for x in ranks], dtype=float)
    return {
        "recall@1": float((r <= 1).mean()),
        "recall@5": float((r <= 5).mean()),
        "mrr@10": float(np.where(r <= TOP_K, 1 / r, 0).mean()),
    }


def bootstrap_interval(ranks, metric="recall@5", n=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        picks = rng.integers(0, len(ranks), len(ranks))
        values.append(score([ranks[i] for i in picks])[metric])
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    data = json.loads((HERE / "questions.json").read_text())
    dev, test = split(data["in_scope"], key=lambda q: q["q"])
    oos_dev, oos_test = split(data["out_of_scope"])

    print(f"Questions: {len(dev)} dev, {len(test)} test")
    best, best_score, grid = None, -1.0, []
    for size, (k1, b), dims in product([80, 150, 300], [(1.5, 0.75), (0.9, 0.4)], [100, 200]):
        retriever = Retriever(build_chunks(size, size // 5), k1=k1, b=b, dims=dims)
        s = score([rank_of_first_hit(retriever, q, "hybrid") for q in dev])
        grid.append({"chunk_words": size, "k1": k1, "b": b, "dims": dims, **s})
        print(grid[-1])
        if s["mrr@10"] > best_score:
            best, best_score = grid[-1], s["mrr@10"]
    size, k1, b, dims = best["chunk_words"], best["k1"], best["b"], best["dims"]
    chunks = build_chunks(size, size // 5)
    retriever = Retriever(chunks, k1=k1, b=b, dims=dims)

    results = {}
    for mode in Retriever.MODES:
        ranks = [rank_of_first_hit(retriever, q, mode) for q in test]
        results[mode] = {**score(ranks), "recall@5_ci95": bootstrap_interval(ranks)}

    # Pick an abstain threshold on dev, then measure it on test.
    in_dev = [retriever.confidence(q["q"]) for q in dev]
    out_dev = [retriever.confidence(q) for q in oos_dev]
    candidates = np.linspace(0, max(in_dev + out_dev), 200)
    balanced = [((np.array(in_dev) >= t).mean() + (np.array(out_dev) < t).mean()) / 2 for t in candidates]
    threshold = float(candidates[int(np.argmax(balanced))])
    abstain = {
        "threshold": threshold,
        "test_answered_when_in_scope": float((np.array([retriever.confidence(q["q"]) for q in test]) >= threshold).mean()),
        "test_abstained_when_out_of_scope": float((np.array([retriever.confidence(q) for q in oos_test]) < threshold).mean()),
        "out_of_scope_test_questions": len(oos_test),
    }

    misses = [q["q"] for q in test if not rank_of_first_hit(retriever, q, "hybrid") or rank_of_first_hit(retriever, q, "hybrid") > 5]
    output = {
        "python": platform.python_version(),
        "corpus_topics": len(topics),
        "corpus_chunks": len(chunks),
        "dev_questions": len(dev),
        "test_questions": len(test),
        "chosen_settings": {"chunk_words": size, "k1": k1, "b": b, "dims": dims},
        "dev_grid": grid,
        "test_results": results,
        "abstain": abstain,
        "hybrid_misses_at_5": misses,
    }
    (HERE / "results.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({k: output[k] for k in ("chosen_settings", "corpus_chunks", "test_results", "abstain", "hybrid_misses_at_5")}, indent=2))


if __name__ == "__main__":
    main()
