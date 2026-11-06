# 2. RAG from scratch, with measured retrieval

Chunking, BM25 (written by hand), dense search, rank fusion and answering. The answer step uses Claude when an API key is set and quotes source sentences when it isn't.

```
python -m pytest
python evaluate.py
python rag.py ../3_agent/policy "What happens if a report has a blocking issue?"
python rag.py <folder> "<question>" --claude
```

## How it was measured
- **Corpus:** Python's language reference (81 topics, split into 349 chunks). It's real text that wasn't written for this project, and it ships with Python.
- **Questions:** 48 hand-labelled questions in everyday wording, split by hash into 30 dev and 18 test.
- **Settings:** chosen on dev only (chunk size 300 words, BM25 k1 1.5, b 0.75, 100 dense dimensions).
- **Hit:** a retrieved chunk comes from a topic labelled correct for that question.

## Results on the 18 test questions

| Retriever | Recall@1 | Recall@5 | MRR@10 | Recall@5 95% interval |
|---|---|---|---|---|
| BM25 | 61% | 89% | 0.73 | 72–100% |
| TF-IDF | 61% | 89% | 0.74 | 72–100% |
| Dense (LSA) | 67% | 94% | 0.79 | 83–100% |
| Hybrid (BM25 + dense) | 67% | 94% | 0.78 | 83–100% |

## How to read these honestly
- **Hybrid did not beat dense alone.** The usual advice is "always use hybrid", but on this data it made no measurable difference.
- **The retrievers can't be separated.** With 18 questions, the intervals overlap heavily. These results show the method works, and they aren't evidence that one retriever wins.
- **Test scored higher than dev** (dev recall@5 was 83%), so the test questions probably came out easier. Expect something nearer the lower figure on new questions.
- **I wrote the questions and labels.** A real test set should come from real users.
- **Knowing when to say "I don't know" doesn't work yet.** A simple score threshold answered 67% of in-scope test questions and declined only 2 of 4 out-of-scope ones. That's too weak to rely on, so it's off by default.
- **No neural embeddings.** `sentence-transformers` couldn't be installed where this was built. LSA is the stand-in, and replacing `DenseIndex.embed` is the obvious next experiment.
- The Claude answer step has not been run against the live API. Only retrieval is measured here, not answer quality.
