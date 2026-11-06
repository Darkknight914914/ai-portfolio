"""RAG built from its parts: chunking, BM25, dense search, fusion, answering.

The retrievers are written to be read. BM25 is implemented by hand. Dense search
uses LSA (TF-IDF plus SVD) because it runs offline on a laptop; to use neural
embeddings, replace `DenseIndex.embed`.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TOKEN = re.compile(r"[a-z0-9_]+")
STOP = frozenset(
    "a an and are as at be but by can do does for from has have how i if in into is it its of on or so "
    "that the their them then there these this to was were what when where which while who why will with "
    "you your my me we".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.source}#{self.index}"


def chunk_text(source: str, text: str, size: int = 150, overlap: int = 30) -> list[Chunk]:
    """Split into chunks of about `size` words on paragraph breaks, with overlap.
    A paragraph longer than a chunk is cut by words."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words_out: list[list[str]] = []
    current: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        words = paragraph.split()
        while words:
            room = size - len(current)
            current.extend(words[:room])
            words = words[room:]
            if len(current) >= size:
                words_out.append(current)
                current = current[-overlap:] if overlap else []
    if current and (not words_out or len(current) > overlap):
        words_out.append(current)
    return [Chunk(source, i, " ".join(w)) for i, w in enumerate(words_out)]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        docs = [tokenize(c.text) for c in chunks]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avg_length = max(self.lengths.mean(), 1.0) if len(docs) else 1.0
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                postings.setdefault(term, []).append((i, count))
        self.postings = {
            t: (np.array([p[0] for p in ps]), np.array([p[1] for p in ps], dtype=float)) for t, ps in postings.items()
        }
        self.n = len(docs)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        for term in tokenize(query):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            norm = self.k1 * (1 - self.b + self.b * self.lengths[ids] / self.avg_length)
            out[ids] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class TfidfIndex:
    def __init__(self, chunks: list[Chunk]):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, sublinear_tf=True, ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(c.text for c in chunks)

    def scores(self, query: str) -> np.ndarray:
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()


class DenseIndex:
    """Dense vectors from LSA. Words that appear in similar chunks end up close
    together, so a query can match a chunk that doesn't share its exact words."""

    def __init__(self, chunks: list[Chunk], dims: int = 200, seed: int = 0):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, sublinear_tf=True, min_df=2)
        tfidf = self.vectorizer.fit_transform(c.text for c in chunks)
        dims = min(dims, min(tfidf.shape) - 1)
        self.svd = TruncatedSVD(n_components=dims, random_state=seed)
        self.vectors = normalize(self.svd.fit_transform(tfidf))

    def embed(self, text: str) -> np.ndarray:
        return normalize(self.svd.transform(self.vectorizer.transform([text])))[0]

    def scores(self, query: str) -> np.ndarray:
        return self.vectors @ self.embed(query)


def reciprocal_rank_fusion(score_lists: Iterable[np.ndarray], k: int = 60) -> np.ndarray:
    """Combine rankings by position, so the retrievers' scales don't need to match."""
    lists = list(score_lists)
    out = np.zeros(len(lists[0]))
    for scores in lists:
        ranks = np.empty(len(scores), dtype=int)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
        out += 1.0 / (k + ranks + 1)
    return out


class Retriever:
    MODES = ("bm25", "tfidf", "dense", "hybrid")

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75, dims: int = 200):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks, k1, b)
        self.tfidf = TfidfIndex(chunks)
        self.dense = DenseIndex(chunks, dims)

    def scores(self, query: str, mode: str = "hybrid") -> np.ndarray:
        if mode == "hybrid":
            return reciprocal_rank_fusion([self.bm25.scores(query), self.dense.scores(query)])
        return {"bm25": self.bm25, "tfidf": self.tfidf, "dense": self.dense}[mode].scores(query)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[tuple[Chunk, float]]:
        scores = self.scores(query, mode)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def confidence(self, query: str) -> float:
        """Best TF-IDF match: a rough signal of whether the corpus covers the question."""
        return float(self.tfidf.scores(query).max())


ABSTAIN = "I couldn't find this in the documents."


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(f"[{c.ref}]\n{c.text}" for c, _ in hits)
    return (
        "Answer the question using only the sources below. Cite the source id in square brackets "
        f"after each claim. If the sources don't contain the answer, reply exactly: {ABSTAIN}\n\n"
        f"SOURCES:\n{context}\n\nQUESTION: {question}"
    )


def extractive_answer(question: str, hits: list[tuple[Chunk, float]], sentences: int = 2) -> str:
    """Offline fallback: quote the sentences that best overlap the question.
    It isn't a real generated answer, but it runs with no API key and can't invent anything."""
    wanted = set(tokenize(question))
    scored = []
    for chunk, _ in hits:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            overlap = len(wanted & set(tokenize(sentence)))
            if overlap:
                scored.append((overlap / math.sqrt(len(tokenize(sentence)) + 1), sentence.strip(), chunk.ref))
    scored.sort(key=lambda s: -s[0])
    if not scored:
        return ABSTAIN
    return " ".join(f"{text} [{ref}]" for _, text, ref in scored[:sentences])


def answer(
    question: str,
    retriever: Retriever,
    k: int = 5,
    min_confidence: float = 0.0,
    llm: Callable[[str], str] | None = None,
) -> dict:
    """Retrieve, then answer. Without `llm` the answer is extractive."""
    if retriever.confidence(question) < min_confidence:
        return {"answer": ABSTAIN, "sources": []}
    hits = retriever.search(question, k)
    text = llm(build_prompt(question, hits)) if llm else extractive_answer(question, hits)
    return {"answer": text, "sources": [c.ref for c, _ in hits]}


def load_folder(folder: str | Path, size: int = 150, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            chunks.extend(chunk_text(path.stem, path.read_text(encoding="utf-8"), size, overlap))
    return chunks


def claude_llm(prompt: str) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    from claude import call_claude, text_of

    return text_of(call_claude([{"role": "user", "content": prompt}], max_tokens=800))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask a question about a folder of .md or .txt files.")
    parser.add_argument("folder")
    parser.add_argument("question")
    parser.add_argument("--claude", action="store_true", help="Generate with Claude (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()
    result = answer(args.question, Retriever(load_folder(args.folder)), llm=claude_llm if args.claude else None)
    print(result["answer"], "\n\nRetrieved:", ", ".join(result["sources"]))
