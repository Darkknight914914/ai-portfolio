import math

import numpy as np

from rag import ABSTAIN, BM25Index, Chunk, Retriever, answer, chunk_text, reciprocal_rank_fusion, tokenize

DOCS = [
    Chunk("a", 0, "Interest is paid at maturity on fixed term deposits."),
    Chunk("b", 0, "Partner banks must send reconciliation files each morning."),
    Chunk("c", 0, "Customer complaints are answered within five working days."),
    Chunk("d", 0, "Fixed term products cannot be closed early by customers."),
]


def test_chunks_respect_size_and_overlap():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text("x", text, size=30, overlap=10)
    assert all(len(c.text.split()) <= 30 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-10:] == second[:10]
    assert chunks[-1].text.split()[-1] == "99"


def test_chunking_keeps_all_words():
    text = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(17)) for p in range(9))
    seen = {w for c in chunk_text("x", text, 25, 5) for w in c.text.split()}
    assert seen == set(text.split())


def test_bm25_matches_hand_calculation():
    index = BM25Index(DOCS, k1=1.5, b=0.75)
    scores = index.scores("reconciliation")
    n, df = 4, 1
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    length = len(tokenize(DOCS[1].text))
    avg = np.mean([len(tokenize(d.text)) for d in DOCS])
    expected = idf * (1 * 2.5) / (1 + 1.5 * (0.25 + 0.75 * length / avg))
    assert math.isclose(scores[1], expected)
    assert scores[0] == scores[2] == scores[3] == 0


def test_rare_word_outweighs_common_word():
    index = BM25Index(DOCS)
    assert index.scores("fixed reconciliation").argmax() == 1


def test_fusion_rewards_agreement():
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 3.0, 2.0])
    assert reciprocal_rank_fusion([a, b]).argmax() == 1


def test_search_finds_the_right_chunk():
    retriever = Retriever(DOCS, dims=2)
    assert retriever.search("when is interest paid", 1, "bm25")[0][0].source == "a"
    assert retriever.search("when is interest paid", 1, "hybrid")[0][0].source == "a"


def test_abstains_below_threshold():
    retriever = Retriever(DOCS, dims=2)
    out = answer("best pizza in naples", retriever, min_confidence=0.1)
    assert out == {"answer": ABSTAIN, "sources": []}


def test_llm_receives_sources_and_question():
    seen = {}

    def fake(prompt):
        seen["prompt"] = prompt
        return "ok"

    out = answer("complaints deadline", Retriever(DOCS, dims=2), k=2, llm=fake)
    assert out["answer"] == "ok"
    assert "[c#0]" in seen["prompt"] and "complaints deadline" in seen["prompt"]
