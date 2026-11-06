"""Train the character model and compare it with simple baselines.

    python train.py                       trains on Python's language reference
    python train.py --data input.txt      trains on any text file, e.g. Tiny Shakespeare
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from model import GPT, AdamW, Config

HERE = Path(__file__).parent


def load_text(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    from pydoc_data.topics import topics

    return "\n\n".join(topics[k] for k in sorted(topics))


def ngram_loss(train: np.ndarray, val: np.ndarray, vocab: int, n: int) -> float:
    """Validation loss of a smoothed counting model, in nats per character."""
    counts: dict[bytes, np.ndarray] = {}
    context = n - 1
    for i in range(context, len(train)):
        key = train[i - context : i].tobytes()
        if key not in counts:
            counts[key] = np.zeros(vocab)
        counts[key][train[i]] += 1
    overall = np.bincount(train, minlength=vocab) + 1.0
    overall /= overall.sum()
    total = 0.0
    for i in range(context, len(val)):
        row = counts.get(val[i - context : i].tobytes())
        if row is None:
            p = overall[val[i]]
        else:
            p = (row[val[i]] + 5.0 * overall[val[i]]) / (row.sum() + 5.0)
        total -= np.log(p)
    return float(total / (len(val) - context))


def batch(data: np.ndarray, size: int, block: int, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - block - 1, size)
    x = np.stack([data[s : s + block] for s in starts])
    y = np.stack([data[s + 1 : s + block + 1] for s in starts])
    return x, y


def estimate_loss(model: GPT, data: np.ndarray, batches: int, size: int, seed: int) -> float:
    """Fixed seed, so every evaluation sees the same characters."""
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(batches):
        x, y = batch(data, size, model.config.block_size, rng)
        losses.append(float(model.forward(x, y)[1].data))
    return float(np.mean(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=96)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default=str(HERE / "results.json"))
    args = p.parse_args()

    text = load_text(args.data)
    chars = sorted(set(text))
    to_id = {c: i for i, c in enumerate(chars)}
    data = np.array([to_id[c] for c in text], dtype=np.int32)
    cut = int(0.9 * len(data))
    train_data, val_data = data[:cut], data[cut:]

    baselines = {
        "uniform": float(np.log(len(chars))),
        "bigram": ngram_loss(train_data, val_data, len(chars), 2),
        "trigram": ngram_loss(train_data, val_data, len(chars), 3),
        "5-gram": ngram_loss(train_data, val_data, len(chars), 5),
    }
    print("Baselines (validation, nats per character):", {k: round(v, 3) for k, v in baselines.items()}, flush=True)

    config = Config(len(chars), args.block_size, args.n_embd, args.n_head, args.n_layer)
    model = GPT(config, seed=args.seed)
    optimizer = AdamW(model.params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    print(f"{len(text):,} characters, vocabulary {len(chars)}, {model.num_params():,} parameters", flush=True)

    history = []
    start = time.time()
    warmup = 200
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            row = {
                "step": step,
                "train": estimate_loss(model, train_data, 15, 32, 11),
                "val": estimate_loss(model, val_data, 15, 32, 22),
                "seconds": round(time.time() - start),
            }
            history.append(row)
            print(row, flush=True)
        if step == args.steps:
            break
        progress = step / args.steps
        lr = args.lr * (min(1.0, (step + 1) / warmup)) * (0.5 * (1 + np.cos(np.pi * progress)))
        optimizer.lr = lr
        x, y = batch(train_data, args.batch_size, args.block_size, rng)
        optimizer.zero_grad()
        _, loss = model.forward(x, y, rng)
        loss.backward()
        optimizer.step()

    prompt = np.array([[to_id[c] for c in "The "]])
    sample = "".join(chars[i] for i in model.generate(prompt, 500, 0.8, seed=args.seed)[0])
    best = min(history, key=lambda r: r["val"])
    results = {
        "data": args.data or "pydoc_data.topics (Python language reference)",
        "characters": len(text),
        "vocab": len(chars),
        "parameters": model.num_params(),
        "config": model.config_dict(),
        "steps": args.steps,
        "seconds": round(time.time() - start),
        "baselines_val": baselines,
        "final": history[-1],
        "best_val": best,
        "history": history,
        "sample": sample,
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    model.save(Path(args.out).with_suffix(".npz"))
    print("\nSAMPLE:\n" + sample)


if __name__ == "__main__":
    main()
