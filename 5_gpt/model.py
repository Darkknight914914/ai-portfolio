"""A GPT-style character model: embeddings, masked self-attention, MLP, repeat."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from autograd import Tensor, cross_entropy, dropout, embed


@dataclass
class Config:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 96
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.1


class GPT:
    def __init__(self, config: Config, seed: int = 0, dtype=np.float32):
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must divide evenly by n_head")
        self.config, self.dtype = config, dtype
        rng = np.random.default_rng(seed)
        self.params: dict[str, Tensor] = {}
        scaled = 0.02 / np.sqrt(2 * config.n_layer)  # keeps variance stable through residual additions

        def add(name, shape, std=0.02, zeros=False, ones=False):
            data = np.ones(shape) if ones else np.zeros(shape) if zeros else rng.normal(0, std, shape)
            self.params[name] = Tensor(data.astype(dtype))

        c = config
        add("tok", (c.vocab_size, c.n_embd))
        add("pos", (c.block_size, c.n_embd))
        for i in range(c.n_layer):
            add(f"h{i}.ln1.g", (c.n_embd,), ones=True)
            add(f"h{i}.ln1.b", (c.n_embd,), zeros=True)
            add(f"h{i}.qkv", (c.n_embd, 3 * c.n_embd))
            add(f"h{i}.proj", (c.n_embd, c.n_embd), std=scaled)
            add(f"h{i}.ln2.g", (c.n_embd,), ones=True)
            add(f"h{i}.ln2.b", (c.n_embd,), zeros=True)
            add(f"h{i}.mlp1", (c.n_embd, 4 * c.n_embd))
            add(f"h{i}.mlp1b", (4 * c.n_embd,), zeros=True)
            add(f"h{i}.mlp2", (4 * c.n_embd, c.n_embd), std=scaled)
        add("ln.g", (c.n_embd,), ones=True)
        add("ln.b", (c.n_embd,), zeros=True)
        add("head", (c.n_embd, c.vocab_size))
        self.future = {}

    def num_params(self) -> int:
        return sum(p.data.size for p in self.params.values())

    def _layernorm(self, x: Tensor, prefix: str) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centred = x - mean
        variance = (centred * centred).mean(axis=-1, keepdims=True)
        std = (variance + 1e-5).sqrt()
        return centred / std * self.params[f"{prefix}.g"] + self.params[f"{prefix}.b"]

    def _attention(self, x: Tensor, i: int, rng) -> Tensor:
        c = self.config
        batch, t, _ = x.shape
        head_dim = c.n_embd // c.n_head
        qkv = x @ self.params[f"h{i}.qkv"]

        def split(start: int) -> Tensor:
            part = qkv.slice_last(start * c.n_embd, (start + 1) * c.n_embd)
            return part.reshape(batch, t, c.n_head, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split(0), split(1), split(2)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(head_dim))
        if t not in self.future:
            self.future[t] = np.triu(np.ones((t, t), dtype=bool), 1)  # hides later characters
        weights = dropout(scores.masked_fill(self.future[t], -1e9).softmax(), c.dropout, rng)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, t, c.n_embd)
        return dropout(out @ self.params[f"h{i}.proj"], c.dropout, rng)

    def _block(self, x: Tensor, i: int, rng) -> Tensor:
        x = x + self._attention(self._layernorm(x, f"h{i}.ln1"), i, rng)
        hidden = (self._layernorm(x, f"h{i}.ln2") @ self.params[f"h{i}.mlp1"] + self.params[f"h{i}.mlp1b"]).gelu()
        return x + dropout(hidden @ self.params[f"h{i}.mlp2"], self.config.dropout, rng)

    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None, rng=None):
        """Pass a NumPy Generator as rng to turn dropout on (training)."""
        t = idx.shape[1]
        if t > self.config.block_size:
            raise ValueError("input is longer than block_size")
        x = embed(self.params["tok"], idx) + embed(self.params["pos"], np.arange(t))
        x = dropout(x, self.config.dropout, rng)
        for i in range(self.config.n_layer):
            x = self._block(x, i, rng)
        logits = self._layernorm(x, "ln") @ self.params["head"]
        return logits, (cross_entropy(logits, targets) if targets is not None else None)

    def generate(self, idx: np.ndarray, steps: int, temperature: float = 0.8, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        idx = idx.copy()
        for _ in range(steps):
            logits, _ = self.forward(idx[:, -self.config.block_size :])
            last = logits.data[:, -1, :].astype(np.float64) / temperature
            last -= last.max(axis=-1, keepdims=True)
            probs = np.exp(last)
            probs /= probs.sum(axis=-1, keepdims=True)
            next_ids = np.array([[rng.choice(len(p), p=p)] for p in probs])
            idx = np.concatenate([idx, next_ids], axis=1)
        return idx

    def save(self, path) -> None:
        np.savez(path, **{k: v.data for k, v in self.params.items()})

    def load(self, path) -> None:
        with np.load(path) as saved:
            for key, tensor in self.params.items():
                tensor.data = saved[key]

    def config_dict(self) -> dict:
        return asdict(self.config)


class AdamW:
    """Weight decay applies to matrices only, not to norms, biases or embeddings."""

    def __init__(self, params: dict[str, Tensor], lr=3e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        self.params, self.lr, self.betas, self.eps, self.weight_decay = params, lr, betas, eps, weight_decay
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self, clip: float = 1.0) -> float:
        self.t += 1
        norm = float(np.sqrt(sum(float((p.grad**2).sum()) for p in self.params.values() if p.grad is not None)))
        scale = min(1.0, clip / (norm + 1e-6))
        b1, b2 = self.betas
        for key, p in self.params.items():
            if p.grad is None:
                continue
            g = p.grad * scale
            self.m[key] = b1 * self.m[key] + (1 - b1) * g
            self.v[key] = b2 * self.v[key] + (1 - b2) * g * g
            m_hat = self.m[key] / (1 - b1**self.t)
            v_hat = self.v[key] / (1 - b2**self.t)
            decay = self.weight_decay * p.data if p.data.ndim == 2 and key not in ("tok", "pos") else 0.0
            p.data = p.data - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + decay)
        return norm

    def zero_grad(self) -> None:
        for p in self.params.values():
            p.grad = None
