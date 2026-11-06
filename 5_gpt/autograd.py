"""A small reverse-mode autograd engine on NumPy: the part PyTorch normally gives you.

Each operation returns a new Tensor and remembers how to send gradients back to
its inputs. backward() walks the graph in reverse order and calls those functions.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Undo NumPy broadcasting, so a gradient matches its input's shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "parents", "backward_fn")

    def __init__(self, data, parents: Iterable["Tensor"] = (), backward_fn: Callable | None = None):
        self.data = np.asarray(data)
        self.grad: np.ndarray | None = None
        self.parents = tuple(parents)
        self.backward_fn = backward_fn

    @property
    def shape(self):
        return self.data.shape

    def _add_grad(self, grad: np.ndarray) -> None:
        grad = _unbroadcast(grad, self.data.shape)
        self.grad = grad.copy() if self.grad is None else self.grad + grad

    def backward(self) -> None:
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node.parents)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            if node.backward_fn is not None:
                node.backward_fn(node.grad)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        def back(g):
            self._add_grad(g)
            other._add_grad(g)

        return Tensor(self.data + other.data, (self, other), back)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        def back(g):
            self._add_grad(g * other.data)
            other._add_grad(g * self.data)

        return Tensor(self.data * other.data, (self, other), back)

    def __matmul__(self, other):
        def back(g):
            self._add_grad(g @ np.swapaxes(other.data, -1, -2))
            other._add_grad(np.swapaxes(self.data, -1, -2) @ g)

        return Tensor(self.data @ other.data, (self, other), back)

    def reshape(self, *shape):
        def back(g):
            self._add_grad(g.reshape(self.data.shape))

        return Tensor(self.data.reshape(*shape), (self,), back)

    def transpose(self, *axes):
        inverse = np.argsort(axes)

        def back(g):
            self._add_grad(g.transpose(*inverse))

        return Tensor(self.data.transpose(*axes), (self,), back)

    def slice_last(self, start: int, stop: int):
        """Take a range from the last axis. Everything outside it gets zero gradient."""

        def back(g):
            full = np.zeros_like(self.data)
            full[..., start:stop] = g
            self._add_grad(full)

        return Tensor(self.data[..., start:stop], (self,), back)

    def sum(self, axis=None, keepdims=False):
        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._add_grad(np.broadcast_to(g, self.data.shape))

        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), back)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-(other if isinstance(other, Tensor) else Tensor(other)))

    def sqrt(self):
        root = np.sqrt(self.data)

        def back(g):
            self._add_grad(g * 0.5 / root)

        return Tensor(root, (self,), back)

    def __truediv__(self, other):
        def back(g):
            self._add_grad(g / other.data)
            other._add_grad(-g * self.data / other.data**2)

        return Tensor(self.data / other.data, (self, other), back)

    def gelu(self):
        # Fast approximation from the GELU paper, as used in GPT-style models.
        x = self.data
        gate = 1.0 / (1.0 + np.exp(np.clip(-1.702 * x, -50, 50)))

        def back(g):
            self._add_grad(g * (gate + x * 1.702 * gate * (1.0 - gate)))

        return Tensor(x * gate, (self,), back)

    def softmax(self):
        """Over the last axis, stable against very negative inputs."""
        shifted = self.data - self.data.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs = probs / probs.sum(axis=-1, keepdims=True)

        def back(g):
            inner = (g * probs).sum(axis=-1, keepdims=True)
            self._add_grad(probs * (g - inner))

        return Tensor(probs, (self,), back)

    def masked_fill(self, mask: np.ndarray, value: float):
        def back(g):
            self._add_grad(np.where(mask, 0.0, g))

        return Tensor(np.where(mask, value, self.data), (self,), back)


def embed(weights: Tensor, indices: np.ndarray) -> Tensor:
    def back(g):
        grad = np.zeros_like(weights.data)
        np.add.at(grad, indices.ravel(), g.reshape(-1, weights.data.shape[-1]))
        weights._add_grad(grad)

    return Tensor(weights.data[indices], (weights,), back)


def dropout(x: Tensor, rate: float, rng: np.random.Generator | None) -> Tensor:
    if rate <= 0 or rng is None:
        return x
    keep = rng.random(x.data.shape, dtype=x.data.dtype if x.data.dtype in (np.float32, np.float64) else np.float64) >= rate
    return x * (keep / (1.0 - rate)).astype(x.data.dtype)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean loss over all positions. logits: (..., vocab). targets: (...)."""
    vocab = logits.data.shape[-1]
    flat = logits.data.reshape(-1, vocab)
    y = targets.ravel()
    shifted = flat - flat.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_norm
    n = len(y)
    loss = -log_probs[np.arange(n), y].mean()

    def back(g):
        probs = np.exp(log_probs)
        probs[np.arange(n), y] -= 1.0
        logits._add_grad((g * probs / n).reshape(logits.data.shape))

    return Tensor(np.asarray(loss, dtype=logits.data.dtype), (logits,), back)
