"""Numerical gradient checks: the proof that hand-written backprop is correct."""
import numpy as np
import pytest

from autograd import Tensor, cross_entropy, embed
from model import GPT, AdamW, Config


def numeric(f, x: Tensor, eps=1e-6, samples=40, seed=0):
    """Central differences on a random sample of entries."""
    rng = np.random.default_rng(seed)
    flat = x.data.reshape(-1)
    picks = rng.choice(flat.size, min(samples, flat.size), replace=False)
    out = {}
    for i in picks:
        keep = flat[i]
        flat[i] = keep + eps
        up = float(f().data)
        flat[i] = keep - eps
        down = float(f().data)
        flat[i] = keep
        out[i] = (up - down) / (2 * eps)
    return out


def assert_grad(f, *inputs):
    for x in inputs:
        x.grad = None
    f().backward()
    for x in inputs:
        for i, value in numeric(f, x).items():
            assert x.grad.reshape(-1)[i] == pytest.approx(value, rel=1e-4, abs=1e-6)


def t(shape, seed=0):
    return Tensor(np.random.default_rng(seed).normal(size=shape))


@pytest.mark.parametrize(
    "name,f",
    [
        ("add_broadcast", lambda a, b: (a + b.sum(axis=0, keepdims=True)).sum()),
        ("mul", lambda a, b: (a * b).sum()),
        ("matmul", lambda a, b: (a @ b.transpose(1, 0)).sum()),
        ("reshape", lambda a, b: (a.reshape(6, 2) * b.reshape(6, 2)).sum()),
        ("mean", lambda a, b: (a.mean(axis=-1, keepdims=True) * b).sum()),
        ("slice", lambda a, b: (a.slice_last(1, 3) * b.slice_last(0, 2)).sum()),
        ("gelu", lambda a, b: (a.gelu() * b).sum()),
        ("softmax", lambda a, b: (a.softmax() * b).sum()),
        ("masked", lambda a, b: (a.masked_fill(np.eye(3, 4, dtype=bool), -5.0).softmax() * b).sum()),
    ],
)
def test_operation_gradients(name, f):
    a, b = t((3, 4), 1), t((3, 4), 2)
    assert_grad(lambda: f(a, b), a, b)


def test_sqrt_and_divide():
    a, b = Tensor(np.array([1.5, 2.5, 3.0])), Tensor(np.array([0.5, 2.0, 4.0]))
    assert_grad(lambda: (a.sqrt() / b).sum(), a, b)


def test_embedding_with_repeated_indices():
    weights = t((5, 3))
    idx = np.array([[0, 1, 1, 4]])
    assert_grad(lambda: (embed(weights, idx) * embed(weights, idx)).sum(), weights)


def test_cross_entropy():
    logits = t((2, 3, 5))
    y = np.array([[0, 1, 2], [4, 3, 3]])
    assert_grad(lambda: cross_entropy(logits, y), logits)
    probs = np.exp(logits.data) / np.exp(logits.data).sum(-1, keepdims=True)
    expected = -np.log(probs[np.arange(2)[:, None], np.arange(3), y]).mean()
    assert float(cross_entropy(logits, y).data) == pytest.approx(expected)


def tiny(dtype=np.float64, **kw):
    settings = dict(vocab_size=7, block_size=6, n_embd=8, n_head=2, n_layer=2, dropout=0.0)
    settings.update(kw)
    return GPT(Config(**settings), seed=3, dtype=dtype)


def test_full_model_gradients():
    model = tiny()
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 7, (2, 6)), rng.integers(0, 7, (2, 6))

    def loss():
        return model.forward(x, y)[1]

    loss().backward()
    for name, param in model.params.items():
        assert param.grad is not None, name
    for name in ("tok", "pos", "h0.qkv", "h1.proj", "h0.ln1.g", "h1.mlp2", "h0.mlp1b", "head"):
        param = model.params[name]
        grad = param.grad.copy()
        for i, value in numeric(loss, param, samples=8).items():
            assert grad.reshape(-1)[i] == pytest.approx(value, rel=1e-3, abs=1e-6), name


def test_attention_cannot_see_the_future():
    model = tiny()
    a = np.array([[1, 2, 3, 4, 5, 6]])
    b = np.array([[1, 2, 3, 0, 0, 0]])
    logits_a = model.forward(a)[0].data
    logits_b = model.forward(b)[0].data
    assert np.allclose(logits_a[0, :3], logits_b[0, :3])
    assert not np.allclose(logits_a[0, 3], logits_b[0, 3])


def test_starting_loss_is_about_uniform():
    model = tiny()
    x = np.zeros((4, 6), dtype=int)
    y = np.ones((4, 6), dtype=int)
    assert float(model.forward(x, y)[1].data) == pytest.approx(np.log(7), abs=0.3)


def test_can_memorise_a_small_batch():
    model = tiny(np.float32, dropout=0.0)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 7, (4, 6)), rng.integers(0, 7, (4, 6))
    optimizer = AdamW(model.params, lr=5e-3, weight_decay=0.0)
    for _ in range(300):
        optimizer.zero_grad()
        loss = model.forward(x, y)[1]
        loss.backward()
        optimizer.step()
    assert float(loss.data) < 0.1


def test_generate_is_reproducible_and_respects_block_size():
    model = tiny()
    start = np.array([[1, 2]])
    one, two = model.generate(start, 12, seed=5), model.generate(start, 12, seed=5)
    assert one.shape == (1, 14) and (one == two).all()


def test_save_and_load(tmp_path):
    a, b = tiny(), GPT(tiny().config, seed=99, dtype=np.float64)
    a.save(tmp_path / "m.npz")
    b.load(tmp_path / "m.npz")
    x = np.array([[1, 2, 3]])
    assert np.allclose(a.forward(x)[0].data, b.forward(x)[0].data)
