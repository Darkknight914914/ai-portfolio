# 5. A GPT from scratch

A small character-level transformer written with NumPy only, including its own autograd engine: the same architecture as Karpathy's "Let's build GPT" video, but without PyTorch doing the backpropagation.

- `autograd.py`: reverse-mode autograd, about 200 lines
- `model.py`: embeddings, masked multi-head attention, layer norm, MLP, AdamW
- `train.py`: training, baselines and text generation
- `test_gpt.py`: gradient checks against numerical differences

```
python -m pytest
python train.py --steps 2500                 # about 20 minutes on 2 CPU cores
python train.py --data input.txt             # any text, e.g. Tiny Shakespeare
```

## Results
Trained on Python's language reference (560,058 characters, 118-character vocabulary), split 90/10. The model has 363,072 parameters: 3 layers, 4 heads, 96-dimensional embeddings, 64-character context. Training took 19 minutes on 2 CPU cores.

| Model | Validation loss (nats per character) |
|---|---|
| Uniform guess | 4.77 |
| Bigram counts | 2.39 |
| Trigram counts | 1.80 |
| 5-gram counts | 1.49 |
| **This GPT, 2,500 steps** | **1.35** |

Loss here is on the whole validation set, the same characters as the baselines. During training, train loss reached 1.22, so it was starting to overfit slightly.

## Sample output (prompt: "The ")
```
The setatement
       else to "except".                                                              | This method in the must with "guard" is a class result as unlooking
will called indisplay result is the looks are from allows, or instance.

   It class a pass can be pattern the line define.   That last return the "s
```

## How to read these honestly
- **It's a learning project, not a useful model.** It writes text that looks like Python docs: real words and structure, but no meaning. That's expected from a model this small.
- **Beating a 5-gram is the meaningful test.** A model that can't beat simple counting isn't learning anything useful.
- **It isn't PyTorch.** PyTorch couldn't be installed where this was built, so the engine was written by hand. That makes it slower, but the gradients are verified by tests.
- **It's character-level** with no tokenizer, and this is one training run with one seed.

## The tests that prove it's correct
- Gradients for every operation, and for the full model, match numerical differences.
- Changing later characters doesn't change earlier predictions, so the attention mask works.
- Starting loss is close to ln(vocabulary), as it should be for an untrained model.
- It can memorise a small batch, which shows that training works at all.
