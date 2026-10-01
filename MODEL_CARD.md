---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-0.8B
tags:
  - uncertainty-quantification
  - calibration
  - llm
language:
  - en
---

# Pinocchio 0.8B (text)

Pinocchio is an external calibrator that estimates the correctness of responses from black-box API models. It needs only a single forward pass to generate an uncertainty estimate and requires no access to the target model's logits, weights, or internal states.

This repository holds the lightweight text-only 0.8B checkpoint: a LoRA adapter for [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).

- Paper: [arxiv.org/abs/2609.24881](https://arxiv.org/abs/2609.24881) (COLM 2026)
- Project page: [pinocchio-uq.com](https://pinocchio-uq.com)
- Package: [pypi.org/project/pinocchio-uq](https://pypi.org/project/pinocchio-uq/)

## Usage

```bash
pip install pinocchio-uq
```

```python
from pinocchio import Pinocchio

judge = Pinocchio()  # loads Qwen/Qwen3.5-0.8B and this adapter

response = client.chat.completions.create(model="gpt-5", messages=messages)
p_correct = judge.score(response, messages=messages)
```

## How it works

The calibrator reads the question and response (plus optional benchmark and model-identity metadata) and predicts a single token, `i` (incorrect) or `ii` (correct). The correctness probability is the softmax over those two logits. During training, every text example was paired with a blank placeholder image; the `pinocchio-uq` package reproduces that input.

## Training

| | |
|---|---|
| Base model | Qwen/Qwen3.5-0.8B |
| Method | LoRA (rank 16), 3 epochs |
| Source models | Claude Fable 5, Claude Opus 5, GPT-5.6, Kimi 3, GPT-5-mini, GPT-5.2, Qwen3.5-397B |
| Benchmarks | The 9 text benchmarks of the paper's training set: ARC-AGI, BBEH, ChemBench, GPQA Diamond, HLE, LiveBench, OmniMath, PRBench, SimpleQA |
| Split | Question-level held-out split, the same as the paper's largest model |

## Evaluation

On held-out text responses from the seven source models (n = 2,293):

| Metric | Value |
|---|---|
| AUROC | 0.867 |
| ECE | 0.080 |

As the paper reports, this lightweight text-only checkpoint matches the AUROC of the largest model. Results for vision-language inputs, transfer to unseen models, and baselines are in the paper.

## Limitations

- Text only: images are not used.
- Pinocchio can be miscalibrated for data far from its training mix; the paper shows that recalibrating with about 100 labeled examples (Platt scaling) restores calibration without changing the ranking.
- The training data is predominantly English.

## Citation

```bibtex
@inproceedings{hayes2026pinocchio,
  title     = {Pinocchio: Fast Uncertainty Estimates for Black-Box Language Models},
  author    = {Hayes, Kevin David and Pal, Arka and Zhang, Haosong and
               Goldstein, Tom and Goldblum, Micah},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```
