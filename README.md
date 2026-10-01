# Pinocchio

**Fast uncertainty estimates for black-box language models.**

Pinocchio is an external calibrator that estimates the correctness of responses from black-box API models. It needs only a single forward pass to generate an uncertainty estimate and requires no access to the target model's logits, weights, or internal states.

Paper: [arxiv.org/abs/2609.24881](https://arxiv.org/abs/2609.24881) · Project page: [pinocchio-uq.com](https://pinocchio-uq.com) · Accepted to COLM 2026

## Quickstart

```bash
pip install pinocchio-uq
```

```python
from openai import OpenAI
from pinocchio import Pinocchio

client = OpenAI()
judge = Pinocchio()  # load once

messages = [{"role": "user", "content": "What is the capital of France?"}]
response = client.chat.completions.create(model="gpt-5", messages=messages)

p_correct = judge.score(response, messages=messages)
```

`judge = Pinocchio()` and `judge.score(...)` are the two lines Pinocchio adds to existing code. The first call downloads the base model (`Qwen/Qwen3.5-0.8B`) and the Pinocchio adapter (`KevinDavidHayes/pinocchio-0.8b`) from the Hugging Face Hub, about 2 GB in total.

## What it is

- **Black-box.** Pinocchio reads the question and the response. It never touches the target model.
- **One forward pass.** No sampling, no logit access, and no modification of the target model.
- **Lightweight.** This package ships a text-only 0.8B checkpoint, which matches the AUROC of the paper's largest model. It runs on a CPU.

In the paper, Pinocchio reaches 0.863 AUROC predicting the correctness of held-out responses, compared with 0.649 for the strongest single-pass black-box baseline, and transfers zero-shot to thirteen unseen models across eight organizations (mean AUROC 0.814). Those results use the paper's 8B model; the text-only 0.8B checkpoint here reaches 0.867 AUROC on the held-out text responses.

## Usage

### Score an OpenAI response

```python
p_correct = judge.score(response, messages=messages)
```

### Score a question and answer from any source

```python
p_correct = judge.score(question="What is 2+2?", answer="4")
```

### Batch scoring

```python
items = [
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the capital of Japan?", "answer": "Kyoto"},
]
scores = judge.score_batch(items)
```

### Abstain or escalate when confidence is low

```python
if judge.score(response, messages=messages) < 0.5:
    escalate(response)  # e.g. route to a human or a stronger model
```

## How it works

Pinocchio is a small calibrator trained jointly on responses from seven LLMs that span a wide capability range. It takes the question, the response, and the identity of the model that produced it, and predicts a single token, `i` (incorrect) or `ii` (correct). The correctness probability is the softmax over those two logits:

P(correct) = exp(z_ii) / (exp(z_i) + exp(z_ii))

The logits are the calibrator's own; the target model is never inspected.

## API

### `Pinocchio(adapter, base_model, device_map, torch_dtype, use_placeholder_image)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adapter` | `"KevinDavidHayes/pinocchio-0.8b"` | Hugging Face Hub repo or local path of the adapter |
| `base_model` | `"Qwen/Qwen3.5-0.8B"` | Base model |
| `device_map` | `"auto"` | Device placement |
| `torch_dtype` | bfloat16 on GPU, float32 on CPU | Model precision |
| `use_placeholder_image` | `True` | Reproduce the training input, which paired every text example with a blank image |

### `judge.score(response=None, *, question, answer, messages, source_model, benchmark)`

Returns P(correct), a float in [0, 1].

| Parameter | Description |
|-----------|-------------|
| `response` | OpenAI `ChatCompletion`; the answer and model name are read from it |
| `question` | The question that was asked |
| `answer` | The answer to score, if `response` is not given |
| `messages` | OpenAI messages list; the last user message is used as the question |
| `source_model` | Optional name of the model that produced the answer |
| `benchmark` | Optional benchmark name |

### `judge.score_batch(items, prompt_variant="combined", show_progress=True)`

Returns a list of P(correct), one per item.

## Limitations

- **Distribution shift.** Pinocchio can be miscalibrated for data far from its training mix. The paper shows that recalibrating with about 100 labeled examples (Platt scaling) restores calibration on new targets without changing the ranking.
- **Easy questions.** Pinocchio was trained on challenging benchmarks where models frequently make mistakes. On trivially easy questions its scores are less reliable: in our tests, the wrong answer "5" to "What is 2+2?" still scored 0.97.
- **Text only.** This 0.8B release scores text questions and answers. Images in `messages` are ignored.
- **English.** The training data is predominantly English.

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

## License

Apache 2.0. See [LICENSE](LICENSE).
