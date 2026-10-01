# Changelog

All notable changes to Pinocchio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-01

### Fixed
- Read P(correct) from the answer tokens the calibrator was trained on (`i` / `ii`), not `(i` / `(ii`.
- Reproduce the training input: pair each prompt with the blank placeholder image used in training
  (`use_placeholder_image=True`, the default) and close any open `<think>` block at the generation prompt.
- Require `transformers>=5.2.0` (Qwen3.5 model class), `peft>=0.18.0`, `accelerate` and `pillow`.

### Added
- Initial release of the `pinocchio-uq` package.
- `Pinocchio` class with `score()` and `score_batch()` methods for estimating LLM response correctness.
- Direct integration with OpenAI `ChatCompletion` response objects.
- Support for raw question/answer string scoring.
- Two prompt variants: `"combined"` (with benchmark and model metadata) and `"baseline"` (question/answer only).
- Automatic truncation of long inputs to stay within model context limits.
- Automatic model weight download from HuggingFace Hub.
- CPU inference support (no GPU required for the 0.8B model).
- Optional dependencies for OpenAI integration (`pinocchio-uq[openai]`) and evaluation metrics (`pinocchio-uq[eval]`).
