"""Model loading and inference for Pinocchio."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None

from .prompts import format_prompt

logger = logging.getLogger(__name__)

# Default HuggingFace Hub repo for the LoRA adapter
DEFAULT_HUB_REPO = "KevinDavidHayes/pinocchio-0.8b"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"

# The calibrator predicts one answer token: "i" = incorrect, "ii" = correct.
TOKEN_INCORRECT = "i"
TOKEN_CORRECT = "ii"

# During training, every text-only example was paired with a blank gray image at a fixed
# resolution. Scoring reproduces that input so the calibrator sees what it was trained on.
PLACEHOLDER_IMAGE_SIZE = (336, 336)
PLACEHOLDER_PIXELS = 256 * 28 * 28


def load_model(
    adapter: str = DEFAULT_HUB_REPO,
    base_model: str = DEFAULT_BASE_MODEL,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> tuple:
    """Load the Pinocchio model (base + LoRA adapter).

    Args:
        adapter: Path or HuggingFace Hub repo ID for the LoRA adapter.
        base_model: Base model name or path. Defaults to Qwen3.5-0.8B.
        device_map: Device mapping strategy. Defaults to "auto".
        torch_dtype: Torch dtype. Defaults to bfloat16 if supported, else float16.

    Returns:
        Tuple of (model, tokenizer, device).
    """
    if device_map == "auto" and not torch.cuda.is_available():
        # Without CUDA, "auto" can place the model on Apple's MPS backend, where loading stalls. Use the CPU.
        device_map = "cpu"

    if torch_dtype is None:
        # Guard against CPU-only environments where cuda.is_bf16_supported() would fail
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    logger.info("Loading base model: %s", base_model)
    # Qwen3.5 is a VLM (ConditionalGeneration), not a pure CausalLM.
    # We must load the correct architecture so LoRA weight keys match.
    if Qwen3_5ForConditionalGeneration is not None and "Qwen3.5" in base_model:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    logger.info("Loading LoRA adapter: %s", adapter)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    device = next(model.parameters()).device
    logger.info("Model loaded on device: %s", device)

    return model, tokenizer, device


def load_processor(base_model: str = DEFAULT_BASE_MODEL):
    """Load the base model's processor, used to reproduce the training input. Returns None if unavailable."""
    if AutoProcessor is None:
        return None
    try:
        return AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - depends on the installed transformers
        logger.warning("Could not load a processor for %s (%s); scoring text without the placeholder image.",
                       base_model, e)
        return None


def _close_think_block(text: str) -> str:
    """Close an open <think> block at the generation prompt, as during training and evaluation."""
    if text.rstrip().endswith("<think>"):
        return text.rstrip() + "\n\n</think>\n\n"
    return text


def score_single(
    model,
    tokenizer,
    device,
    question: str,
    response: str,
    prompt_variant: str = "combined",
    benchmark: str = "",
    source_model: str = "",
    processor=None,
) -> float:
    """Score a single question-answer pair.

    Args:
        model: The loaded Pinocchio model.
        tokenizer: The tokenizer.
        device: Torch device.
        question: The question that was asked.
        response: The model's answer to evaluate.
        prompt_variant: "combined" (default) or "baseline".
        benchmark: Optional benchmark name for metadata-aware scoring.
        source_model: Optional source model name for metadata-aware scoring.
        processor: Optional processor. When given, the prompt is paired with the gray placeholder
            image used in training; otherwise the prompt is scored as text alone.

    Returns:
        P(correct) — a float in [0, 1].
    """
    prompt = format_prompt(
        question=question,
        response=response,
        variant=prompt_variant,
        benchmark=benchmark,
        source_model=source_model,
    )

    if processor is not None:
        from PIL import Image

        image = Image.new("RGB", PLACEHOLDER_IMAGE_SIZE, color="gray")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = _close_think_block(
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            # shortest_edge / longest_edge are the min / max pixel counts for Qwen vision processors
            size={"shortest_edge": PLACEHOLDER_PIXELS, "longest_edge": PLACEHOLDER_PIXELS},
        )
    else:
        messages = [{"role": "user", "content": prompt}]
        text = _close_think_block(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :].float()

    token_incorrect = tokenizer.encode(TOKEN_INCORRECT, add_special_tokens=False)[-1]
    token_correct = tokenizer.encode(TOKEN_CORRECT, add_special_tokens=False)[-1]

    probs = torch.softmax(logits[[token_incorrect, token_correct]], dim=0)
    return probs[1].item()  # P(correct) = P("ii")
