"""Pinocchio: Estimate the uncertainty of any LLM in a single forward pass.

Usage with the OpenAI API::

    from openai import OpenAI
    from pinocchio import Pinocchio

    client = OpenAI()
    judge = Pinocchio()

    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = client.chat.completions.create(model="gpt-5", messages=messages)

    score = judge.score(response, messages=messages)
    print(f"P(correct) = {score:.3f}")
"""

from __future__ import annotations

__version__ = "0.1.0"

import logging
from typing import Optional

from .model import DEFAULT_BASE_MODEL, DEFAULT_HUB_REPO, load_model, load_processor, score_single

logger = logging.getLogger(__name__)


def _extract_from_openai_response(response) -> tuple[str, str, str]:
    """Extract question and answer from an OpenAI ChatCompletion response.

    Returns (question, answer, model_name).
    """
    # Get the answer from the first choice
    answer = response.choices[0].message.content or ""

    # Get model name
    model_name = getattr(response, "model", "")

    # Try to get the question from the request messages stored on the response
    # OpenAI responses don't include the input messages, so we return empty
    question = ""

    return question, answer, model_name


class Pinocchio:
    """Score LLM outputs for correctness using a fine-tuned uncertainty judge.

    Returns P(correct) — a calibrated probability that a given answer is correct.
    Works on any model's outputs (GPT-5, Claude, Qwen, LLaMA, etc.) without
    needing access to that model's internals.

    Args:
        adapter: HuggingFace Hub repo or local path for the LoRA adapter.
        base_model: Base model name. Defaults to ``Qwen/Qwen3.5-0.8B``.
        device_map: Device mapping strategy. Defaults to ``"auto"``.
        torch_dtype: Override torch dtype (default: bfloat16 if supported).
        use_placeholder_image: Pair each prompt with the gray placeholder image the calibrator saw
            during training. Defaults to ``True``.

    Example::

        from openai import OpenAI
        from pinocchio import Pinocchio

        client = OpenAI()
        judge = Pinocchio()

        messages = [{"role": "user", "content": "What is 2+2?"}]
        response = client.chat.completions.create(model="gpt-5", messages=messages)

        score = judge.score(response, messages=messages)
    """

    def __init__(
        self,
        adapter: str = DEFAULT_HUB_REPO,
        base_model: str = DEFAULT_BASE_MODEL,
        device_map: str = "auto",
        torch_dtype=None,
        use_placeholder_image: bool = True,
    ):
        self._model, self._tokenizer, self._device = load_model(
            adapter=adapter,
            base_model=base_model,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self._processor = load_processor(base_model) if use_placeholder_image else None

    def score(
        self,
        response=None,
        *,
        question: str = "",
        answer: str = "",
        messages: Optional[list[dict]] = None,
        source_model: str = "",
        benchmark: str = "",
        prompt_variant: str = "combined",
    ) -> float:
        """Score an LLM response for correctness.

        Accepts an OpenAI response object directly, or raw question + answer strings.

        Args:
            response: An OpenAI ``ChatCompletion`` response object.
                If provided, the answer is extracted automatically.
                You should also pass ``messages`` or ``question`` so the judge
                knows what was asked.
            question: The question that was asked. Either pass this directly,
                or pass ``messages`` to extract it.
            answer: The model's answer. Not needed if ``response`` is provided.
            messages: The OpenAI messages list (e.g. ``[{"role": "user", "content": "..."}]``).
                Used to extract the question when scoring an API response.
            source_model: Name of the model that produced the answer (optional).
            benchmark: Benchmark name if applicable (optional).
            prompt_variant: ``"combined"`` (default) or ``"baseline"``.

        Returns:
            P(correct) — a float in [0, 1].

        Examples::

            # From OpenAI response + messages
            msgs = [{"role": "user", "content": "What is 2+2?"}]
            resp = client.chat.completions.create(model="gpt-5", messages=msgs)
            score = judge.score(resp, messages=msgs)

            # From raw strings
            score = judge.score(question="What is 2+2?", answer="4")

            # From OpenAI response with question
            score = judge.score(resp, question="What is 2+2?")
        """
        # Extract answer and model from OpenAI response object
        if response is not None and hasattr(response, "choices"):
            _, resp_answer, resp_model = _extract_from_openai_response(response)
            if not answer:
                answer = resp_answer
            if not source_model:
                source_model = resp_model

        # Extract question from messages list
        if not question and messages:
            question = _extract_question_from_messages(messages)

        if not answer:
            raise ValueError(
                "No answer provided. Pass an OpenAI response object or answer='...'."
            )

        return score_single(
            self._model,
            self._tokenizer,
            self._device,
            question=question,
            response=answer,
            prompt_variant=prompt_variant,
            benchmark=benchmark,
            source_model=source_model,
            processor=getattr(self, "_processor", None),
        )

    def score_batch(
        self,
        items: list[dict],
        prompt_variant: str = "combined",
        show_progress: bool = True,
    ) -> list[float]:
        """Score a batch of question-answer pairs.

        Args:
            items: List of dicts. Each dict can contain:
                - ``response``: OpenAI ChatCompletion object, OR
                - ``question`` (str) + ``answer`` (str)
                - ``messages`` (list[dict], optional): to extract the question
                - ``source_model`` (str, optional)
                - ``benchmark`` (str, optional)
            prompt_variant: ``"combined"`` (default) or ``"baseline"``.
            show_progress: Show a tqdm progress bar. Defaults to True.

        Returns:
            List of P(correct) floats, one per item.
        """
        scores = []
        items_iter = items

        if show_progress:
            try:
                from tqdm import tqdm

                items_iter = tqdm(items, desc="Scoring", unit="sample")
            except ImportError:
                pass

        for item in items_iter:
            p = self.score(
                response=item.get("response"),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                messages=item.get("messages"),
                source_model=item.get("source_model", ""),
                benchmark=item.get("benchmark", ""),
                prompt_variant=prompt_variant,
            )
            scores.append(p)

        return scores


def _extract_question_from_messages(messages: list[dict]) -> str:
    """Extract the user's question from an OpenAI messages list."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle content as list of parts (vision messages, etc.)
            if isinstance(content, list):
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return " ".join(texts)
    return ""


__all__ = ["Pinocchio", "__version__"]
