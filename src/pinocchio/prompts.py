"""Prompt templates for Pinocchio UQ scoring."""

PROMPT_BASELINE = """Question: {question}

Answer: {response}

Is the answer correct? (i) No (ii) Yes"""

PROMPT_COMBINED = """Benchmark: {benchmark}
Source model: {source_model}

Question: {question}

Answer: {response}

Analyze whether the answer above is correct. Consider:
- Does the answer address the question?
- Are there factual errors or logical flaws?
- Is the answer complete?

Based on your analysis, is the answer correct? (i) No (ii) Yes"""

# Truncation lengths per prompt variant: (question_len, response_len)
TRUNCATION = {
    "baseline": (500, 300),
    "combined": (1500, 800),
}

DEFAULT_VARIANT = "combined"


def format_prompt(
    question: str,
    response: str,
    variant: str = DEFAULT_VARIANT,
    benchmark: str = "",
    source_model: str = "",
) -> str:
    """Format a scoring prompt with appropriate truncation.

    Args:
        question: The question that was asked.
        response: The model's answer to evaluate.
        variant: Prompt variant — "combined" (default, recommended) or "baseline".
        benchmark: Optional benchmark name (used by "combined" variant).
        source_model: Optional source model name (used by "combined" variant).

    Returns:
        Formatted prompt string.
    """
    valid_variants = {"combined", "baseline"}
    if variant not in valid_variants:
        raise ValueError(
            f"Unknown prompt variant {variant!r}. Must be one of: {', '.join(sorted(valid_variants))}"
        )

    q_len, r_len = TRUNCATION[variant]

    if variant == "combined":
        template = PROMPT_COMBINED
    else:
        template = PROMPT_BASELINE

    # Use manual replacement instead of str.format() to avoid KeyError/ValueError
    # when user content contains curly braces (e.g., code snippets like {x: 1}).
    result = template.replace("{question}", question[:q_len])
    result = result.replace("{response}", response[:r_len])
    result = result.replace("{benchmark}", benchmark)
    result = result.replace("{source_model}", source_model)
    return result
