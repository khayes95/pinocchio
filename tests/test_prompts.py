"""Tests for prompt formatting."""

import pytest

from pinocchio.prompts import TRUNCATION, format_prompt


def test_baseline_prompt():
    result = format_prompt(
        question="What is 2+2?",
        response="4",
        variant="baseline",
    )
    assert "Question: What is 2+2?" in result
    assert "Answer: 4" in result
    assert "(i) No (ii) Yes" in result
    # baseline should NOT include benchmark/source_model headers
    assert "Benchmark:" not in result


def test_combined_prompt():
    result = format_prompt(
        question="What is 2+2?",
        response="4",
        variant="combined",
        benchmark="gsm8k",
        source_model="gpt-5",
    )
    assert "Benchmark: gsm8k" in result
    assert "Source model: gpt-5" in result
    assert "Question: What is 2+2?" in result
    assert "Answer: 4" in result
    assert "(i) No (ii) Yes" in result


def test_truncation_baseline():
    long_q = "x" * 1000
    result = format_prompt(question=long_q, response="y", variant="baseline")
    # baseline truncates question to 500
    assert "x" * 500 in result
    assert "x" * 501 not in result


def test_truncation_combined():
    long_q = "x" * 2000
    result = format_prompt(question=long_q, response="y", variant="combined")
    # combined truncates question to 1500
    assert "x" * 1500 in result
    assert "x" * 1501 not in result


def test_truncation_response_baseline():
    long_r = "r" * 1000
    result = format_prompt(question="Q?", response=long_r, variant="baseline")
    r_len = TRUNCATION["baseline"][1]
    assert "r" * r_len in result
    assert "r" * (r_len + 1) not in result


def test_truncation_response_combined():
    long_r = "r" * 2000
    result = format_prompt(question="Q?", response=long_r, variant="combined")
    r_len = TRUNCATION["combined"][1]
    assert "r" * r_len in result
    assert "r" * (r_len + 1) not in result


def test_default_variant_is_combined():
    result = format_prompt(question="Q?", response="A")
    assert "Benchmark:" in result  # combined includes this header


def test_empty_question():
    result = format_prompt(question="", response="42", variant="baseline")
    assert "Question: " in result
    assert "Answer: 42" in result


def test_empty_response():
    result = format_prompt(question="Q?", response="", variant="baseline")
    assert "Answer: " in result


def test_empty_both():
    result = format_prompt(question="", response="", variant="baseline")
    assert "(i) No (ii) Yes" in result


def test_combined_empty_metadata():
    result = format_prompt(
        question="Q?", response="A", variant="combined",
        benchmark="", source_model="",
    )
    assert "Benchmark: \n" in result
    assert "Source model: \n" in result


def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        format_prompt(question="Q?", response="A", variant="invalid")


def test_invalid_variant_typo():
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        format_prompt(question="Q?", response="A", variant="Combined")


def test_special_characters_in_question():
    """Curly braces in user content must not cause format errors."""
    q = 'What is the result of {x: 1} if x = "hello"?'
    result = format_prompt(question=q, response="A", variant="baseline")
    assert "{x: 1}" in result


def test_curly_braces_in_response():
    """Code snippets with braces in the response must not cause errors."""
    r = 'def foo():\n    return {"key": "value"}'
    result = format_prompt(question="Q?", response=r, variant="baseline")
    assert '{"key": "value"}' in result


def test_newlines_in_question():
    q = "Line 1\nLine 2\nLine 3"
    result = format_prompt(question=q, response="A", variant="baseline")
    assert "Line 1\nLine 2\nLine 3" in result


def test_unicode_in_content():
    result = format_prompt(
        question="Qu'est-ce que c'est?",
        response="C'est un cafe.",
        variant="baseline",
    )
    assert "Qu'est-ce que c'est?" in result
