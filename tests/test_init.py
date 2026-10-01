"""Tests for package initialization, imports, and the Pinocchio class interface."""

from unittest.mock import MagicMock, patch

import pytest


def test_version():
    from pinocchio import __version__

    assert __version__ == "0.1.0"


def test_pinocchio_class_importable():
    from pinocchio import Pinocchio

    assert Pinocchio is not None


def test_all_exports():
    import pinocchio

    assert "Pinocchio" in pinocchio.__all__
    assert "__version__" in pinocchio.__all__


# --- _extract_question_from_messages ---


def test_extract_question_from_messages():
    from pinocchio import _extract_question_from_messages

    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert _extract_question_from_messages(msgs) == "What is 2+2?"


def test_extract_question_from_messages_multipart():
    from pinocchio import _extract_question_from_messages

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
            ],
        },
    ]
    assert _extract_question_from_messages(msgs) == "Describe this image."


def test_extract_question_empty_messages():
    from pinocchio import _extract_question_from_messages

    assert _extract_question_from_messages([]) == ""


def test_extract_question_no_user_message():
    from pinocchio import _extract_question_from_messages

    msgs = [{"role": "system", "content": "You are helpful."}]
    assert _extract_question_from_messages(msgs) == ""


def test_extract_question_multiple_user_messages():
    """Should extract the LAST user message."""
    from pinocchio import _extract_question_from_messages

    msgs = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up question"},
    ]
    assert _extract_question_from_messages(msgs) == "Follow-up question"


def test_extract_question_missing_content_key():
    from pinocchio import _extract_question_from_messages

    msgs = [{"role": "user"}]
    assert _extract_question_from_messages(msgs) == ""


def test_extract_question_none_content():
    from pinocchio import _extract_question_from_messages

    msgs = [{"role": "user", "content": None}]
    # content is not a str and not a list, so falls through
    assert _extract_question_from_messages(msgs) == ""


def test_extract_question_multipart_no_text():
    from pinocchio import _extract_question_from_messages

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
            ],
        },
    ]
    assert _extract_question_from_messages(msgs) == ""


# --- _extract_from_openai_response ---


def test_extract_from_openai_response():
    from pinocchio import _extract_from_openai_response

    choice = MagicMock()
    choice.message.content = "Paris"

    response = MagicMock()
    response.choices = [choice]
    response.model = "gpt-5"

    question, answer, model = _extract_from_openai_response(response)
    assert answer == "Paris"
    assert model == "gpt-5"


def test_extract_from_openai_response_none_content():
    from pinocchio import _extract_from_openai_response

    choice = MagicMock()
    choice.message.content = None

    response = MagicMock()
    response.choices = [choice]
    response.model = "gpt-5"

    _, answer, _ = _extract_from_openai_response(response)
    assert answer == ""


def test_extract_from_openai_response_no_model():
    from pinocchio import _extract_from_openai_response

    choice = MagicMock()
    choice.message.content = "Paris"

    response = MagicMock(spec=[])  # no attributes at all
    response.choices = [choice]

    _, _, model = _extract_from_openai_response(response)
    assert model == ""


# --- Pinocchio.score validation ---


def test_score_no_answer_raises():
    """score() with no answer and no response object should raise ValueError."""
    from pinocchio import Pinocchio

    with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
        judge = Pinocchio()
        judge._model = None
        judge._tokenizer = None
        judge._device = None

        with pytest.raises(ValueError, match="No answer provided"):
            judge.score(question="What is 2+2?")


def test_score_empty_answer_raises():
    """score() with explicit empty answer string should raise ValueError."""
    from pinocchio import Pinocchio

    with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
        judge = Pinocchio()
        judge._model = None
        judge._tokenizer = None
        judge._device = None

        with pytest.raises(ValueError, match="No answer provided"):
            judge.score(question="What is 2+2?", answer="")


def test_score_extracts_from_response_object():
    """score() should extract answer from OpenAI response object."""
    from pinocchio import Pinocchio

    with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
        judge = Pinocchio()
        judge._model = MagicMock()
        judge._tokenizer = MagicMock()
        judge._device = "cpu"

        choice = MagicMock()
        choice.message.content = "4"
        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-5"

        messages = [{"role": "user", "content": "What is 2+2?"}]

        with patch("pinocchio.score_single", return_value=0.95) as mock_score:
            result = judge.score(response, messages=messages)
            assert result == 0.95
            # Verify score_single was called with the extracted answer
            call_kwargs = mock_score.call_args
            assert call_kwargs[1]["response"] == "4"
            assert call_kwargs[1]["question"] == "What is 2+2?"
            assert call_kwargs[1]["source_model"] == "gpt-5"


def test_score_batch_empty_list():
    """score_batch with empty list should return empty list."""
    from pinocchio import Pinocchio

    with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
        judge = Pinocchio()
        judge._model = None
        judge._tokenizer = None
        judge._device = None

        result = judge.score_batch([], show_progress=False)
        assert result == []


def test_score_batch_calls_score_per_item():
    """score_batch should call score once per item."""
    from pinocchio import Pinocchio

    with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
        judge = Pinocchio()
        judge._model = None
        judge._tokenizer = None
        judge._device = None

        items = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
            {"question": "Q3?", "answer": "A3"},
        ]

        with patch.object(judge, "score", return_value=0.5) as mock_score:
            result = judge.score_batch(items, show_progress=False)
            assert len(result) == 3
            assert mock_score.call_count == 3
