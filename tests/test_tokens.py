"""Tests for token counting."""

import pytest

from omnivorous.tokens import (
    DEFAULT_ENCODING,
    count_tokens,
    get_encoding_name,
    reset_encoding,
    set_encoding,
)


@pytest.fixture(autouse=True)
def _reset_encoding():
    """Reset encoding to default before each test."""
    reset_encoding()
    yield
    reset_encoding()


def test_count_tokens_basic():
    count = count_tokens("Hello, world!")
    assert count > 0
    assert isinstance(count, int)


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_longer():
    text = "The quick brown fox jumps over the lazy dog. " * 10
    count = count_tokens(text)
    assert count > 50


def test_set_encoding_and_get():
    set_encoding(DEFAULT_ENCODING)
    assert get_encoding_name() == DEFAULT_ENCODING

    set_encoding("cl100k_base")
    assert get_encoding_name() == "cl100k_base"


def test_different_encodings_produce_different_counts():
    # Use text with varied vocabulary to maximize encoding differences
    text = (
        "Implementing the asynchronous WebSocket middleware for the "
        "containerized microservices architecture required refactoring "
        "the authentication serialization layer. "
    ) * 20
    set_encoding("cl100k_base")
    count_cl100k = count_tokens(text)

    set_encoding(DEFAULT_ENCODING)
    count_o200k = count_tokens(text)

    # Both should produce valid counts, but they should differ
    assert count_cl100k > 0
    assert count_o200k > 0
    assert count_cl100k != count_o200k


def test_reset_encoding_restores_default():
    set_encoding("cl100k_base")
    reset_encoding()

    assert get_encoding_name() == DEFAULT_ENCODING


def test_invalid_encoding_raises():
    with pytest.raises(ValueError, match="Unsupported encoding"):
        set_encoding("invalid_encoding")
