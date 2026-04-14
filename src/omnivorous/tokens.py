"""Token counting using tiktoken."""

from __future__ import annotations

import tiktoken

DEFAULT_ENCODING = "o200k_base"
SUPPORTED_ENCODINGS: set[str] = {"cl100k_base", "o200k_base"}

_encoding_name: str = DEFAULT_ENCODING
_encoding: tiktoken.Encoding | None = None


def set_encoding(name: str) -> None:
    """Set the tiktoken encoding to use for token counting."""
    global _encoding_name, _encoding
    if name not in SUPPORTED_ENCODINGS:
        raise ValueError(
            f"Unsupported encoding: {name!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_ENCODINGS))}"
        )
    _encoding_name = name
    _encoding = None  # reset cache


def reset_encoding() -> None:
    """Reset token counting to the built-in default encoding."""
    set_encoding(DEFAULT_ENCODING)


def get_encoding_name() -> str:
    """Return the current encoding name."""
    return _encoding_name


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(_encoding_name)
    return _encoding


def count_tokens(text: str) -> int:
    """Count the number of tokens in text using the current encoding."""
    return len(_get_encoding().encode(text))
