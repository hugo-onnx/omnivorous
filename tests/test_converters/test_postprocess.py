"""Tests for PDF post-processing utilities."""

from omnivorous.converters.pdf._postprocess import strip_toc


def test_strip_toc_no_toc_returns_unchanged():
    text = "# Introduction\n\nSome content about the topic.\n"
    assert strip_toc(text) == text


def test_strip_toc_heading_table_of_contents():
    text = (
        "# Title\n\n"
        "## Table of Contents\n\n"
        "1.1  Purpose......7\n"
        "1.2  Requirements......8\n\n"
        "## Introduction\n\n"
        "This is the real introduction with enough characters to be prose.\n"
    )
    result = strip_toc(text)
    assert "Table of Contents" not in result
    assert "Purpose......7" not in result
    assert "## Introduction" in result
    assert "real introduction" in result
    assert "# Title" in result


def test_strip_toc_bare_line():
    text = (
        "Table of Contents\n\n"
        "1.1  Purpose......7\n"
        "1.2  Scope......9\n\n"
        "# Introduction\n\n"
        "This document describes a framework for network protocols in detail.\n"
    )
    result = strip_toc(text)
    assert "Table of Contents" not in result
    assert "Purpose......7" not in result
    assert "# Introduction" in result


def test_strip_toc_case_insensitive():
    text = (
        "TABLE OF CONTENTS\n\n"
        "1.1  Purpose......7\n\n"
        "# Introduction\n\n"
        "This is the main body of the document with real prose content.\n"
    )
    result = strip_toc(text)
    assert "TABLE OF CONTENTS" not in result
    assert "# Introduction" in result


def test_strip_toc_marker_broken_tables():
    text = (
        "# Document Title\n\n"
        "## Table of Contents\n\n"
        "| 8.1.4<br>Practical Considerations 46<br>8.2<br>Message Transmission Requirements 47 |  |\n"
        "|---|--|\n"
        "| 9.1<br>Persistent Connections 52 |  |\n\n"
        "## Overview\n\n"
        "This section provides a high-level overview of the protocol specification.\n"
    )
    result = strip_toc(text)
    assert "Table of Contents" not in result
    assert "8.1.4<br>" not in result
    assert "|---|" not in result
    assert "## Overview" in result
    assert "high-level overview" in result


def test_strip_toc_preserves_content_before_toc():
    text = (
        "# RFC 2616\n\n"
        "**Abstract:** This document defines HTTP/1.1 protocol.\n\n"
        "## Table of Contents\n\n"
        "1.1  Purpose......7\n\n"
        "## Introduction\n\n"
        "The Hypertext Transfer Protocol is an application-level protocol.\n"
    )
    result = strip_toc(text)
    assert "# RFC 2616" in result
    assert "Abstract" in result
    assert "Table of Contents" not in result
    assert "## Introduction" in result


def test_strip_toc_at_end_of_document():
    text = (
        "# Title\n\n"
        "Some content here.\n\n"
        "## Table of Contents\n\n"
        "1.1  Purpose......7\n"
        "1.2  Scope......9\n"
    )
    result = strip_toc(text)
    assert "Table of Contents" not in result
    assert "Purpose......7" not in result
    assert "# Title" in result
    assert "Some content here." in result


def test_strip_toc_does_not_match_partial():
    text = (
        "## Package Contents\n\n"
        "This package contains the following modules with detailed descriptions.\n"
    )
    result = strip_toc(text)
    assert result == text
