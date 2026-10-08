"""Verification module for quote matching and citation validation.

Implements Hard Rule #1: no claim, fact, or citation may appear unless it is
traced to an indexed chunk with a verbatim supporting quote.
"""

from typing import Optional, Tuple, Dict, Any

from ..schemas import Chunk


def _normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces and lowercase."""
    return " ".join(text.split()).lower()


def _normalized_span(quote: str, text: str) -> Optional[Tuple[int, int]]:
    """Locate ``quote`` inside ``text`` ignoring case/whitespace differences.

    Returns the approximate ``(char_start, char_end)`` offsets in the raw text.
    """
    norm_quote = _normalize(quote)
    if not norm_quote:
        return None

    # Build a normalized copy of text plus a map back to raw indices
    norm_chars = []
    index_map = []
    last_was_space = True
    for raw_idx, ch in enumerate(text):
        if ch.isspace():
            if not last_was_space and norm_chars:
                norm_chars.append(" ")
                index_map.append(raw_idx)
                last_was_space = True
        else:
            norm_chars.append(ch.lower())
            index_map.append(raw_idx)
            last_was_space = False

    norm_text = "".join(norm_chars).strip()
    # Trim leading whitespace artifacts from the map as well
    if norm_chars and norm_chars[0] == " ":
        index_map = index_map[1:]

    pos = norm_text.find(norm_quote)
    if pos < 0 or pos + len(norm_quote) > len(index_map):
        return None

    start = index_map[pos]
    end = index_map[pos + len(norm_quote) - 1] + 1
    return start, end


def find_quote_span(quote: str, text: str) -> Optional[Tuple[int, int]]:
    """Return ``(char_start, char_end)`` of ``quote`` within ``text`` if present.

    Tries an exact verbatim match first, then a case-insensitive match, then a
    whitespace/case-insensitive match.
    """
    if not quote or not text:
        return None

    idx = text.find(quote)
    if idx >= 0:
        return idx, idx + len(quote)

    idx = text.lower().find(quote.lower())
    if idx >= 0:
        return idx, idx + len(quote)

    return _normalized_span(quote, text)


def verify_quote(quote: Optional[str], chunk: Optional[Chunk]) -> Dict[str, Any]:
    """Check that ``quote`` is grounded in ``chunk`` (Hard Rule #1).

    Returns a dict with:
      - verified: True only if the quote occurs in the chunk text
      - char_start / char_end: offsets of the quote inside the chunk text
      - reason: human-readable explanation
    """
    if chunk is None:
        return {
            "verified": False,
            "char_start": None,
            "char_end": None,
            "reason": "Source chunk not found in index.",
        }
    if not quote or not quote.strip():
        return {
            "verified": False,
            "char_start": None,
            "char_end": None,
            "reason": "No quote supplied to verify.",
        }

    span = find_quote_span(quote, chunk.text)
    if span is None:
        return {
            "verified": False,
            "char_start": None,
            "char_end": None,
            "reason": "Quote not found verbatim in the cited chunk.",
        }

    start, end = span
    exact = chunk.text[start:end] == quote
    return {
        "verified": True,
        "char_start": start,
        "char_end": end,
        "reason": "Quote found verbatim in cited chunk."
        if exact
        else "Quote found in cited chunk (case/whitespace-insensitive match).",
    }
