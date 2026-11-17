"""Turn (text, field value, label) into a character span, safely.

Fixes the main problems of the notebook's re.search approach:
- matches whole "words" only, so 'ON' no longer matches inside 'ONNIS'
- if the value occurs 0 times -> 'not_found'
- if it occurs more than once -> 'ambiguous' (we don't guess)
- whitespace inside the value is matched flexibly, so a single-space value
  still matches a double-space location string
"""
import re
from typing import Optional, Tuple

MISSING_VALUES = {"", "nan", "none", "null"}

Span = Tuple[int, int, str]


def is_missing(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_VALUES


def find_span(text: str, value, label: str) -> Tuple[Optional[Span], Optional[str]]:
    """Return (span, error).

    (None, None)      -> field is empty; nothing to label (not an error)
    (None, "reason")  -> value couldn't be located unambiguously
    ((s, e, label), None) -> success
    """
    if is_missing(value):
        return None, None

    parts = str(value).split()
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(p) for p in parts) + r"(?!\w)"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return None, "not_found"
    if len(matches) > 1:
        return None, "ambiguous"

    m = matches[0]
    return (m.start(), m.end(), label), None


def has_overlap(spans) -> bool:
    ordered = sorted(spans, key=lambda s: (s[0], s[1]))
    return any(ordered[i][0] < ordered[i - 1][1] for i in range(1, len(ordered)))
