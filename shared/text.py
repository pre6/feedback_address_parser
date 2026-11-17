"""Text helpers shared by the trainer, API and UI.

The same normalize_text() must be used when building training data and when
parsing at inference time, otherwise the model sees different text than it
was trained on.

Design notes:
- Internal runs of spaces are PRESERVED. In the IDEM data, double spaces
  separate fields ("200 TROWBRIDGE RD  INDIANAPOLIS, MARION CO") and that is
  a useful signal for the model.
- Newlines/tabs/non-breaking spaces become a double space (a field break).
- '&' and '@' become ' AND ' (same rule you used in the notebook).
"""
import hashlib
import re

_CONTROL_WS = re.compile(r"[\r\n\t\u00a0]+")
_AMPERSAT = re.compile(r"\s*[&@]\s*")


def normalize_text(text) -> str:
    if text is None:
        return ""
    t = str(text)
    t = _CONTROL_WS.sub("  ", t)
    t = _AMPERSAT.sub(" AND ", t)
    return t.strip().upper()


def text_hash(text) -> str:
    """sha256 of the normalized text. Used for dedup and leakage checks."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
