"""Confidence scoring and flagging rules for parsed addresses.

spaCy's NER has no built-in confidence score, so we use beam search: it keeps
several candidate parses with probabilities, and an entity's confidence is the
total probability of the candidate parses that contain it. These are useful
for ranking and thresholding but are not perfectly calibrated probabilities,
so pick your threshold by looking at real flagged/unflagged examples.
"""
import re
from collections import defaultdict

FIELD_FOR_LABEL = {"ADDRESS": "address", "CITY": "city", "ZIP": "zip"}


def beam_scores(nlp, docs, beam_width=16, beam_density=0.0001):
    """For each already-processed doc: {(start_token, end_token, label): probability}."""
    ner = nlp.get_pipe("ner")
    beams = ner.beam_parse(docs, beam_width=beam_width, beam_density=beam_density)
    out = []
    for beam in beams:
        scores = defaultdict(float)
        for score, ents in ner.moves.get_beam_parses(beam):
            for start, end, label in ents:
                scores[(start, end, label)] += score
        out.append(scores)
    return out


def analyze(doc, scores, threshold, required, zip_regex=None):
    """Turn a parsed doc into fields + confidence + flag reasons.

    scores: output of beam_scores for this doc, or None for rules-only mode.
    Returns dict with address/city/zip, min_conf, entity_conf, flags, status.
    """
    flags = []
    by_label = defaultdict(list)
    for e in doc.ents:
        conf = None if scores is None else round(min(scores.get((e.start, e.end, e.label_), 0.0), 1.0), 4)
        by_label[e.label_].append((conf, e.text))

    if not doc.ents:
        flags.append("no_entities")

    fields, conf_by_label = {}, {}
    for label, field in FIELD_FOR_LABEL.items():
        cands = by_label.get(label, [])
        if not cands:
            fields[field] = None
            if label in required:
                flags.append(f"missing:{label}")
            continue
        if len(cands) > 1:
            flags.append(f"multiple:{label}")
        best = max(cands, key=lambda c: -1.0 if c[0] is None else c[0])
        fields[field] = best[1]
        conf_by_label[label] = best[0]
        if best[0] is not None and best[0] < threshold:
            flags.append(f"low_conf:{label}")

    if zip_regex and fields.get("zip") and not re.fullmatch(zip_regex, fields["zip"]):
        flags.append("bad_zip_format")

    confs = [c for c in conf_by_label.values() if c is not None]
    if any(f.startswith(("missing:", "no_entities")) for f in flags):
        min_conf = 0.0  # worst case, floats to the top of the queue
    else:
        min_conf = min(confs) if confs else None

    return {
        **fields,
        "min_conf": min_conf,
        "entity_conf": conf_by_label,
        "flags": flags,
        "status": "needs_review" if flags else "auto",
    }
