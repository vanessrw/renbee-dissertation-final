"""Faithfulness scoring for RQ1: information preservation + fabrication.

Information preservation is deterministic and objective: for each relevant field
in the structured input, we check whether its information is represented in the
generated text. This is a heuristic (token-overlap with normalisation), but it
removes judge bias from the preservation number and acts as an objective anchor
that the LLM-judge quality scores can be reviewed against (see review.py).

Fabrication is Gemini-judged: the judge sees the structured input and the
output and counts statements not supported by the input.

The two together answer "did the model keep what it was given, and did it invent
anything it was not given".
"""
from __future__ import annotations

import re
from typing import Any, Optional

from epc_to_rfq import CONTACT_DETAIL_FIELDS, FIELDS, _TECH_SECTIONS
from evaluation import rubric
from evaluation.gemini_client import GeminiClient

# Meta / identifier fields that carry no prose-level information to preserve.
_EXCLUDE_FIELDS = {
    "enquiry_id", "submission_date", "epc_found", "epc_source", "epc_score",
    "proxy_comparator_count", "proxy_confidence", "proxy_postcodes", "proxy_picked",
}

_STOPWORDS = {
    "the", "a", "an", "of", "or", "and", "to", "for", "is", "are", "with",
    "type", "status", "level", "known", "available", "preference", "interest",
    "basic", "m2", "gbp", "years", "band", "system",
}

_YES_NO = {"yes", "no", "unknown", "true", "false", "none", "n/a"}

# Optional in FIELDS, but still scored when supplied: `required` gates both the
# completeness denominator and the filter below, so these would otherwise vanish.
ALWAYS_CHECKED_IF_PRESENT: dict[str, tuple[str, ...]] = {
    "property": ("floor_area_m2",),
    "solar_pv": ("usable_roof_area_m2",),
    "solar_thermal": ("usable_roof_area_m2",),
}


def _normalise(text: str) -> str:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", _normalise(s)) if t]


def _content_tokens(s: str) -> list[str]:
    return [t for t in _tokens(s) if t not in _STOPWORDS and len(t) > 2]


def _standalone_present(token: str, norm_text: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", norm_text) is not None


def _number_strings(value: Any) -> list[str]:
    out = []
    try:
        f = float(value)
        if f.is_integer():
            out.append(str(int(f)))
        out.append(str(value).strip())
    except (TypeError, ValueError):
        pass
    return list(dict.fromkeys(out))


def _field_preserved(field_name: str, value: Any, norm_text: str,
                     threshold: float = 0.6) -> tuple[bool, list[str]]:
    """Return (preserved, salient_tokens) for one field against the output text."""
    # Numeric value: require the number to appear.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        nums = _number_strings(value)
        # Word-boundary match, else e.g. floor_area 90 is falsely "preserved" by
        # the "90" inside "1990".
        return (any(n and _standalone_present(n, norm_text) for n in nums), nums)

    sval = str(value).strip()
    nval = _normalise(sval)

    # Yes/no/boolean-style: the value carries no lexical content, so check that
    # the field's concept is mentioned (tokens derived from the field name).
    if isinstance(value, bool) or nval in _YES_NO:
        concept = _content_tokens(field_name)
        if not concept:
            return (True, [])  # nothing meaningful to check
        hits = sum(1 for t in concept if _standalone_present(t, norm_text))
        return (hits / len(concept) >= threshold, concept)

    # Short code (e.g. EPC rating "D"): standalone match of the code.
    if len(nval) <= 2:
        return (_standalone_present(nval, norm_text), [nval])

    # General string: match the value's content tokens, falling back to the
    # field-name concept if the value is all stopwords.
    salient = _content_tokens(sval) or _content_tokens(field_name)
    if not salient:
        return (True, [])
    hits = sum(1 for t in salient if _standalone_present(t, norm_text))
    return (hits / len(salient) >= threshold, salient)


def _relevant_rfq_fields(rfq_input: dict) -> list[tuple[str, str, Any]]:
    """(section, field, value) for fields the RFQ summary should preserve.

    Only required fields (marked required: True in FIELDS) are checked, plus
    epc_rating which is always essential for an installer quote, plus anything in
    ALWAYS_CHECKED_IF_PRESENT that the homeowner actually supplied. Other optional
    EPC fields (walls_description, hot_water_system, postcode, etc.) are excluded
    because a good summary is selective — the metric measures whether the model
    conveyed what an installer needs, not whether it transcribed every attribute.
    """
    out = []
    tech = (rfq_input.get("common") or {}).get("technology_requested")
    sections = ["common", "property"]
    if tech in _TECH_SECTIONS:
        sections.append(_TECH_SECTIONS[tech])

    for section in sections:
        data = rfq_input.get(section) or {}
        section_fields = FIELDS.get(section, {})

        # epc_rating is not in FIELDS but is always essential when present.
        if section == "property":
            rating = data.get("epc_rating")
            if rating not in (None, "", []):
                out.append((section, "epc_rating", rating))

        for field, meta in section_fields.items():
            if not (isinstance(meta, dict) and meta.get("required")):
                continue
            if field in CONTACT_DETAIL_FIELDS:
                continue
            value = data.get(field)
            if value is None or value == "" or value == []:
                continue
            out.append((section, field, value))

        # Skip any already added, so re-promoting one to required can't double-count.
        already = {f for s, f, _ in out if s == section}
        for field in ALWAYS_CHECKED_IF_PRESENT.get(section, ()):
            if field in already:
                continue
            value = data.get(field)
            if value in (None, "", []):
                continue
            out.append((section, field, value))

    return out


def preservation_score(rfq_input: dict, output_text: str, output_type: str) -> dict:
    """Deterministic information-preservation score for one output.

    output_type is "rfq" or "recommendation".
    """
    norm_text = _normalise(output_text)
    details = []

    if output_type == "recommendation":
        # Relevant information: the EPC rating and each recommendation item.
        rating = (rfq_input.get("property") or {}).get("epc_rating")
        if rating:
            ok, salient = _field_preserved("epc_rating", rating, norm_text)
            details.append({"field": "property.epc_rating", "value": rating,
                            "preserved": ok, "salient": salient})
        items = (rfq_input.get("recommendation") or {}).get("raw_recommendation_items") or []
        for i, item in enumerate(items):
            ok, salient = _field_preserved(f"recommendation_item_{i}", item, norm_text, threshold=0.5)
            details.append({"field": f"recommendation.item[{i}]", "value": item,
                            "preserved": ok, "salient": salient})
    else:  # rfq
        for section, field, value in _relevant_rfq_fields(rfq_input):
            ok, salient = _field_preserved(field, value, norm_text)
            details.append({"field": f"{section}.{field}", "value": value,
                            "preserved": ok, "salient": salient})

    total = len(details)
    preserved = sum(1 for d in details if d["preserved"])
    return {
        # None (not 1.0) when there is nothing to preserve — e.g. a Case B
        # recommendation with no EPC rating and no items. A vacuous 1.0 would
        # inflate the aggregate; None is dropped from it (as site_context does).
        "preservation_rate": (preserved / total) if total else None,
        "preserved": preserved,
        "total": total,
        "missed": [d["field"] for d in details if not d["preserved"]],
        "details": details,
    }


def site_context_coverage(rfq_input: dict, output_text: str) -> Optional[dict]:
    """Secondary metric: fraction of present site-context items woven into the
    RFQ. Returns None when there is no truthy site context to report on."""
    sc = rfq_input.get("site_context") or {}
    planning = sc.get("planning") or {}
    grid = sc.get("grid") or {}
    norm_text = _normalise(output_text)
    items: list[tuple[str, list[str]]] = []

    if planning.get("listed_building"):
        items.append(("listed_building", ["listed"]))
    if planning.get("conservation_area_name"):
        items.append(("conservation_area", _content_tokens(planning["conservation_area_name"]) or ["conservation"]))
    if planning.get("article_4"):
        items.append(("article_4", ["article"]))
    for key in ("aonb_name", "whs_name", "national_park_name"):
        if planning.get(key):
            items.append((key, _content_tokens(planning[key])))
    if grid:
        if grid.get("primary_substation"):
            items.append(("primary_substation", _content_tokens(grid["primary_substation"])))
        for key in ("demand_headroom_mw", "generation_headroom_mw"):
            if grid.get(key) is not None:
                items.append((key, _number_strings(grid[key])))

    if not items:
        return None
    covered = 0
    detail = []
    for name, salient in items:
        if not salient:
            hit = True
        else:
            hit = sum(1 for t in salient if t and t in norm_text) / len(salient) >= 0.5
        covered += 1 if hit else 0
        detail.append({"item": name, "covered": hit, "salient": salient})
    return {"coverage_rate": covered / len(items), "covered": covered,
            "total": len(items), "details": detail}


def fabrication_score(client: GeminiClient, structured_input: dict,
                      output_text: str) -> dict:
    """Gemini-judged fabrication for one output."""
    user = rubric.fabrication_judge_user(structured_input, output_text)
    res = client.judge_json(
        rubric.FABRICATION_JUDGE_SYSTEM, user,
        mock_response=rubric.mock_fabrication_response,
    )
    def _as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0  # a malformed judge field must not crash the whole run

    total = _as_int(res.get("total_statements"))
    # Clamp: the judge produces both counts independently, so guard against a
    # fabricated > total slip that would yield a rate > 1.0.
    fab = min(max(_as_int(res.get("fabricated_statements")), 0), total)
    return {
        "total_statements": total,
        "fabricated_statements": fab,
        "fabrication_rate": (fab / total) if total else 0.0,
        "fabricated_list": res.get("fabricated_list") or [],
    }
