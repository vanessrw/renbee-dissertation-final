"""Rubric definitions and judge prompts for the LLM-as-judge evaluation.

Two quality rubrics map directly to the research questions:

  RQ2 (installer-facing RFQ summary):    clarity, usability, perceived_efficiency
  RQ3 (homeowner-facing recommendation): clarity, understandability, perceived_helpfulness

Each criterion is scored 1-5 by Gemini, which adopts an audience-specific
persona. A separate prompt drives the fabrication check used in the faithfulness
component (RQ1). All prompts ask for strict JSON so the client can parse them.
"""
from __future__ import annotations

import json
from typing import Any

# Bump when any wording below changes. Written into scores.json metadata so a
# run scored under one rubric can never be silently compared with another.
RUBRIC_VERSION = "2.0-anchored"


# --------------------------------------------------------------------------
# Rubric criteria (1-5 Likert, fully anchored at every level)
#
# v1 defined only the endpoints ("1 = very poor, 5 = excellent"), leaving the
# judge to invent 2, 3 and 4 on every call. The three widest-spread metrics in
# the first scored run were all on this scale. v2 describes all five levels.
#
# Two deliberate choices in the anchor wording:
#   - They are BEHAVIOURAL (what the reader could do), not fact counts. Counting
#     facts would re-measure information preservation, which faithfulness.py
#     already measures deterministically, and make the two collinear.
#   - No reference examples. The cases span four technologies, so a worked
#     example in one of them would anchor the judge to that technology.
# --------------------------------------------------------------------------

RFQ_CRITERIA = {
    "clarity": "How clearly and logically the RFQ is written and structured for a reader.",
    "usability": "Whether the RFQ gives an installer sufficient, well-organised information to prepare a quote.",
    "perceived_efficiency": "Whether the RFQ is likely to reduce the need for follow-up clarification.",
}

RECOMMENDATION_CRITERIA = {
    "clarity": "How readable and well organised the summary is.",
    "understandability": "How easily a non-technical homeowner can understand the content.",
    "perceived_helpfulness": "Whether the summary helps the homeowner understand and act on the recommendations.",
}

# criterion -> {level: description}, plus an optional "note" for guidance that
# applies to the criterion as a whole rather than to one level.
RFQ_ANCHORS: dict[str, dict] = {
    "clarity": {
        1: "Hard to follow. Broken grammar, contradictory statements, or no discernible order.",
        2: "Readable in places but disjointed. Abrupt jumps, repetition, or sentences that must be re-read.",
        3: "Understandable throughout, but the ordering is arbitrary or some phrasing is clumsy.",
        4: "Clear and logically ordered. Reads professionally, with at most minor wording niggles.",
        5: "Immediately clear. Information is grouped sensibly and phrased precisely. Nothing needs re-reading.",
    },
    "usability": {
        1: "Could not begin. The core request, meaning the technology and the property, is not identifiable.",
        2: "The request is identifiable but little else is usable. A full fact-finding call would be needed before any quoting work.",
        3: "Preliminary work could begin, but several material details would have to be chased first.",
        4: "A substantive quote could be prepared, needing confirmation of only one or two points.",
        5: "A quote could be prepared directly from this. What an installer needs is present and easy to locate.",
        "note": "Judge by what you could actually do with the summary, not by how many facts it lists.",
    },
    "perceived_efficiency": {
        1: "Almost everything would need clarifying. The RFQ raises more questions than it answers.",
        2: "Several substantial follow-ups would be needed before work could proceed.",
        3: "A handful of follow-ups likely, on material points.",
        4: "Perhaps one clarifying question, on a minor point.",
        5: "No follow-up needed. Ambiguities are either resolved or explicitly flagged as unknown.",
        "note": "Explicitly stating that something was not provided counts IN FAVOUR of this "
                "score, not against it, because it saves the installer a wasted call.",
    },
}

RECOMMENDATION_ANCHORS: dict[str, dict] = {
    "clarity": {
        1: "Hard to read. Garbled, contradictory, or badly broken sentences.",
        2: "Parts are readable, but it jumps around or repeats itself.",
        3: "Understandable, though some sentences are clumsy or the order is arbitrary.",
        4: "Clear and well ordered. Easy to read straight through.",
        5: "Immediately clear and naturally phrased. Nothing needs re-reading.",
    },
    "understandability": {
        1: "Unintelligible without expertise. Dense jargon and unexplained acronyms.",
        2: "Mostly opaque. Some plain sentences, but the key parts need technical knowledge.",
        3: "Broadly graspable, but several terms go unexplained and the reader would have to look things up.",
        4: "Understandable throughout. Technical terms are either explained or avoided.",
        5: "Effortless for a layperson. Any necessary technical term is explained in plain words where it first appears.",
        "note": "Terms such as EPC, SAP, kWh, U-value and 'fabric first' are the kind that "
                "need explaining where they first appear.",
    },
    "perceived_helpfulness": {
        1: "Gives the reader nothing to understand or act on.",
        2: "States findings but offers no sense of what to do about them.",
        3: "Names improvements but not why they matter or what to do next.",
        4: "Explains what to improve and why, with some sense of next steps.",
        5: "The reader finishes knowing their situation, what would help, why, and what to do next.",
        "note": "Where the summary says figures are estimated or based on similar nearby "
                "properties, treat that honesty as appropriate rather than unhelpful.",
    },
}

# Replaces the old LIKERT_ANCHOR. The conservative-scoring and independence
# rules are anti-inflation devices; the length rule exists because the RFQ
# prompt truncates at 800 tokens, so verbosity is a failure mode, not a virtue.
SCORING_GUIDANCE = (
    "Scoring rules:\n"
    "- Score each criterion independently. Do not let a favourable impression on "
    "one criterion raise the others.\n"
    "- When the evidence does not clearly justify the higher of two scores, give "
    "the lower one.\n"
    "- Length is not quality. A short summary can score 5 and a long one can "
    "score 2.\n"
    "- Do not penalise wording or structure that differs from how you would have "
    "written it, provided the meaning holds.\n"
    "- Judge only the text in front of you. You are not shown the underlying "
    "property record and must not speculate about what it might contain.\n"
    "- Give a one-sentence reason for each criterion, citing something specific "
    "in the text."
)


# --------------------------------------------------------------------------
# Quality-judge prompts (RQ2 / RQ3)
# --------------------------------------------------------------------------

def _criteria_block(criteria: dict[str, str], anchors: dict[str, dict]) -> str:
    """Render each criterion with its definition and all five level anchors."""
    blocks = []
    for name, definition in criteria.items():
        levels = anchors.get(name, {})
        lines = [f'"{name}": {definition}']
        for score in (1, 2, 3, 4, 5):
            if score in levels:
                lines.append(f"    {score} = {levels[score]}")
        note = levels.get("note")
        if note:
            lines.append(f"    Note: {note}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _quality_schema_hint(criteria: dict[str, str]) -> str:
    fields = ", ".join(f'"{k}": <1-5>' for k in criteria)
    reasons = ", ".join(f'"{k}": "<one sentence>"' for k in criteria)
    return (
        "Return ONLY a JSON object of the form:\n"
        f"{{ {fields}, \"reasons\": {{ {reasons} }} }}"
    )


RFQ_JUDGE_SYSTEM = (
    "You are an experienced UK heating and renewables installer reviewing an "
    "incoming Request for Quote (RFQ) summary to decide whether you could "
    "prepare a quote from it. Judge the summary only as a reader: assess how it "
    "is written and how useful it is to you, not whether the underlying job is "
    "viable.\n\n"
    "Rate the RFQ summary on these criteria, using the scale described under "
    "each one:\n\n"
    f"{_criteria_block(RFQ_CRITERIA, RFQ_ANCHORS)}\n\n"
    f"{SCORING_GUIDANCE}\n\n"
    f"{_quality_schema_hint(RFQ_CRITERIA)}"
)

RECOMMENDATION_JUDGE_SYSTEM = (
    "You are a non-technical UK homeowner with no background in energy or "
    "construction, reading a short summary of your home's energy-improvement "
    "recommendations. Judge the summary only as a lay reader.\n\n"
    "Rate the recommendation summary on these criteria, using the scale "
    "described under each one:\n\n"
    f"{_criteria_block(RECOMMENDATION_CRITERIA, RECOMMENDATION_ANCHORS)}\n\n"
    f"{SCORING_GUIDANCE}\n\n"
    f"{_quality_schema_hint(RECOMMENDATION_CRITERIA)}"
)


def rfq_judge_user(rfq_summary: str, technology: str) -> str:
    return (
        f"Technology requested: {technology}\n\n"
        f"RFQ summary to evaluate:\n\"\"\"\n{rfq_summary}\n\"\"\""
    )


def recommendation_judge_user(recommendation_summary: str) -> str:
    return (
        f"Recommendation summary to evaluate:\n\"\"\"\n{recommendation_summary}\n\"\"\""
    )


# --------------------------------------------------------------------------
# Fabrication-judge prompt (part of RQ1)
# --------------------------------------------------------------------------

FABRICATION_JUDGE_SYSTEM = (
    "You are a meticulous fact-checker. You are given a structured INPUT record "
    "and a generated SUMMARY produced from it. Your job is to find statements in "
    "the SUMMARY that are NOT supported by the INPUT, that is, fabricated or "
    "unsupported content. A statement that says information is 'not specified' or "
    "'not provided' is NOT a fabrication. Restating, rephrasing, or reasonably "
    "describing values that ARE in the input is NOT a fabrication. Only count "
    "concrete claims that introduce facts absent from the input.\n\n"
    "Return ONLY a JSON object of the form:\n"
    "{ \"total_statements\": <int>, \"fabricated_statements\": <int>, "
    "\"fabricated_list\": [\"<verbatim unsupported claim>\", ...] }\n"
    "total_statements is your count of distinct factual claims in the SUMMARY. "
    "fabricated_statements is how many of those are unsupported."
)


def fabrication_judge_user(structured_input: dict[str, Any], output_text: str) -> str:
    return (
        "INPUT (the only facts that are true):\n"
        f"{json.dumps(structured_input, indent=2, ensure_ascii=False)}\n\n"
        f"SUMMARY to check:\n\"\"\"\n{output_text}\n\"\"\""
    )


# --------------------------------------------------------------------------
# Deterministic mock responses (used in --mock-judge / GEMINI_MOCK mode)
# --------------------------------------------------------------------------

def mock_quality_response(criteria: dict[str, str]) -> dict:
    """Plausible mid/high canned scores so the loop and tables can be tested."""
    out: dict = {k: 4 for k in criteria}
    out["reasons"] = {
        k: "[mock judge] canned reason for offline testing." for k in criteria
    }
    return out


def mock_fabrication_response() -> dict:
    return {"total_statements": 8, "fabricated_statements": 0, "fabricated_list": []}
