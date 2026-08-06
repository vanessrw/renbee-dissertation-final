"""Constrained Llama generation for the two LLM tasks in the thesis (§III.5):

  - Task 1: RFQ summary  (installer-facing prose)
  - Task 2: EPC recommendation  (homeowner-facing plain English)

The two tasks map directly to the two LLM moments in the Renbee customer
journey and each has a dedicated system prompt. `run_case()` evaluates a
case against both prompts and merges the outputs — the same prompts that
serve the live API.

Generation runs against a hosted OpenAI-compatible /chat/completions endpoint
(Llama 3.3 70B on Vertex AI). There is no local-weights path.

Public API:
  generate_rfq_summary(rfq_input)    -> {"rfq_summary": "..."}
  generate_recommendation(rfq_input) -> {"recommendation_summary": "...",
                                         "recommendation_disclaimer": "..."}
  run_case(case)                     -> {"rfq_summary": ..., "recommendation_summary": ...,
                                         "recommendation_disclaimer": ...}
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

# Model of record for the thesis. LLM_MODEL overrides it for a different target.
MODEL_NAME = "meta/llama-3.3-70b-instruct-maas"
RECOMMENDATION_DISCLAIMER = "Based only on official EPC recommendation data."

# Backoff schedule for the cloud backend: 4s, 8s, 16s, 32s, 60s, 60s ...
_RETRY_BASE_SECONDS = 4.0
_RETRY_MAX_SECONDS = 60.0


# --------------------------------------------------------------------------
# System prompts
# --------------------------------------------------------------------------

# (a) Installer-facing RFQ only
RFQ_SYSTEM_PROMPT = """\
You are an AI assistant that generates installer-ready RFQ (Request for Quote) summaries for UK residential decarbonisation enquiries.

AUDIENCE: a contractor preparing a quote. Write professionally and neutrally.

The customer's chosen technology is given by `common.technology_requested` and may be one of: heat pump, solar PV, battery storage, or solar thermal. The technology-specific details live in the section matching that technology (`heat_pump`, `solar_pv`, `battery`, or `solar_thermal`). Match the prose to the chosen technology — do not mention other technologies' fields.

STRICT RULES:
- Use ONLY the provided input data
- Do NOT add assumptions or fabricate missing information
- If an enquiry, property, or technology field is missing, you may say it is "not specified". This applies ONLY to those fields — NEVER to `site_context` (planning or grid): for those, silently omit anything not present (see the site_context rule below). Do not write "substation not specified" or "grid headroom unknown".
- Field values containing underscores (e.g. `gas_boiler_combi`, `cavity_insulated`, `pitched_insulated`) should be rendered in natural English (e.g. "a combi gas boiler", "cavity walls with insulation", "an insulated pitched roof"), not output verbatim
- Reproduce provided values faithfully — do NOT narrow, specialise, or substitute them. If `preferred_contact_method` is "either", say the homeowner is happy to be contacted by either email or phone (not just one of them). If a value is "flexible" or "unknown", convey exactly that; do not invent a specific figure or timeframe in its place.
- EPC SOURCE: if `property.epc_source` is ABSENT, the EPC figures come from THIS property's own certificate — state them directly (e.g. "has an EPC rating of D", "an EPC score of 58"). Do NOT call them "estimated" and do NOT say "similar properties" or "on the same street". That softening applies ONLY when `property.epc_source` is present: say "based on similar properties on the same street" (for `"proxy"`), "based on a nearby property the homeowner indicated is similar to theirs" (for `"proxy_nearby"` with `property.proxy_picked: true`), or "based on similar properties in nearby postcodes" (for `"proxy_nearby"` without `proxy_picked`). This is important for honesty about uncertainty.
- SITE CONTEXT: the `site_context` section lists ONLY the constraints that actually apply to this property. Anything not listed there does NOT apply. Mention each item that IS present, exactly as given:
  - `site_context.planning.listed_building: true` — mention the grade (e.g. "Grade II listed").
  - `site_context.planning.conservation_area_name` set — name the conservation area and note that Permitted Development rights are restricted.
  - `site_context.planning.article_4: true` — note that an Article 4 direction is in force and the installer should check which Permitted Development rights are removed.
  - `site_context.planning.aonb_name` / `whs_name` / `national_park_name` set — name the heritage/landscape designation and note the planning sensitivity.
  - `site_context.grid` present (UKPN only) — for heat pump enquiries, mention the demand headroom at `primary_substation` (`demand_headroom_mw`, `demand_rag`). For solar PV / battery enquiries, mention the generation headroom (`generation_headroom_mw`).
  - Mention ONLY the specific items that are present. Do NOT infer or add any related constraint that is not listed: do NOT use the words "listed" or "Grade" (e.g. "Grade II listed", "listed building consent") AT ALL unless `listed_building: true` is present; do NOT mention an Article 4 direction unless `article_4: true` is present; do NOT mention a conservation area unless a name is present. When such a name IS present you MUST name it.
  - NEVER state the ABSENCE of anything: do not write "not listed", "not in a conservation area", "no planning restrictions", or "no grid data". If an item is absent, simply say nothing about it.
  - For any consequence, HEDGE rather than assert a legal requirement as fact — write "the installer should confirm what planning permission or consent applies" rather than "planning permission will be required".
- Be concise but complete — every relevant detail should appear once
- The summary MUST be a single, fluent natural-language paragraph — NOT a list, NOT nested JSON, NOT a table

OUTPUT FORMAT — return ONLY a JSON object with exactly one string field:
{
  "rfq_summary": "<one paragraph summarising the enquiry, property, and technology-specific details>"
}

EXAMPLE OUTPUT:
{
  "rfq_summary": "The homeowner is seeking a heat pump quotation for a semi-detached, owner-occupied property of approximately 120 m², built between 1930 and 1949, with an EPC rating of D. The property is currently heated by a gas combi boiler and uses radiators throughout. Insulation levels are reported as moderate, and there is sufficient space for both a hot water cylinder and an external heat pump unit. Garden or side access is available for installation. The homeowner aims to install within the next six months, primarily to reduce energy bills and replace gas heating. Their preferred contact method is email."
}"""


# (b) Homeowner-facing EPC recommendation explanation
RECOMMENDATION_SYSTEM_PROMPT = """\
You are an AI assistant that explains EPC (Energy Performance Certificate) findings to UK homeowners in plain English.

AUDIENCE: a homeowner who has submitted an enquiry about a low-carbon home upgrade. They are non-technical.

YOUR JOB:
- Briefly acknowledge their EPC rating in context.
- Present the official EPC improvement recommendations clearly.
- Be friendly and supportive, not preachy.

STRICT RULES:
- Use ONLY the provided input data
- Every recommendation you mention MUST come from `recommendation.raw_recommendation_items` — do NOT invent suggestions
- COSTS AND SAVINGS: quote a figure ONLY when it is given in `recommendation.recommendation_details` for that item. `indicative_cost_low_gbp` and `indicative_cost_high_gbp` bound the typical installation cost, `typical_yearly_saving_gbp` is the typical yearly saving. Copy the numbers exactly, formatted as £2,700. When the low and high costs differ write the range as "£4,000 to £14,000"; when they are equal write the single figure. NEVER estimate, infer, round, or total them up, and never average a range or quote only one end of it. If an item has no figure, or `recommendation_details` is absent, simply omit that line for that step — do NOT write "unknown" or "not provided"
- If `recommendation.epc_recommendations_available` is false or the items list is empty, say so politely without fabricating advice
- If `recommendation.recommendation_source` is `"proxy_aggregate"`, these items came from neighbouring properties' EPCs (not from the homeowner's own EPC). Phrase the prose to make this clear. If `property.proxy_picked` is true, the items came from a single nearby property the homeowner picked as similar to theirs — say "Based on a nearby property similar to yours, common upgrades include...". Otherwise, if `property.epc_source` is `"proxy_nearby"` say "Based on similar properties in nearby postcodes, common upgrades include...", and if `"proxy"` say "Based on similar properties on your street, common upgrades include..." — rather than "Your EPC recommends...".
- If a `site_context` section is provided with planning constraints that ARE present and truthy, add one short sentence reflecting the most material one in plain English so the homeowner isn't surprised later — for example: "Because your home is listed (Grade II), your installer will need to apply for listed building consent before any external work" or "Your home is in the {conservation_area_name} conservation area, so any external equipment will likely need planning permission." Don't list every constraint; pick the most material one. NEVER state the absence of a constraint (do not say "your home is not listed" or "there are no planning restrictions"). Skip this sentence entirely if no planning constraint is present.
- Open with ONE short sentence about their EPC rating, then list the improvements as numbered steps in the order given
- Each step is: the improvement in plain English, then on its own line "Typical installation cost: £X" (or "£X to £Y" for a range) and "Typical yearly saving: £Z" — include only the figures that are present
- No jargon and no tables. Keep each step to one short sentence plus its figures
- Do NOT discuss the homeowner's chosen technology (heat pump / solar PV) — this output is only about the EPC recommendations

OUTPUT FORMAT — return ONLY a JSON object with exactly one string field, with the steps as newline-separated text inside that string:
{
  "recommendation_summary": "Your home is rated D, so there is room to improve.\n\n1. Add room-in-roof insulation.\nTypical installation cost: £2,700\nTypical yearly saving: £309\n\n2. ..."
}

EXAMPLE WITH RECOMMENDATIONS:
INPUT (excerpt):
  property.epc_rating = "D"
  recommendation.epc_recommendations_available = true
  recommendation.raw_recommendation_items = ["Loft insulation", "Heating controls upgrade"]
  recommendation.recommendation_details = [
    {"item": "Loft insulation", "indicative_cost_low_gbp": 100, "indicative_cost_high_gbp": 350, "typical_yearly_saving_gbp": 62},
    {"item": "Heating controls upgrade", "indicative_cost_low_gbp": 350, "indicative_cost_high_gbp": 450}
  ]

{
  "recommendation_summary": "Your home is rated D, so there is room to improve.\n\n1. Add loft insulation.\nTypical installation cost: £100 to £350\nTypical yearly saving: £62\n\n2. Upgrade your heating controls.\nTypical installation cost: £350 to £450"
}

EXAMPLE WITHOUT RECOMMENDATIONS:
{
  "recommendation_summary": "Your property does not currently have any official EPC improvement recommendations on record. We'll proceed with your enquiry based on what you've told us."
}"""


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def llm_target_label() -> str:
    """Human-readable description of the generation target, for /health."""
    return f"{os.getenv('LLM_MODEL', MODEL_NAME)} @ {os.getenv('LLM_API_BASE', '?')}"


def _cloud_auth_header() -> str:
    """Bearer token for the cloud backend.

    Static key when LLM_API_KEY is set (Together, Fireworks, Groq, OpenRouter).
    Otherwise mint a Google ADC access token, which is what Vertex's Llama
    Model-as-a-Service endpoint expects.
    """
    key = os.getenv("LLM_API_KEY")
    if key:
        return f"Bearer {key}"
    import google.auth  # lazy: only needed on the Vertex path
    import google.auth.transport.requests as _tr
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(_tr.Request())
    return f"Bearer {creds.token}"


def generate_output(messages, max_new_tokens: int = 800) -> str:
    """Generate via an OpenAI-compatible /chat/completions endpoint.

    Returns raw text for the `parse_model_output` / `_extract_or_fallback`
    pipeline. Sampling parameters are fixed here (thesis III.5.1) rather than
    read from the environment, so a run cannot silently differ from the paper.
    """
    import requests  # lazy

    base = os.getenv("LLM_API_BASE", "").rstrip("/")
    model_id = os.getenv("LLM_MODEL") or MODEL_NAME
    if not base:
        raise RuntimeError(
            "Generation requires LLM_API_BASE (the /chat/completions host). "
            "See .env.example."
        )

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_new_tokens,
        # Decoding settings of record (thesis III.5.1).
        "temperature": 0.2,
        "top_p": 0.9,
    }
    timeout = float(os.getenv("LLM_TIMEOUT", "120"))
    attempts = int(os.getenv("LLM_MAX_RETRIES", "6"))

    last_error = ""
    retry_after = None
    for attempt in range(attempts):
        if attempt:
            # Vertex's Llama Model-as-a-Service runs on shared capacity, so a
            # 429 is routine throttling rather than a real failure. A two-call
            # demo never sees one, a 180-call evaluation run sees several.
            # Honour Retry-After when the server sends it, otherwise back off
            # exponentially. Jitter stops parallel callers resynchronising.
            delay = min(_RETRY_BASE_SECONDS * 2 ** (attempt - 1), _RETRY_MAX_SECONDS)
            time.sleep(retry_after if retry_after else delay * (0.5 + random.random()))
            retry_after = None
        try:
            resp = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": _cloud_auth_header(),
                         "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as exc:
            last_error = f"request failed: {exc}"
            continue

        if resp.status_code == 200:
            return (resp.json()["choices"][0]["message"]["content"] or "").strip()

        last_error = f"returned {resp.status_code}: {resp.text[:300]}"
        # 429 is throttling and 5xx is a transient server fault, so both are
        # worth another attempt. Anything else (401, 404, malformed request)
        # will fail identically forever, so surface it immediately.
        if resp.status_code != 429 and resp.status_code < 500:
            break
        header = resp.headers.get("Retry-After")
        retry_after = float(header) if header and header.isdigit() else None

    made = attempt + 1
    raise RuntimeError(
        f"LLM endpoint {last_error} "
        f"(after {made} attempt{'s' if made > 1 else ''})"
    )


# --------------------------------------------------------------------------
# Input filtering & prompt building
# --------------------------------------------------------------------------

def _strip_nulls(obj):
    if isinstance(obj, dict):
        return {
            k: _strip_nulls(v)
            for k, v in obj.items()
            if v is not None and v != "" and v != []
        }
    if isinstance(obj, list):
        return [_strip_nulls(item) for item in obj if item is not None]
    return obj


def _prune_planning_flags(filtered_input: dict) -> dict:
    """Drop falsy planning flags so the model can't negate them: `_strip_nulls`
    keeps `False`, and a small model turns `listed_building: false` into "not
    listed". Mutates and returns the already-filtered input."""
    sc = filtered_input.get("site_context")
    if isinstance(sc, dict):
        if isinstance(sc.get("planning"), dict):
            sc["planning"] = {k: v for k, v in sc["planning"].items() if v}
            if not sc["planning"]:
                sc.pop("planning", None)
        # Nothing substantive left (no planning, no grid): drop the whole section
        # so the model isn't tempted to narrate the data_sources breadcrumb into
        # a "no planning restrictions" absence claim.
        if not sc.get("planning") and not sc.get("grid"):
            filtered_input.pop("site_context", None)
    return filtered_input


def _redact_contact_details(filtered_input: dict) -> dict:
    """Drop contact details before the model sees them."""
    from epc_to_rfq import CONTACT_DETAIL_FIELDS

    common = filtered_input.get("common")
    if isinstance(common, dict):
        for field in CONTACT_DETAIL_FIELDS:
            common.pop(field, None)
    return filtered_input


def build_rfq_prompt(rfq_input: dict):
    """Installer-facing RFQ summary only."""
    filtered_input = _redact_contact_details(
        _prune_planning_flags(_strip_nulls(rfq_input))
    )
    user_message = (
        "Generate an RFQ Summary for the following input.\n\n"
        f"INPUT DATA:\n{json.dumps(filtered_input, indent=2)}"
    )
    return [
        {"role": "system", "content": RFQ_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_recommendation_prompt(rfq_input: dict):
    """Homeowner-facing EPC recommendation explanation only.

    Pares the input down to only what's relevant for the recommendation —
    EPC rating + recommendation items — to reduce the chance the model
    pulls in heat-pump or solar-PV details.
    """
    filtered_input = _prune_planning_flags(_strip_nulls(rfq_input))
    minimal = {
        "property": {
            k: filtered_input.get("property", {}).get(k)
            for k in (
                "epc_found", "epc_rating", "epc_score",
                "construction_age_band", "epc_source", "proxy_picked",
            )
            if filtered_input.get("property", {}).get(k) is not None
        },
        "recommendation": filtered_input.get("recommendation", {}),
    }
    site_context = filtered_input.get("site_context") or {}
    if site_context.get("planning"):
        minimal["site_context"] = {"planning": site_context["planning"]}
    user_message = (
        "Produce a homeowner-facing recommendation summary based on the "
        "EPC data below. Follow the strict rules in the system prompt.\n\n"
        f"INPUT DATA:\n{json.dumps(minimal, indent=2)}"
    )
    return [
        {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

def parse_model_output(raw: str) -> dict:
    """Extract the JSON object from the model's raw text output.

    Tries every {...} substring in the response (smaller models sometimes
    emit thinking-aloud prose around the JSON, or multiple fragments).
    Returns the first that parses cleanly, or {} if none do.
    """
    if not raw:
        return {}

    # Find every balanced {..} block via a brace-depth scan, then try each.
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(raw[start : i + 1])
                start = -1

    for cand in candidates:
        try:
            # strict=False permits literal control characters inside strings.
            # The recommendation prompt asks for newline-separated steps and the
            # model emits real newlines, which strict JSON rejects.
            obj = json.loads(cand, strict=False)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    print(
        f"[generate_rfq] no parseable JSON object in model output; "
        f"raw output starts with: {raw[:200]!r}"
    )
    return {}


# --------------------------------------------------------------------------
# High-level generation functions
# --------------------------------------------------------------------------

def _extract_or_fallback(raw: str, parsed: dict, key: str) -> tuple[str, str]:
    """Pull `key` from parsed JSON, with progressive fallbacks if the model
    misbehaves. Returns (value, parse_status) where status is one of:

      ok                — JSON parsed cleanly, key present, value taken as-is
      nested_unwrapped  — model double-wrapped the value as JSON; unwrapped one layer
      regex_fallback    — JSON parse failed; regex'd the value out of raw text
      raw_fallback      — neither worked; returning raw text so UI isn't blank
    """
    if isinstance(parsed, dict):
        val = parsed.get(key)
        if val and isinstance(val, str):
            stripped = val.strip()
            # Smaller models occasionally emit nested JSON, e.g.
            # {"rfq_summary": "{\"rfq_summary\": \"...\"}"} — detect and unwrap.
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    inner = json.loads(stripped, strict=False)
                    if isinstance(inner, dict) and inner.get(key):
                        return str(inner[key]).strip(), "nested_unwrapped"
                except json.JSONDecodeError:
                    pass
            return stripped, "ok"
        if val and isinstance(val, dict) and val.get(key):
            # Triple-nested: {"rfq_summary": {"rfq_summary": "..."}}
            return str(val[key]).strip(), "nested_unwrapped"
        if parsed:
            print(
                f"[generate_rfq] parsed JSON did not contain '{key}'; "
                f"keys present: {list(parsed.keys())!r}"
            )

    # Strip markdown code fences the model sometimes wraps output in.
    cleaned = (raw or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Try key extraction anywhere in the text — don't require the whole string
    # to be a clean JSON object, as the model may add preamble/postamble.
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned)
    if m:
        # Decode JSON escapes without mangling real UTF-8 (m², £); the old
        # .encode().decode("unicode_escape") turned "m²" into "mÂ²".
        try:
            value = json.loads('"' + m.group(1) + '"', strict=False)
        except json.JSONDecodeError:
            value = m.group(1)
        return value, "regex_fallback"
    return cleaned, "raw_fallback"


def generate_rfq_summary(rfq_input: dict) -> dict:
    """Generate the installer-facing RFQ summary only.

    Returns the parsed `rfq_summary` plus `raw_response` and `parse_status`
    for debugging.
    """
    messages = build_rfq_prompt(rfq_input)
    raw = generate_output(messages)
    parsed = parse_model_output(raw)
    summary, status = _extract_or_fallback(raw, parsed, "rfq_summary")
    return {
        "rfq_summary": summary,
        "raw_response": raw,
        "parse_status": status,
    }


def generate_recommendation(rfq_input: dict) -> dict:
    """Generate the homeowner-facing EPC recommendation explanation only.

    Returns the parsed `recommendation_summary` plus `raw_response` and
    `parse_status` for debugging.
    """
    messages = build_recommendation_prompt(rfq_input)
    raw = generate_output(messages, max_new_tokens=400)
    parsed = parse_model_output(raw)
    summary, status = _extract_or_fallback(raw, parsed, "recommendation_summary")
    return {
        "recommendation_summary": summary,
        "recommendation_disclaimer": RECOMMENDATION_DISCLAIMER,
        "raw_response": raw,
        "parse_status": status,
    }


# --------------------------------------------------------------------------
# Thesis evaluation runner — calls both production prompts and merges
# --------------------------------------------------------------------------

def run_case(case):
    """Run a single evaluation case through both production prompts.

    Calls the same two functions the live API uses (`generate_rfq_summary`
    and `generate_recommendation`) so evaluation measures real production
    behaviour, not a separate combined prompt.
    """
    rfq_input = case["input"]
    rfq = generate_rfq_summary(rfq_input)
    rec = generate_recommendation(rfq_input)
    return {
        "rfq_summary": rfq.get("rfq_summary", ""),
        "recommendation_summary": rec.get("recommendation_summary", ""),
        "recommendation_disclaimer": rec.get("recommendation_disclaimer", RECOMMENDATION_DISCLAIMER),
    }


def main():
    # Anchored to this file, so the script works from any working directory.
    root = Path(__file__).resolve().parent
    with open(root / "rfq_cases_real_v1.json", "r") as f:
        cases = json.load(f)

    results = []
    for case in cases:
        print(f"\n=== Running {case['case_id']} ===")
        parsed = run_case(case)
        print(json.dumps(parsed, indent=2))
        results.append({"case_id": case["case_id"], "output": parsed})

    with open(root / "rfq_generated_outputs.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
