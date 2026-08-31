"""FastAPI wrapper for the RFQ pipeline.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from epc_fetch import fetch_epc_data
from epc_to_rfq import (
    AmbiguousAddress,
    assemble_rfq_input,
    build_nearby_candidate_list,
    build_picked_property_section,
    build_site_context,
    completeness_score,
    map_recommendation_section,
    missing_fields,
    optional_fields,
)


# --------------------------------------------------------------------------
# Demo mode: when DEMO_MOCK_LLM=1 in the environment, swap the hosted LLM
# generation functions for canned responses. Lets the Webflow demo run
# with no network access and no credentials.
# --------------------------------------------------------------------------

DEMO_MOCK_LLM = os.getenv("DEMO_MOCK_LLM") in {"1", "true", "yes"}

if DEMO_MOCK_LLM:
    import generate_rfq as _gen_module

    def _mock_recommendation(rfq_input: dict) -> dict:
        rec = rfq_input.get("recommendation") or {}
        prop = rfq_input.get("property") or {}
        items = rec.get("raw_recommendation_items") or []
        rating = prop.get("epc_rating") or "?"
        src = prop.get("epc_source")
        picked = prop.get("proxy_picked")
        # Match the real prompt's proxy softening so the demo doesn't assert a
        # proxy estimate as the homeowner's own EPC.
        is_proxy = rec.get("recommendation_source") == "proxy_aggregate" or src in ("proxy", "proxy_nearby")
        if not items:
            text = (
                "Your property does not currently have any official EPC improvement "
                "recommendations on record. We'll proceed with your enquiry based on "
                "what you've told us."
            )
        elif is_proxy:
            top = ", ".join(items[:3])
            if picked:
                lead = "Based on a nearby property similar to yours"
            elif src == "proxy_nearby":
                lead = "Based on similar properties in nearby postcodes"
            else:
                lead = "Based on similar properties on your street"
            text = (
                f"{lead}, common upgrades that could help include: {top}. These are "
                f"estimates for your area rather than your home's own certificate, but "
                f"they're a useful starting point."
            )
        else:
            top = ", ".join(items[:3])
            text = (
                f"Your home currently has an EPC rating of {rating}, which means "
                f"there's room to improve its energy performance. The official EPC "
                f"report suggests these upgrades: {top}. Following them could "
                f"improve your rating and reduce running costs over time."
            )
        from generate_rfq import recommendation_disclaimer

        text += _mock_cost_block(rfq_input)
        return {
            "recommendation_summary": text,
            "recommendation_disclaimer": recommendation_disclaimer(rfq_input),
            "raw_response": (
                "[mock LLM — DEMO_MOCK_LLM=1]\n"
                f'{{"recommendation_summary": {json.dumps(text)}}}'
            ),
            "parse_status": "mock",
        }

    def _gbp(low, high) -> str:
        """Same rendering contract the real prompt is given."""
        return f"£{low:,}" if low == high else f"£{low:,} to £{high:,}"

    def _mock_cost_block(rfq_input: dict) -> str:
        """Mirror the real prompt's cost block so a mocked demo isn't missing
        the feature entirely."""
        est = rfq_input.get("cost_estimate")
        if not est:
            return ""
        tech = "solar panels" if est["technology"] == "solar_pv" else "heat pump"
        lines = [f"\n\nIndicative cost for your {tech}"]
        if est["match_type"] == "nearest_band":
            lines.append(
                "Your home doesn't fall exactly into one of the standard bands, so "
                f"these figures are for the closest comparable band, a {est['matched_band']}."
            )
        else:
            lines.append(f"These figures are for a {est['matched_band']} home like yours.")

        if est["technology"] == "solar_pv":
            panels = (f"{est['panels_low']}" if est["panels_low"] == est["panels_high"]
                      else f"{est['panels_low']}-{est['panels_high']}")
            lines += [
                f"System size: {est['system_size_kw']} kW ({panels} panels)",
                "Typical installed cost: " + _gbp(est["installed_cost_low_gbp"],
                                                  est["installed_cost_high_gbp"]),
                "Typical yearly saving: " + _gbp(est["annual_saving_low_gbp"],
                                                 est["annual_saving_high_gbp"]),
            ]
        else:
            lines += [
                "Air source heat pump: " + _gbp(est["air_source_cost_low_gbp"],
                                                est["air_source_cost_high_gbp"]) + " installed",
                "Ground source heat pump: " + _gbp(est["ground_source_cost_low_gbp"],
                                                   est["ground_source_cost_high_gbp"]) + " installed",
                f"If you qualify for the {est['grant_name']}, an air source system would "
                "come down to " + _gbp(est["net_cost_after_grant_low_gbp"],
                                       est["net_cost_after_grant_high_gbp"]) + ".",
            ]
        lines.append(est["guide_price_note"])
        return "\n".join(lines)

    def _mock_rfq(rfq_input: dict) -> dict:
        common = rfq_input.get("common") or {}
        prop = rfq_input.get("property") or {}
        tech_raw = common.get("technology_requested", "low-carbon technology")
        tech = tech_raw.replace("_", " ")
        ptype = prop.get("property_type", "property of unknown type")
        bform = prop.get("built_form")
        epc = prop.get("epc_rating") or "not specified"
        area = prop.get("floor_area_m2") or "not specified"
        age = prop.get("construction_age_band") or "not specified"
        heating = prop.get("current_heating_system") or "not specified"
        timeline = common.get("desired_installation_timeline", "not specified").replace("_", " ")
        contact = common.get("preferred_contact_method", "not specified")
        motivation = common.get("motivation", "not specified")
        descriptor = f"{bform}, " if bform else ""
        src = prop.get("epc_source")
        if src in ("proxy", "proxy_nearby"):
            if prop.get("proxy_picked"):
                epc_note = f"an estimated EPC rating of {epc} (based on a nearby property the homeowner indicated is similar)"
            elif src == "proxy_nearby":
                epc_note = f"an estimated EPC rating of {epc} (based on similar properties in nearby postcodes)"
            else:
                epc_note = f"an estimated EPC rating of {epc} (based on similar properties on the same street)"
        else:
            epc_note = f"an EPC rating of {epc}"
        text = (
            f"The homeowner is requesting a {tech} quotation for a {descriptor}{ptype} "
            f"property with {epc_note} and an approximate floor area of "
            f"{area} m². The property falls within the {age} construction age band "
            f"and is currently heated by: {heating}. Installation is desired {timeline}. "
            f"The homeowner's main motivation is {motivation}. Preferred contact is by {contact}. "
            f"(Demo mock — replace with real model output for evaluation runs.)"
        )
        return {
            "rfq_summary": text,
            "raw_response": (
                "[mock LLM — DEMO_MOCK_LLM=1]\n"
                f'{{"rfq_summary": {json.dumps(text)}}}'
            ),
            "parse_status": "mock",
        }

    _gen_module.generate_recommendation = _mock_recommendation
    _gen_module.generate_rfq_summary = _mock_rfq


app = FastAPI(
    title="Renbee RFQ API",
    description=(
        "Constrained LLM-based RFQ generation for residential decarbonisation "
        "enquiries. Companion to Vanessa Wiyono's MSc thesis."
    ),
    version="0.1.0",
)

# Permissive CORS for local development; tighten allow_origins to Renbee's
# domain (e.g. ["https://renbee.uk"]) before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# In-memory session store
# --------------------------------------------------------------------------

_SESSIONS: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------

class InitiateRequest(BaseModel):
    postcode: str = Field(..., description="UK postcode, e.g. 'E1 6AN'")
    house_number: Optional[str] = Field(
        None,
        description=(
            "House or flat number to filter the postcode lookup. Matched as a "
            "whole token against the start of the address (so '1' matches "
            "'1 Old Penns Yard' but not '10 ...' or '1A ...')."
        ),
    )
    lmk_key: Optional[str] = Field(
        None,
        description=(
            "EPC lmk-key chosen from a previous /api/initiate 409 response's "
            "candidate list. Use this when a postcode resolved to multiple "
            "distinct addresses and the user has selected one."
        ),
    )
    technology: str = Field(
        ..., description="What the homeowner wants to install: 'heat_pump', 'solar_pv', 'battery', or 'solar_thermal'"
    )
    use_proxy: bool = Field(
        False,
        description=(
            "When true, skip address selection and aggregate every EPC in the "
            "postcode into a proxy property section (used when the homeowner "
            "can't find their property in the candidate list)."
        ),
    )
    use_proxy_nearby_average: bool = Field(
        False,
        description=(
            "Layer 2 picker: 'use average of all 5' choice. Re-runs the "
            "nearby-postcode EPC sweep and aggregates the results."
        ),
    )
    proxy_postcode: Optional[str] = Field(
        None,
        description=(
            "Layer 2 picker: paired with lmk_key when the chosen EPC comes "
            "from a nearby postcode (not the homeowner's input postcode). "
            "Identifies which postcode to re-fetch."
        ),
    )



class FieldSpec(BaseModel):
    name: str
    label: str
    type: str
    options: Optional[list[str]] = None


class InitiateResponse(BaseModel):
    session_id: str
    epc_found: bool
    auto_filled: dict
    missing_fields: dict[str, list[FieldSpec]]
    # Offered but never gating. Blanks here do not block /api/generate.
    optional_fields: dict[str, list[FieldSpec]] = {}
    completeness: dict
    site_context: dict


class CandidateAddress(BaseModel):
    address: str
    lmk_key: str
    inspection_date: str


class AmbiguousAddressResponse(BaseModel):
    error: str = "ambiguous_address"
    candidates: list[CandidateAddress]


class ProxyNearbyCandidate(BaseModel):
    lmk_key: str
    postcode: str
    address: str
    inspection_date: str
    property_type: Optional[str] = None
    floor_area_m2: Optional[float] = None
    current_heating_system: Optional[str] = None
    current_fuel_type: Optional[str] = None


class ProxyNearbyCandidatesResponse(BaseModel):
    error: str = "proxy_nearby_candidates"
    candidates: list[ProxyNearbyCandidate]


class GenerateRequest(BaseModel):
    session_id: str
    additional_fields: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Section -> {field_name: value} of the answers from Step 2 of the form",
    )


class GenerateResponse(BaseModel):
    session_id: str
    recommendation_summary: str
    recommendation_disclaimer: str
    rfq_input: dict
    completeness: dict
    raw_response: Optional[str] = None
    parse_status: Optional[str] = None


class GenerateRFQRequest(BaseModel):
    session_id: str
    vendor_id: Optional[str] = Field(
        None, description="Optional installer/vendor identifier for tracking"
    )


class GenerateRFQResponse(BaseModel):
    session_id: str
    rfq_summary: str
    rfq_input: dict
    ready_to_submit: bool
    raw_response: Optional[str] = None
    parse_status: Optional[str] = None


class SaveRFQRequest(BaseModel):
    session_id: Optional[str] = Field(
        None, description="Session this RFQ came from (optional, for traceability)"
    )
    rfq_summary: str = Field(..., description="Final (reviewer-edited) installer-facing RFQ prose")
    rfq_input: dict = Field(..., description="Final (reviewer-edited) structured RFQ input")


class SaveRFQResponse(BaseModel):
    saved: bool
    filename: str
    path: str


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

# System-side sections. The client never supplies these, so a payload naming
# one is either confused or hostile.
_DERIVED_SECTIONS = frozenset({"site_context", "cost_estimate"})


def _merge_additional_fields(rfq_input: dict, additional: dict) -> dict:
    """Merge Step 2 answers into the assembled dict (in place + returned)."""
    for section, fields in (additional or {}).items():
        if section in _DERIVED_SECTIONS:
            continue
        if section not in rfq_input or not isinstance(rfq_input[section], dict):
            rfq_input[section] = {}
        for k, v in (fields or {}).items():
            rfq_input[section][k] = v
    return rfq_input


def _attach_cost_estimate(rfq_input: dict) -> dict:
    """Derive the indicative cost band, or clear a stale one.

    Assigned unconditionally so a second /api/generate with a different bedroom
    count cannot leave the previous band behind.
    """
    from cost_tables import build_cost_estimate

    estimate = build_cost_estimate(rfq_input)
    if estimate:
        rfq_input["cost_estimate"] = estimate
    else:
        rfq_input.pop("cost_estimate", None)
    return rfq_input


def _require_session(session_id: str) -> dict:
    if session_id not in _SESSIONS:
        raise HTTPException(
            status_code=404,
            detail={"error": "unknown_session", "session_id": session_id},
        )
    return _SESSIONS[session_id]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    from generate_rfq import llm_target_label
    return {
        "status": "ok",
        "demo_mock_llm": DEMO_MOCK_LLM,
        "llm_target": llm_target_label(),
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Browsers auto-request /favicon.ico; we don't ship one. Return 204 so it
    doesn't log a 404 on every page load."""
    return Response(status_code=204)


@app.get("/demo", include_in_schema=False)
def demo_page():
    """Serves the standalone Webflow-style demo page (single HTML file)."""
    path = Path(__file__).parent / "webflow_demo.html"
    return FileResponse(path)


_ROOT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Renbee RFQ API</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 760px; margin: 80px auto; padding: 0 24px; color: #1f2937;
         line-height: 1.55; }
  h1 { margin: 0 0 4px; font-size: 22px; }
  h2 { margin: 36px 0 12px; font-size: 16px; color: #374151; }
  .sub { color: #6b7280; font-size: 14px; margin-bottom: 28px; }
  a.btn { display: inline-block; padding: 10px 18px; background: #111827;
          color: #fff; text-decoration: none; border-radius: 8px;
          font-weight: 500; font-size: 15px; }
  a.btn:hover { background: #374151; }
  ul { list-style: none; padding: 0; margin: 28px 0 0; font-size: 14px; }
  ul li { padding: 6px 0; color: #6b7280; }
  ul li a { color: #2563eb; text-decoration: none; }
  ul li a:hover { text-decoration: underline; }
  code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
  .mode { display: inline-block; padding: 2px 8px; border-radius: 12px;
          background: __MODE_BG__; color: __MODE_FG__; font-size: 12px; font-weight: 500; }
  details { margin-top: 28px; border: 1px solid #e5e7eb; border-radius: 10px;
            padding: 12px 16px; background: #fafafa; }
  details summary { cursor: pointer; font-size: 14px; font-weight: 500;
                    color: #374151; }
  details[open] summary { margin-bottom: 12px; }
  .mermaid { background: #fff; padding: 12px; border-radius: 6px; }
  .legend { font-size: 12px; color: #6b7280; margin-top: 10px; }
</style>
</head>
<body>
  <h1>Renbee RFQ API</h1>
  <div class="sub">Llama-backed RFQ generation for residential decarbonisation enquiries.
  Mode: <span class="mode">__MODE_LABEL__</span></div>
  <a class="btn" href="/demo">Open homeowner demo →</a>

  <details open>
    <summary>How it works (flow)</summary>
    <div class="mermaid">
flowchart TD
    A[Homeowner enters<br/>postcode + technology] --> B{Look up<br/>property's EPC}
    A --> S[Site intelligence:<br/>planning + grid<br/>checks in parallel]
    B -->|0 records<br/>for postcode| N{Look at<br/>nearby postcodes}
    B -->|1 record| E[Use the homeowner's<br/>own EPC]
    B -->|multiple<br/>addresses| D[Ask homeowner<br/>which is their<br/>address]
    N -->|no nearby<br/>EPCs either| C[Ask homeowner<br/>4 property questions:<br/>type / area /<br/>heating / fuel]
    N -->|nearby<br/>EPCs found| NP[Show homeowner<br/>the nearby properties<br/>and let them choose]
    NP -->|pick the one<br/>most like mine| P2A[Use that property<br/>as a proxy estimate]
    NP -->|use the typical<br/>values across all| P2B[Average across all<br/>nearby properties]
    D -->|picks their<br/>address| E
    D -->|use the typical<br/>values for this<br/>street| P1[Average across all<br/>properties on the<br/>same street]
    C & E & P1 & P2A & P2B & S --> F{Step 2: tech<br/>specific form}
    F -->|heat pump| G1[emitter / cylinder /<br/>external space / garden access /<br/>bedrooms / bathrooms /<br/>occupants / smart meter]
    F -->|solar PV| G2[orientation / shading /<br/>roof condition]
    F -->|battery| G3[existing PV / purpose /<br/>backup need / space /<br/>location]
    F -->|solar thermal| G4[orientation / shading /<br/>cylinder / occupants /<br/>bathrooms]
    G1 & G2 & G3 & G4 --> H[Llama 3.3 70B:<br/>homeowner recommendation<br/>+ indicative cost band]
    G1 & G2 & G3 & G4 --> J[Llama 3.3 70B:<br/>installer-facing RFQ<br/>strict no-fabrication prompt]
    H & J --> K[One page: recommendation<br/>+ editable RFQ<br/>HITL review then submit]
    style A fill:#dbeafe,stroke:#3b82f6
    style B fill:#fef3c7,stroke:#f59e0b
    style N fill:#fef3c7,stroke:#f59e0b
    style NP fill:#fef3c7,stroke:#f59e0b
    style F fill:#fef3c7,stroke:#f59e0b
    style P1 fill:#e0e7ff,stroke:#6366f1
    style P2A fill:#e0e7ff,stroke:#6366f1
    style P2B fill:#e0e7ff,stroke:#6366f1
    style S fill:#f3e8ff,stroke:#a855f7
    style H fill:#dcfce7,stroke:#22c55e
    style J fill:#dcfce7,stroke:#22c55e
    style K fill:#fce7f3,stroke:#ec4899
    </div>
    <div class="legend">Yellow = decisions · Green = LLM moments · Blue = entry · Indigo = proxy EPC fallback · Purple = automatic site intelligence · Pink = HITL handoff to installer.</div>
  </details>

  <h2>Endpoints</h2>
  <ul>
    <li><a href="/docs">/docs</a> — interactive Swagger UI for the API</li>
    <li><a href="/health">/health</a> — health check JSON</li>
    <li><code>POST /api/initiate</code> — Step 1: postcode + technology</li>
    <li><code>POST /api/generate</code> — homeowner recommendation</li>
    <li><code>POST /api/generate-rfq</code> — installer-facing RFQ</li>
  </ul>

  <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
  <script>
    mermaid.initialize({ startOnLoad: true, theme: 'default', securityLevel: 'loose' });
  </script>
</body>
</html>"""


@app.get("/", include_in_schema=False)
def root():
    if DEMO_MOCK_LLM:
        html = _ROOT_HTML.replace("__MODE_LABEL__", "demo · LLM mocked") \
                         .replace("__MODE_BG__", "#fef3c7") \
                         .replace("__MODE_FG__", "#92400e")
    else:
        html = _ROOT_HTML.replace("__MODE_LABEL__", "hosted Llama 3.3 70B") \
                         .replace("__MODE_BG__", "#dcfce7") \
                         .replace("__MODE_FG__", "#166534")
    return HTMLResponse(html)


@app.post(
    "/api/initiate",
    response_model=InitiateResponse,
    responses={
        409: {
            "model": AmbiguousAddressResponse,
            "description": (
                "Either ambiguous_address (postcode has multiple distinct "
                "addresses) or proxy_nearby_candidates (input postcode has "
                "zero EPCs but nearby postcodes have EPCs to choose from)."
            ),
        }
    },
)
def initiate(req: InitiateRequest) -> InitiateResponse:
    """Step 1: postcode + technology -> EPC fetch + assemble + missing_fields list."""
    from epc_to_rfq import _TECH_SECTIONS
    if req.technology not in _TECH_SECTIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_technology",
                "expected": sorted(_TECH_SECTIONS.keys()),
                "received": req.technology,
            },
        )

    # Decide how to source EPC data. Four entry paths:
    #   use_proxy=True               → same-postcode aggregate (Layer 1)
    #   use_proxy_nearby_average=True → nearby-postcode aggregate (Layer 2 avg)
    #   lmk_key + proxy_postcode     → fetch that specific nearby EPC (Layer 2 pick)
    #   default                      → fetch input postcode; if 0 EPCs, show
    #                                   Layer 2 picker (or Case B if no neighbours)
    auto_proxy_used = False
    proxy_postcodes_used: list[str] = [req.postcode]
    picked_section: Optional[dict] = None
    picked_recommendation: Optional[dict] = None

    if req.lmk_key and req.proxy_postcode:
        # User picked a single nearby EPC from the Layer 2 picker. Re-fetch
        # that postcode and pull out the chosen certificate.
        try:
            nearby_data = fetch_epc_data(req.proxy_postcode)
        except Exception:
            nearby_data = {"postcode": req.proxy_postcode, "count": 0, "properties": []}
        chosen = None
        for p in nearby_data.get("properties") or []:
            if (p.get("certificate") or {}).get("lmk-key") == req.lmk_key:
                chosen = p
                break
        if chosen is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "unknown_lmk_key",
                    "lmk_key": req.lmk_key,
                    "proxy_postcode": req.proxy_postcode,
                },
            )
        picked_section = build_picked_property_section(
            chosen.get("certificate") or {}, req.proxy_postcode
        )
        picked_recommendation = map_recommendation_section(
            chosen.get("recommendations") or []
        )
        picked_recommendation["recommendation_source"] = "proxy_aggregate"
        # epc_data carries just the chosen cert so the session has a record.
        epc_data = {
            "postcode": req.postcode,
            "count": 1,
            "properties": [chosen],
            "proxy_postcodes": [req.proxy_postcode],
        }

    else:
        # Wrap the initial fetch so a bad/unknown postcode (gov.uk returns 400)
        # doesn't 500 the request — fall through to the proxy / Case B paths.
        try:
            epc_data = fetch_epc_data(req.postcode)
        except Exception:
            epc_data = {"postcode": req.postcode, "count": 0, "properties": []}

        # Layer 2 trigger: input postcode has 0 EPCs and the caller hasn't
        # opted in to a specific resolution.
        if (
            not req.use_proxy
            and not req.use_proxy_nearby_average
            and epc_data.get("count", 0) == 0
        ):
            from epc_fetch import fetch_epc_data_with_neighbour_fallback
            combined = fetch_epc_data_with_neighbour_fallback(req.postcode)
            if combined.get("count", 0) >= 1:
                # Surface the candidates to the homeowner instead of silently
                # aggregating. They'll either pick one, ask for the average,
                # or fall through to Case B manual entry.
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "proxy_nearby_candidates",
                        "candidates": build_nearby_candidate_list(combined),
                    },
                )
            # No nearby EPCs either — fall through to Case B with empty data.

        # Layer 2 "use average of all" follow-up call.
        if req.use_proxy_nearby_average and epc_data.get("count", 0) == 0:
            from epc_fetch import fetch_epc_data_with_neighbour_fallback
            combined = fetch_epc_data_with_neighbour_fallback(req.postcode)
            if combined.get("proxy_used"):
                epc_data = combined
                auto_proxy_used = True
                proxy_postcodes_used = combined.get("proxy_postcodes") or []

    user_form = {"common": {"technology_requested": req.technology}}

    if picked_section is not None:
        # Bypass assemble_rfq_input's property/recommendation building — we've
        # already produced both sections from the user's chosen nearby EPC.
        # Still call assemble_rfq_input with empty data to get the standard
        # common/technology scaffolding, then overlay our picked sections.
        rfq_input = assemble_rfq_input(
            {"postcode": req.postcode, "count": 0, "properties": []},
            user_form,
        )
        rfq_input["property"] = picked_section
        rfq_input["recommendation"] = picked_recommendation
    else:
        try:
            rfq_input = assemble_rfq_input(
                epc_data,
                user_form,
                house_number=req.house_number,
                lmk_key=req.lmk_key,
                use_proxy=req.use_proxy or req.use_proxy_nearby_average or auto_proxy_used,
            )
        except AmbiguousAddress as e:
            raise HTTPException(
                status_code=409,
                detail={"error": "ambiguous_address", "candidates": e.candidates},
            )

        # Tag the property section so the front-end / LLM can distinguish:
        #   "proxy"          → user clicked "use street average" (same postcode)
        #   "proxy_nearby"   → Layer 2 "use average of all" follow-up
        prop = rfq_input.get("property") or {}
        if prop.get("epc_source") == "proxy":
            prop["proxy_postcodes"] = proxy_postcodes_used
            if req.use_proxy_nearby_average or auto_proxy_used:
                prop["epc_source"] = "proxy_nearby"

    # Site intelligence — planning constraints + grid headroom. External
    # APIs, failure-tolerant; we never let these block the RFQ pipeline.
    try:
        site_context = build_site_context(req.postcode)
    except Exception:
        site_context = {"planning": {}, "grid": None, "data_sources": []}
    rfq_input["site_context"] = site_context

    missing = missing_fields(rfq_input)
    optional = optional_fields(rfq_input)
    score = completeness_score(rfq_input)

    sid = str(uuid.uuid4())
    _SESSIONS[sid] = {"rfq_input": rfq_input, "epc_data": epc_data}

    return InitiateResponse(
        session_id=sid,
        epc_found=rfq_input["property"]["epc_found"],
        auto_filled=rfq_input["property"],
        missing_fields={k: [FieldSpec(**f) for f in v] for k, v in missing.items()},
        optional_fields={k: [FieldSpec(**f) for f in v] for k, v in optional.items()},
        completeness=score,
        site_context=site_context,
    )


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Step 2 done: produce the homeowner-facing recommendation summary."""
    session = _require_session(req.session_id)
    rfq_input = _merge_additional_fields(session["rfq_input"], req.additional_fields)

    still_missing = missing_fields(rfq_input)
    if still_missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "still_missing_required_fields",
                "missing_fields": still_missing,
            },
        )

    _attach_cost_estimate(rfq_input)

    from generate_rfq import generate_recommendation
    rec = generate_recommendation(rfq_input)

    session["rfq_input"] = rfq_input
    session["recommendation"] = rec

    return GenerateResponse(
        session_id=req.session_id,
        recommendation_summary=rec.get("recommendation_summary", ""),
        recommendation_disclaimer=rec.get("recommendation_disclaimer", ""),
        rfq_input=rfq_input,
        completeness=completeness_score(rfq_input),
        raw_response=rec.get("raw_response"),
        parse_status=rec.get("parse_status"),
    )


@app.post("/api/generate-rfq", response_model=GenerateRFQResponse)
def generate_rfq(req: GenerateRFQRequest) -> GenerateRFQResponse:
    """Second LLM moment: produce the installer-facing RFQ summary.

    `vendor_id` is optional tracking only. There is no vendor-selection step in
    the flow; Renbee matches installers itself.
    """
    session = _require_session(req.session_id)
    rfq_input = session["rfq_input"]

    still_missing = missing_fields(rfq_input)
    if still_missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "rfq_input_not_ready",
                "missing_fields": still_missing,
            },
        )

    # A client may skip /api/generate entirely, so derive it here too. Pure and
    # offline, so recomputing costs nothing.
    _attach_cost_estimate(rfq_input)

    from generate_rfq import generate_rfq_summary
    out = generate_rfq_summary(rfq_input)

    session["rfq_summary"] = out.get("rfq_summary", "")
    session["vendor_id"] = req.vendor_id

    return GenerateRFQResponse(
        session_id=req.session_id,
        rfq_summary=out.get("rfq_summary", ""),
        rfq_input=rfq_input,
        ready_to_submit=True,
        raw_response=out.get("raw_response"),
        parse_status=out.get("parse_status"),
    )


# Where finalised RFQs are written. Renbee retrieves these to share with the
# chosen installer. Sits next to the app, separate from the EPC fetch cache.
RFQ_OUTPUT_DIR = Path(__file__).parent / "rfq_outputs"


def _sanitise_token(s: str) -> str:
    """Make a filesystem-safe token (keep alnum, dash, underscore)."""
    keep = "".join(c if (c.isalnum() or c in "-_") else "" for c in (s or "").strip())
    return keep or "rfq"


@app.post("/api/save-rfq", response_model=SaveRFQResponse)
def save_rfq(req: SaveRFQRequest) -> SaveRFQResponse:
    """Persist the final (reviewer-edited) RFQ to disk so Renbee can share it
    with the installer. Accepts the edited prose + structured input from the
    client (edits live in the browser until this point) and writes one combined
    JSON file. Does not require a live session — the client holds the content.
    """
    RFQ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Name token: postcode from the input if present, else session id, else generic.
    postcode = ((req.rfq_input.get("common") or {}).get("postcode")) if isinstance(req.rfq_input, dict) else None
    token = _sanitise_token(postcode or req.session_id or "rfq")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"renbee_rfq_{token}_{stamp}.json"

    vendor_id = None
    if req.session_id and req.session_id in _SESSIONS:
        session = _SESSIONS[req.session_id]
        # Reflect the final edited version back into the session for traceability.
        session["rfq_summary"] = req.rfq_summary
        session["rfq_input"] = req.rfq_input
        vendor_id = session.get("vendor_id")

    content = {
        "rfq_summary": req.rfq_summary,
        "rfq_input": req.rfq_input,
        "session_id": req.session_id,
        "vendor_id": vendor_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(RFQ_OUTPUT_DIR / filename, "w") as f:
        json.dump(content, f, indent=2)

    return SaveRFQResponse(saved=True, filename=filename, path=f"rfq_outputs/{filename}")


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict:
    """Inspect a session's state (debug / HITL review)."""
    session = _require_session(session_id)
    return {
        "session_id": session_id,
        "rfq_input": session.get("rfq_input"),
        "recommendation": session.get("recommendation"),
        "rfq_summary": session.get("rfq_summary"),
        "vendor_id": session.get("vendor_id"),
        "completeness": completeness_score(session.get("rfq_input", {})),
        "missing_fields": missing_fields(session.get("rfq_input", {})),
    }
