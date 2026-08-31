"""End-to-end CLI: postcode + user form -> EPC fetch -> assembler -> LLM -> summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from epc_fetch import fetch_epc_data
from epc_to_rfq import (
    AmbiguousAddress,
    assemble_rfq_input,
    completeness_score,
    missing_fields,
)


def build_rfq_input(
    postcode: str,
    user_form: dict,
    house_number: Optional[str] = None,
) -> dict:
    epc_data = fetch_epc_data(postcode)
    return assemble_rfq_input(epc_data, user_form, house_number=house_number)


def generate_summaries(rfq_input: dict) -> dict:
    """Run both production prompts and merge their outputs.

    Calls the same two functions the live API uses, so this CLI mode and
    the live customer journey are evaluated against identical prompts.
    """
    from generate_rfq import generate_recommendation, generate_rfq_summary

    rfq = generate_rfq_summary(rfq_input)
    rec = generate_recommendation(rfq_input)
    return {**rfq, **rec}


def _example_form() -> dict:
    return {
        "common": {
            "technology_requested": "heat_pump",
            "preferred_contact_method": "email",
            "contact_email": "homeowner@example.com",
            "contact_phone": "07700 900123",
            "desired_installation_timeline": "within_6_months",
            "budget_range": "GBP 8000-12000",
            "motivation": "reduce energy bills and replace gas heating",
            "additional_notes": "Prefers weekday site visits.",
        },
        "heat_pump": {
            "heat_pump_type_interest": "ground_air_source",
            "emitter_type": "radiators",
            "radiator_suitability_known": "unknown",
            "insulation_status_basic": "moderate",
            "hot_water_cylinder_space_available": "yes",
            "external_unit_space": "yes",
            "garden_or_side_access": "yes",
            "number_of_bedrooms": 3,
            "number_of_bathrooms": 2,
            "number_of_occupants": 4,
            "smart_meter_installed": "yes",
            "smart_meter_cutout_fuse_label": "100A",
            "boiler_age_years": 12,
            "noise_or_planning_constraints": "close to neighbour boundary",
            "bus_interest": "yes",
        },
        "solar_pv": {},
        "property_overrides": {
            "access_constraints": "narrow side access",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renbee RFQ pipeline")
    parser.add_argument("postcode", help="UK postcode (e.g. 'E1 6AN')")
    parser.add_argument(
        "house_number",
        nargs="?",
        default=None,
        help="Optional house/flat number to disambiguate multi-address postcodes",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true",
                      help="Run both prompts (RFQ + recommendation) and merge outputs")
    mode.add_argument("--rfq", action="store_true",
                      help="Run installer-facing RFQ prompt only")
    mode.add_argument("--recommendation", action="store_true",
                      help="Run homeowner-facing recommendation prompt only")
    args = parser.parse_args()

    try:
        rfq_input = build_rfq_input(
            args.postcode, _example_form(), house_number=args.house_number
        )
    except AmbiguousAddress as e:
        print(f"AMBIGUOUS: {e}", file=sys.stderr)
        sys.exit(2)

    # Always show validation diagnostics
    diagnostics = {
        "completeness": completeness_score(rfq_input),
        "missing_fields": missing_fields(rfq_input),
    }

    if args.generate or args.rfq or args.recommendation:
        from generate_rfq import generate_recommendation, generate_rfq_summary

        output: dict = {}
        if args.generate:
            output = generate_summaries(rfq_input)
        elif args.rfq:
            output = generate_rfq_summary(rfq_input)
        elif args.recommendation:
            output = generate_recommendation(rfq_input)

        print(json.dumps({
            "input": rfq_input,
            "diagnostics": diagnostics,
            "output": output,
        }, indent=2))
    else:
        print(json.dumps({
            "input": rfq_input,
            "diagnostics": diagnostics,
        }, indent=2))
