"""Build real-postcode-derived evaluation cases.

No LLM and no Gemini are involved here, so this runs now against the EPC and
planning APIs, independent of the pending Vertex judge billing.

Usage:
    python -m evaluation.build_cases
    python -m evaluation.build_cases --out rfq_cases_real_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cost_tables import build_cost_estimate
from epc_fetch import fetch_epc_data, find_nearby_postcodes
from epc_to_rfq import (
    FIELDS,
    AmbiguousAddress,
    _TECH_SECTIONS,
    _is_empty,
    assemble_rfq_input,
    build_picked_property_section,
    build_site_context,
    completeness_score,
    map_recommendation_section,
)


def _missing_required(rfq_input: dict) -> list[str]:
    """Return the names of required fields that are still empty (section.field)."""
    technology = (rfq_input.get("common") or {}).get("technology_requested")
    sections = ["common", "property"]
    if technology in _TECH_SECTIONS:
        sections.append(_TECH_SECTIONS[technology])
    missing = []
    for section_name in sections:
        section_data = rfq_input.get(section_name) or {}
        for field_name, meta in FIELDS.get(section_name, {}).items():
            if meta.get("required") and _is_empty(section_data.get(field_name)):
                missing.append(f"{section_name}.{field_name}")
    return missing

DEFAULT_OUT = ROOT / "rfq_cases_real_v1.json"


# --------------------------------------------------------------------------
# Curated postcodes.
#
# `mode` controls how the EPC layer is exercised:
#   epc          - normal single-property EPC. If the postcode resolves to
#                  several addresses we auto-pick the first candidate (so the
#                  case is deterministic across runs).
#   proxy_same   - aggregate every EPC in the postcode (Layer 1 proxy).
#   proxy_nearby - postcode has no EPC; aggregate nearby postcodes (Layer 2).
#   caseB        - no EPC at all; manual entry via property_overrides.
#
# Form answers are varied across cases (emitter type, roof orientation, backup
# power, timelines, motivations) so the set is not uniform.
# --------------------------------------------------------------------------
CONFIG: list[dict] = [
    {
        "case_id": "REAL_001",
        "postcode": "E1 6AN",
        "technology": "heat_pump",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner01@example.com",
                "contact_phone": "07700 900001",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Cut gas use and lower carbon emissions.",
            },
            "heat_pump": {
                "emitter_type": "radiators",
                "hot_water_cylinder_space_available": "yes",
                "external_unit_space": "yes",
                "garden_or_side_access": "yes",
                "heat_pump_type_interest": "ground_air_source",
                "number_of_bedrooms": 2,
                "number_of_bathrooms": 1,
                "number_of_occupants": 2,
                "smart_meter_installed": "yes",
            },
        },
    },
    {
        "case_id": "REAL_002",
        "postcode": "OX7 3EL",
        "technology": "solar_pv",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner02@example.com",
                "contact_phone": "07700 900002",
                "desired_installation_timeline": "flexible",
                "motivation": "Reduce electricity bills and generate own power.",
            },
            "solar_pv": {
                "roof_orientation": "south",
                "usable_roof_area_m2": 28,
                "roof_shading_level": "low",
                "roof_condition": "yes",
                "number_of_bedrooms": 3,
                "number_of_bathrooms": 2,
                "number_of_occupants": 4,
            },
        },
    },
    {
        "case_id": "REAL_003",
        "postcode": "BA2 6AA",
        "technology": "solar_thermal",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner03@example.com",
                "contact_phone": "07700 900003",
                "desired_installation_timeline": "within_12_months",
                "motivation": "Heat water more cheaply over the summer.",
            },
            "solar_thermal": {
                "roof_orientation": "southeast",
                "usable_roof_area_m2": 6,
                "roof_shading_level": "none",
                "hot_water_cylinder_space_available": "yes",
                "number_of_occupants": 4,
                "number_of_bathrooms": 2,
            },
        },
    },
    {
        "case_id": "REAL_004",
        "postcode": "SE5 8AA",
        "technology": "battery",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner04@example.com",
                "contact_phone": "07700 900004",
                "desired_installation_timeline": "within_3_months",
                "motivation": "Store solar power and have backup during outages.",
            },
            "battery": {
                "existing_solar_pv": "existing",
                "battery_purpose": "both",
                "backup_power_required": "yes",
                "battery_location_preference": "garage",
                "battery_space_available": "yes",
            },
        },
    },
    {
        "case_id": "REAL_005",
        "postcode": "RG9 1AY",
        "technology": "heat_pump",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner05@example.com",
                "contact_phone": "07700 900005",
                "desired_installation_timeline": "asap",
                "motivation": "Replace an ageing oil boiler.",
            },
            "heat_pump": {
                "emitter_type": "underfloor_heating",
                "hot_water_cylinder_space_available": "yes",
                "external_unit_space": "yes",
                "garden_or_side_access": "no",
                "heat_pump_type_interest": "ground_air_source",
                "number_of_bedrooms": 1,
                "number_of_bathrooms": 1,
                "number_of_occupants": 1,
                "smart_meter_installed": "no",
            },
        },
    },
    {
        "case_id": "REAL_006",
        "postcode": "CV12 8UE",
        "technology": "solar_pv",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner06@example.com",
                "contact_phone": "07700 900006",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Charge an electric vehicle from the roof.",
            },
            "solar_pv": {
                "roof_orientation": "west",
                "usable_roof_area_m2": 18,
                "roof_shading_level": "moderate",
                "roof_condition": "yes",
                "number_of_bedrooms": 2,
                "number_of_bathrooms": 1,
                "number_of_occupants": 2,
            },
        },
    },
    {
        "case_id": "REAL_007",
        "postcode": "GL54 2BP",
        "technology": "heat_pump",
        "mode": "proxy_nearby",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner07@example.com",
                "contact_phone": "07700 900007",
                "desired_installation_timeline": "flexible",
                "motivation": "Explore options for a rural off-gas property.",
            },
            "heat_pump": {
                "emitter_type": "mixed",
                "hot_water_cylinder_space_available": "unknown",
                "external_unit_space": "yes",
                "garden_or_side_access": "yes",
                "heat_pump_type_interest": "ground_air_source",
                "number_of_bedrooms": 2,
                "number_of_bathrooms": 1,
                "number_of_occupants": 3,
                "smart_meter_installed": "yes",
            },
        },
    },
    {
        "case_id": "REAL_008",
        "postcode": "SW1A 2AA",
        "technology": "battery",
        "mode": "caseB",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner08@example.com",
                "contact_phone": "07700 900008",
                "desired_installation_timeline": "within_12_months",
                "motivation": "Add storage to an existing solar array.",
            },
            "battery": {
                "existing_solar_pv": "planned",
                "battery_purpose": "self_consumption",
                "backup_power_required": "no",
                "battery_location_preference": "utility_room",
                "battery_space_available": "yes",
            },
            "property_overrides": {
                "property_type": "flat",
                "floor_area_m2": 70,
                "current_heating_system": "electric_storage_heaters",
                "current_fuel_type": "electricity",
            },
        },
    },
    {
        "case_id": "REAL_009",
        "postcode": "LS6 1AA",
        "technology": "solar_pv",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner09@example.com",
                "contact_phone": "07700 900009",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Lower bills in a shared rented house.",
            },
            "solar_pv": {
                "roof_orientation": "southwest",
                "usable_roof_area_m2": 22,
                "roof_shading_level": "none",
                "roof_condition": "no",
                "number_of_bedrooms": 2,
                "number_of_bathrooms": 1,
                "number_of_occupants": 3,
            },
        },
    },
    {
        "case_id": "REAL_010",
        "postcode": "PL4 7AA",
        "technology": "solar_thermal",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner10@example.com",
                "contact_phone": "07700 900010",
                "desired_installation_timeline": "within_3_months",
                "motivation": "Cut the cost of heating water for a large family.",
            },
            "solar_thermal": {
                "roof_orientation": "south",
                "usable_roof_area_m2": 5,
                "roof_shading_level": "low",
                "hot_water_cylinder_space_available": "yes",
                "number_of_occupants": 5,
                "number_of_bathrooms": 2,
            },
        },
    },
    # ---- Expansion cases (REAL_011+): fill coverage gaps -------------------
    # National Parks, EPC extremes (A/F/G via select_rating), the proxy_nearby
    # variants, a Layer-1 proxy_same case, and one deliberately-incomplete case.
    # Grid and listed-building are intentionally not covered: the UKPN feed is
    # retired, and listed_building is never True for a postcode centroid.
    {
        "case_id": "REAL_011",
        "postcode": "LA22 9SH",  # Lake District (Grasmere)
        "technology": "heat_pump",
        "mode": "epc",
        "select_rating": "A",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner11@example.com",
                "contact_phone": "07700 900011",
                "desired_installation_timeline": "within_12_months",
                "motivation": "Low-carbon heating for a high-efficiency Lake District home.",
            },
            "heat_pump": {
                "emitter_type": "radiators",
                "hot_water_cylinder_space_available": "yes",
                "external_unit_space": "yes",
                "garden_or_side_access": "yes",
                "heat_pump_type_interest": "solar_assisted",
                "absorber_mounting_location": "pitched_roof",
                "number_of_bedrooms": 4,
                "number_of_bathrooms": 2,
                "number_of_occupants": 4,
                "smart_meter_installed": "no",
            },
        },
    },
    {
        "case_id": "REAL_012",
        "postcode": "YO62 5AD",  # North York Moors (Helmsley)
        "technology": "solar_pv",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner12@example.com",
                "contact_phone": "07700 900012",
                "desired_installation_timeline": "flexible",
                "motivation": "Generate own power in a national park village.",
            },
            "solar_pv": {
                "roof_orientation": "south",
                "usable_roof_area_m2": 20,
                "roof_shading_level": "low",
                "roof_condition": "yes",
                "number_of_bedrooms": 3,
                "number_of_bathrooms": 1,
                "number_of_occupants": 4,
            },
        },
    },
    {
        "case_id": "REAL_013",
        "postcode": "SO43 7BQ",  # New Forest (Lyndhurst), Article 4
        "technology": "battery",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner13@example.com",
                "contact_phone": "07700 900013",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Store energy and reduce grid reliance.",
            },
            "battery": {
                "existing_solar_pv": "existing",
                "battery_purpose": "self_consumption",
                "backup_power_required": "no",
                "battery_location_preference": "garage",
                "battery_space_available": "yes",
            },
        },
    },
    {
        "case_id": "REAL_014",
        "postcode": "TQ13 7TB",  # Dartmoor (Widecombe)
        "technology": "solar_thermal",
        "mode": "epc",
        "select_rating": "F",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner14@example.com",
                "contact_phone": "07700 900014",
                "desired_installation_timeline": "within_12_months",
                "motivation": "Cheaper hot water for an old moorland cottage.",
            },
            "solar_thermal": {
                "roof_orientation": "south",
                "usable_roof_area_m2": 4,
                "roof_shading_level": "none",
                "hot_water_cylinder_space_available": "yes",
                "number_of_occupants": 3,
                "number_of_bathrooms": 1,
            },
        },
    },
    {
        "case_id": "REAL_015",
        "postcode": "DL8 3RA",  # Yorkshire Dales (Wensleydale)
        "technology": "heat_pump",
        "mode": "epc",
        "select_rating": "G",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner15@example.com",
                "contact_phone": "07700 900015",
                "desired_installation_timeline": "flexible",
                "motivation": "Replace expensive off-gas heating.",
            },
            "heat_pump": {
                "emitter_type": "radiators",
                "hot_water_cylinder_space_available": "yes",
                "external_unit_space": "yes",
                "garden_or_side_access": "yes",
                "heat_pump_type_interest": "solar_assisted",
                "absorber_mounting_location": "pitched_roof",
                "number_of_bedrooms": 3,
                "number_of_bathrooms": 2,
                "number_of_occupants": 3,
                "smart_meter_installed": "yes",
            },
        },
    },
    {
        "case_id": "REAL_016",
        "postcode": "DE45 1BT",  # Peak District (Bakewell), 0 own EPCs
        "technology": "solar_pv",
        "mode": "proxy_nearby",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner16@example.com",
                "contact_phone": "07700 900016",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Explore solar for a Peak District home with no EPC on record.",
            },
            "solar_pv": {
                "roof_orientation": "southwest",
                "usable_roof_area_m2": 25,
                "roof_shading_level": "low",
                "roof_condition": "yes",
                "number_of_bedrooms": 1,
                "number_of_bathrooms": 1,
                "number_of_occupants": 1,
            },
        },
    },
    {
        "case_id": "REAL_017",
        "postcode": "BA1 1LZ",  # Bath WHS + conservation, 0 own EPCs
        "technology": "heat_pump",
        "mode": "proxy_nearby_pick",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner17@example.com",
                "contact_phone": "07700 900017",
                "desired_installation_timeline": "flexible",
                "motivation": "Assess heat pump feasibility near Bath city centre.",
            },
            "heat_pump": {
                "emitter_type": "mixed",
                "hot_water_cylinder_space_available": "unknown",
                "external_unit_space": "yes",
                "garden_or_side_access": "no",
                "heat_pump_type_interest": "ground_air_source",
                "number_of_bedrooms": 2,
                "number_of_bathrooms": 1,
                "number_of_occupants": 2,
                "smart_meter_installed": "no",
            },
        },
    },
    {
        "case_id": "REAL_018",
        "postcode": "CV34 4BJ",  # Warwick conservation, many own EPCs
        "technology": "battery",
        "mode": "proxy_same",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner18@example.com",
                "contact_phone": "07700 900018",
                "desired_installation_timeline": "within_3_months",
                "motivation": "Add storage in a Warwick townhouse.",
            },
            "battery": {
                "existing_solar_pv": "planned",
                "battery_purpose": "both",
                "backup_power_required": "yes",
                # Mid-floor flat: no garage or utility room, so the one case
                # that exercises the "no space" branch.
                "battery_location_preference": "unsure",
                "battery_space_available": "no",
            },
        },
    },
    {
        "case_id": "REAL_019",
        "postcode": "MK10 9AA",  # modern Milton Keynes (Broughton)
        "technology": "solar_pv",
        "mode": "epc",
        "select_rating": "A",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner19@example.com",
                "contact_phone": "07700 900019",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Maximise self-generation on a new-build.",
            },
            "solar_pv": {
                "roof_orientation": "south",
                "usable_roof_area_m2": 30,
                "roof_shading_level": "none",
                "roof_condition": "yes",
                "number_of_bedrooms": 5,
                "number_of_bathrooms": 3,
                "number_of_occupants": 5,
            },
        },
    },
    {
        "case_id": "REAL_020",
        "postcode": "NR21 9AA",  # rural Norfolk (Fakenham) conservation
        "technology": "heat_pump",
        "mode": "epc",
        "select_rating": "G",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner20@example.com",
                "contact_phone": "07700 900020",
                "desired_installation_timeline": "within_12_months",
                "motivation": "Move off oil in a poorly-rated rural home.",
            },
            "heat_pump": {
                "emitter_type": "radiators",
                "hot_water_cylinder_space_available": "yes",
                "external_unit_space": "yes",
                "garden_or_side_access": "yes",
                "heat_pump_type_interest": "ground_air_source",
                "number_of_bedrooms": 2,
                "number_of_bathrooms": 1,
                "number_of_occupants": 2,
                "smart_meter_installed": "yes",
            },
        },
    },
    {
        "case_id": "REAL_021",
        "postcode": "PL15 8AA",  # Launceston conservation. DELIBERATELY INCOMPLETE.
        "technology": "solar_thermal",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner21@example.com",
                "contact_phone": "07700 900021",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Cut water heating costs.",
            },
            # number_of_occupants (required) omitted on purpose so this case
            # exercises the missing_fields path and scores completeness < 1.0.
            "solar_thermal": {
                "roof_orientation": "southeast",
                "usable_roof_area_m2": 4,
                "roof_shading_level": "low",
                "hot_water_cylinder_space_available": "yes",
                "number_of_bathrooms": 2,
            },
        },
    },
    {
        "case_id": "REAL_022",
        "postcode": "TR19 7AA",  # Cornwall AONB, 0 own EPCs
        "technology": "battery",
        "mode": "proxy_nearby",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner22@example.com",
                "contact_phone": "07700 900022",
                "desired_installation_timeline": "flexible",
                "motivation": "Store solar on the Cornish coast.",
            },
            "battery": {
                "existing_solar_pv": "existing",
                "battery_purpose": "backup_power",
                "backup_power_required": "yes",
                "battery_location_preference": "garage",
                "battery_space_available": "yes",
            },
        },
    },
    {
        "case_id": "REAL_023",
        "postcode": "OX1 3BG",  # central Oxford conservation + Article 4, 0 own EPCs
        "technology": "solar_pv",
        "mode": "proxy_nearby_pick",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner23@example.com",
                "contact_phone": "07700 900023",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Solar for a central Oxford property.",
            },
            "solar_pv": {
                "roof_orientation": "south",
                "usable_roof_area_m2": 18,
                "roof_shading_level": "moderate",
                "roof_condition": "no",
                "number_of_bedrooms": 2,
                "number_of_bathrooms": 1,
                "number_of_occupants": 2,
            },
        },
    },
    {
        "case_id": "REAL_024",
        "postcode": "YO1 7HH",  # York centre conservation + Article 4
        "technology": "solar_thermal",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner24@example.com",
                "contact_phone": "07700 900024",
                "desired_installation_timeline": "within_12_months",
                "motivation": "Hot water savings near York Minster.",
            },
            "solar_thermal": {
                "roof_orientation": "southwest",
                "usable_roof_area_m2": 5,
                "roof_shading_level": "low",
                "hot_water_cylinder_space_available": "yes",
                "number_of_occupants": 2,
                "number_of_bathrooms": 1,
            },
        },
    },
    {
        "case_id": "REAL_025",
        "postcode": "CV34 4BJ",  # Warwick conservation (reuse, different tech), EPC B
        "technology": "solar_thermal",
        "mode": "epc",
        "select_rating": "B",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner25@example.com",
                "contact_phone": "07700 900025",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Solar hot water for a Warwick flat.",
            },
            "solar_thermal": {
                "roof_orientation": "south",
                "usable_roof_area_m2": 4,
                "roof_shading_level": "none",
                "hot_water_cylinder_space_available": "yes",
                "number_of_occupants": 2,
                "number_of_bathrooms": 1,
            },
        },
    },
    {
        "case_id": "REAL_026",
        "postcode": "NR21 9AA",  # rural Norfolk (reuse, different tech)
        "technology": "battery",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner26@example.com",
                "contact_phone": "07700 900026",
                "desired_installation_timeline": "within_3_months",
                "motivation": "Back up power in a rural area with an unreliable supply.",
            },
            "battery": {
                "existing_solar_pv": "existing",
                "battery_purpose": "both",
                "backup_power_required": "yes",
                "battery_location_preference": "utility_room",
                "battery_space_available": "yes",
            },
        },
    },
    {
        "case_id": "REAL_027",
        "postcode": "YO62 5AD",  # North York Moors (reuse, different cert to REAL_012)
        "technology": "solar_thermal",
        "mode": "epc",
        "select_rating": "E",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner27@example.com",
                "contact_phone": "07700 900027",
                "desired_installation_timeline": "flexible",
                "motivation": "Summer hot water in a moors village.",
            },
            "solar_thermal": {
                "roof_orientation": "southeast",
                "usable_roof_area_m2": 5,
                "roof_shading_level": "low",
                "hot_water_cylinder_space_available": "yes",
                "number_of_occupants": 4,
                "number_of_bathrooms": 2,
            },
        },
    },
    {
        "case_id": "REAL_028",
        "postcode": "SO43 7BQ",  # New Forest (reuse, different cert to REAL_013)
        "technology": "solar_pv",
        "mode": "epc",
        "select_rating": "B",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner28@example.com",
                "contact_phone": "07700 900028",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Generate power in the New Forest.",
            },
            "solar_pv": {
                "roof_orientation": "south",
                "usable_roof_area_m2": 24,
                "roof_shading_level": "low",
                "roof_condition": "yes",
                "number_of_bedrooms": 4,
                "number_of_bathrooms": 2,
                "number_of_occupants": 4,
            },
        },
    },
    {
        "case_id": "REAL_029",
        "postcode": "MK10 9AA",  # Broughton (reuse, different tech)
        "technology": "battery",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "email",
                "contact_email": "homeowner29@example.com",
                "contact_phone": "07700 900029",
                "desired_installation_timeline": "within_6_months",
                "motivation": "Store new-build solar output.",
            },
            "battery": {
                "existing_solar_pv": "existing",
                "battery_purpose": "self_consumption",
                "backup_power_required": "no",
                "battery_location_preference": "garage",
                "battery_space_available": "yes",
            },
        },
    },
    {
        "case_id": "REAL_030",
        "postcode": "TQ13 7TB",  # Dartmoor (reuse, different tech)
        "technology": "heat_pump",
        "mode": "epc",
        "form": {
            "common": {
                "preferred_contact_method": "phone",
                "contact_email": "homeowner30@example.com",
                "contact_phone": "07700 900030",
                "desired_installation_timeline": "flexible",
                "motivation": "Low-carbon heating on the moor.",
            },
            "heat_pump": {
                "emitter_type": "underfloor_heating",
                "hot_water_cylinder_space_available": "yes",
                "external_unit_space": "yes",
                "garden_or_side_access": "yes",
                "heat_pump_type_interest": "ground_air_source",
                "number_of_bedrooms": 4,
                "number_of_bathrooms": 2,
                "number_of_occupants": 5,
                "smart_meter_installed": "yes",
            },
        },
    },
]


def _build_user_form(entry: dict) -> dict:
    """Assemble the user_form dict expected by assemble_rfq_input."""
    tech = entry["technology"]
    form = entry.get("form", {})
    common = dict(form.get("common", {}))
    common["technology_requested"] = tech
    user_form: dict = {"common": common, tech: dict(form.get(tech, {}))}
    if form.get("property_overrides"):
        user_form["property_overrides"] = dict(form["property_overrides"])
    return user_form


def _empty_epc_data(postcode: str) -> dict:
    """Shape an empty EPC result so assemble_rfq_input takes the Case B path."""
    return {"postcode": postcode, "count": 0, "properties": []}


def _select_lmk_by_rating(epc_data: dict, rating: str) -> str | None:
    """lmk-key of the first certificate with the given EPC rating (for cases
    that need a specific rating, e.g. an A or a G in a mixed postcode)."""
    for p in epc_data.get("properties") or []:
        cert = p.get("certificate") or {}
        if (cert.get("current-energy-rating") or "").upper() == rating.upper():
            return cert.get("lmk-key")
    return None


def _cache_path(postcode: str) -> Path:
    return ROOT / "output" / f"{postcode.replace(' ', '_')}.json"


def _cached_fetch(postcode: str) -> dict:
    """Read the local output/ EPC cache if present, else fetch from the API.

    fetch_epc_data is write-only (it never reads its own cache), so without this
    every build re-hits the EPC API and trips its rate limit. This keeps the
    build idempotent: postcodes already fetched are served from disk.
    """
    path = _cache_path(postcode)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return fetch_epc_data(postcode)


def _cached_neighbour_fallback(postcode: str, target_count: int = 5,
                               max_neighbours: int = 10) -> dict:
    """Cache-aware version of fetch_epc_data_with_neighbour_fallback.

    Mirrors the production logic but routes every EPC fetch through the local
    cache. Nearest-postcode lookup uses postcodes.io (a separate API)."""
    primary = _cached_fetch(postcode)
    if primary.get("count", 0) > 0:
        primary["proxy_used"] = False
        primary["proxy_postcodes"] = [postcode]
        return primary
    accumulated: list = []
    contributing: list = []
    for pc in find_nearby_postcodes(postcode, max_count=max_neighbours):
        try:
            data = _cached_fetch(pc)
        except Exception:
            continue
        props = data.get("properties") or []
        if not props:
            continue
        accumulated.extend(props)
        contributing.append(pc)
        if len(accumulated) >= target_count:
            break
    return {
        "postcode": postcode,
        "fetched_at": primary.get("fetched_at"),
        "count": len(accumulated),
        "properties": accumulated,
        "proxy_used": bool(accumulated),
        "proxy_postcodes": contributing,
    }


def build_case(entry: dict) -> dict:
    """Run one curated postcode through the real pipeline and return a case
    dict shaped like rfq_cases_real_v1.json[i] ({case_id, input, output})."""
    pc = entry["postcode"]
    mode = entry["mode"]
    user_form = _build_user_form(entry)

    if mode == "caseB":
        epc_data = _empty_epc_data(pc)
        rfq_input = assemble_rfq_input(epc_data, user_form)
    elif mode == "proxy_nearby":
        # Layer 2 aggregate of nearby postcodes, tagged proxy_nearby (the
        # aggregate builder defaults to "proxy", so retag it here).
        epc_data = _cached_neighbour_fallback(pc)
        rfq_input = assemble_rfq_input(epc_data, user_form, use_proxy=True)
        prop = rfq_input.get("property") or {}
        if prop.get("epc_source") == "proxy":
            prop["epc_source"] = "proxy_nearby"
            prop["proxy_postcodes"] = epc_data.get("proxy_postcodes") or []
    elif mode == "proxy_nearby_pick":
        # Layer 2 "pick one nearby EPC" branch (proxy_picked: true), mirroring
        # app.py: build the property + recommendation from the chosen cert.
        epc_data = _cached_neighbour_fallback(pc)
        props = epc_data.get("properties") or []
        if not props:
            raise SystemExit(f"{entry['case_id']}: no nearby EPCs for {pc}")
        chosen = props[0]
        source_pc = ((chosen.get("certificate") or {}).get("postcode")
                     or (epc_data.get("proxy_postcodes") or [pc])[0])
        rfq_input = assemble_rfq_input(_empty_epc_data(pc), user_form)
        rfq_input["property"] = build_picked_property_section(
            chosen.get("certificate") or {}, source_pc)
        rec = map_recommendation_section(chosen.get("recommendations") or [])
        rec["recommendation_source"] = "proxy_aggregate"
        rfq_input["recommendation"] = rec
    elif mode == "proxy_same":
        epc_data = _cached_fetch(pc)
        rfq_input = assemble_rfq_input(epc_data, user_form, use_proxy=True)
        prop = rfq_input.get("property") or {}
        if prop.get("epc_source") == "proxy":
            prop["proxy_postcodes"] = [pc]
    else:  # epc
        epc_data = _cached_fetch(pc)
        # Optional: pin a specific certificate by EPC rating (e.g. A / F / G).
        lmk = entry.get("lmk_key")
        if lmk is None and entry.get("select_rating"):
            lmk = _select_lmk_by_rating(epc_data, entry["select_rating"])
        try:
            rfq_input = assemble_rfq_input(epc_data, user_form, lmk_key=lmk)
        except AmbiguousAddress as e:
            lmk = e.candidates[0]["lmk_key"]
            rfq_input = assemble_rfq_input(epc_data, user_form, lmk_key=lmk)

    # Attach the real site context (planning + grid). Failure-tolerant.
    try:
        rfq_input["site_context"] = build_site_context(pc)
    except Exception:
        rfq_input["site_context"] = {"planning": {}, "grid": None, "data_sources": []}

    # Indicative cost band, same derivation the API runs. None for battery and
    # solar thermal, which have no table.
    estimate = build_cost_estimate(rfq_input)
    if estimate:
        rfq_input["cost_estimate"] = estimate

    return {
        "case_id": entry["case_id"],
        "input": rfq_input,
        "output": {},
    }


def _summary_row(entry: dict, case: dict) -> dict:
    rfq = case["input"]
    prop = rfq.get("property") or {}
    rec = rfq.get("recommendation") or {}
    sc = rfq.get("site_context") or {}
    planning = sc.get("planning") or {}
    grid = sc.get("grid") or {}
    comp = completeness_score(rfq)
    flags = []
    if planning.get("listed_building"):
        flags.append("listed")
    if planning.get("conservation_area_name"):
        flags.append("conservation")
    if planning.get("article_4"):
        flags.append("article4")
    if planning.get("aonb_name"):
        flags.append("AONB")
    if planning.get("whs_name"):
        flags.append("WHS")
    if planning.get("national_park_name"):
        flags.append("nat_park")
    if grid:
        flags.append("grid")
    return {
        "case_id": entry["case_id"],
        "postcode": entry["postcode"],
        "technology": entry["technology"],
        "mode": entry["mode"],
        "epc_source": prop.get("epc_source", "none"),
        "epc_rating": prop.get("epc_rating", "n/a"),
        "rec_items": len(rec.get("raw_recommendation_items") or []),
        "site_flags": ",".join(flags) or "none",
        "completeness": comp.get("score"),
        "missing": _missing_required(rfq),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build real-postcode evaluation cases.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    cases = []
    rows = []
    for entry in CONFIG:
        print(f"  building {entry['case_id']} ({entry['postcode']}, "
              f"{entry['technology']}, {entry['mode']}) ...", flush=True)
        try:
            case = build_case(entry)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append({
                "case_id": entry["case_id"], "postcode": entry["postcode"],
                "technology": entry["technology"], "mode": entry["mode"],
                "epc_source": "ERROR", "epc_rating": "-", "rec_items": 0,
                "site_flags": "-", "completeness": 0.0, "missing": [str(e)],
            })
            continue
        cases.append(case)
        rows.append(_summary_row(entry, case))

    out_path = Path(args.out)
    out_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False))

    # Summary table.
    print("\n" + "=" * 100)
    hdr = (f"{'case_id':<10} {'postcode':<9} {'technology':<13} {'mode':<13} "
           f"{'epc_source':<13} {'rating':<7} {'recs':<5} {'site_flags':<22} {'compl':<6}")
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r['case_id']:<10} {r['postcode']:<9} {r['technology']:<13} "
              f"{r['mode']:<13} {str(r['epc_source']):<13} {str(r['epc_rating']):<7} "
              f"{r['rec_items']:<5} {r['site_flags']:<22} {str(r['completeness']):<6}")
    print("=" * 100)

    incomplete = [r for r in rows if (r["completeness"] or 0) < 1.0]
    if incomplete:
        print("\nCases below completeness 1.0 (review / swap postcode):")
        for r in incomplete:
            print(f"  {r['case_id']} ({r['postcode']}): score={r['completeness']} "
                  f"missing={r['missing']}")
    else:
        print("\nAll cases reached completeness 1.0.")

    print(f"\nWrote {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
