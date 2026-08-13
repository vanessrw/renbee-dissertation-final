"""Map EPC API output + user form input -> RFQ schema input dict.

Output shape matches `rfq_cases_real_v1.json[i].input`, ready to feed to generate_rfq.py.

Also declares the field-metadata used for:
- runtime validation (`missing_fields()`)
- Webflow form rendering (label/type/options surface in API responses)
- thesis evaluation completeness scoring (§III.8 of the dissertation)
"""
from __future__ import annotations

import re
import uuid
from datetime import date
from typing import Any, Optional


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class AmbiguousAddress(Exception):
    """Raised when a postcode resolves to multiple distinct addresses.

    `candidates` is a list of dicts: {address, lmk_key, inspection_date}.
    Callers (e.g. the API) should surface these to the user so they can pick
    one, then re-invoke the assembler passing `lmk_key=<chosen>`.
    """

    def __init__(self, candidates: list[dict]):
        self.candidates = candidates
        addrs = [c["address"] for c in candidates]
        super().__init__(
            f"Postcode resolved to {len(candidates)} addresses; "
            f"caller must disambiguate. Addresses: {addrs}"
        )

    @property
    def addresses(self) -> list[str]:
        # Backward-compat shim: old callers expected a list of strings.
        return [c["address"] for c in self.candidates]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

# EPC string values that mean "no data" but aren't empty/None.
_EPC_MISSING_SENTINELS = {"NO DATA!", "not defined", "N/A", "INVALID!", "unknown"}


def _clean(v: Any) -> Any:
    """Coerce EPC missing-data sentinels to None, and strip deprecated code-table
    cruft (e.g. "mains gas - this is for backwards compatibility only and should
    not be used" -> "mains gas"; a bare "To be used only when ... community
    network" placeholder -> None). Pass other values through."""
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s in _EPC_MISSING_SENTINELS:
            return None
        low = s.lower()
        if low.startswith("to be used only when") or "data is from a community network" in low:
            return None
        idx = low.find(" - this is for backwards compatibility")
        if idx != -1:
            return s[:idx].strip() or None
    return v


def _strip_age_band(raw: Optional[str]) -> Optional[str]:
    raw = _clean(raw)
    if not raw:
        return None
    return raw.split(":", 1)[1].strip() if ":" in raw else raw.strip()


def _safe_float(v: Any) -> Optional[float]:
    v = _clean(v)
    try:
        return float(v) if v not in (None, "") else None
    except (ValueError, TypeError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    v = _clean(v)
    try:
        return int(v) if v not in (None, "") else None
    except (ValueError, TypeError):
        return None


def _normalize(s: Optional[str]) -> Optional[str]:
    s = _clean(s)
    if not s:
        return None
    return s.strip().lower().replace(" ", "_")


def _normalize_tenure(s: Optional[str]) -> Optional[str]:
    s = _clean(s)
    if not s:
        return None
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _first_token(address: str) -> str:
    """Return the leading token of an EPC address.

    EPC addresses come in forms like "6, Brushfield Street" or "1 Old Penns Yard".
    The first token is the house/flat number (or letter-suffixed number like "1A").
    """
    if not address:
        return ""
    head = address.split(",", 1)[0].strip()
    parts = head.split(None, 1)
    return parts[0].strip() if parts else ""


def _matches_house_number(address: str, house_number: str) -> bool:
    """Exact match on the first token of the address (case-insensitive).

    Avoids the substring-match trap where '1' would match '10', '21', '1A', etc.
    """
    if not address or not house_number:
        return False
    return _first_token(address).lower() == house_number.strip().lower()


def _candidate_summary(p: dict) -> dict:
    """Compact summary of an EPC property record for disambiguation responses."""
    cert = p.get("certificate", {})
    return {
        "address": cert.get("address", ""),
        "lmk_key": cert.get("lmk-key", ""),
        "inspection_date": cert.get("inspection-date", ""),
    }


# --------------------------------------------------------------------------
# Field metadata (used by missing_fields() and for Webflow rendering)
#
# Each entry: { label, type, options?, required: bool, prompt_if_empty?: bool }
# `required` drives whether absence blocks the LLM call.
# `prompt_if_empty` offers an optional field in the form without gating on it.
# Order of declaration is the order Webflow should render the fields.
# --------------------------------------------------------------------------

_CONTACT_METHODS = ["email", "phone"]
CONTACT_DETAIL_FIELDS = ("contact_email", "contact_phone")
_TIMELINES = ["asap", "within_3_months", "within_6_months", "within_12_months", "flexible"]
_PROPERTY_TYPES = ["detached", "semi-detached", "terraced", "flat", "bungalow", "maisonette"]
_BUILT_FORMS = ["detached", "semi-detached", "mid-terrace", "end-terrace", "enclosed_mid-terrace", "enclosed_end-terrace"]
_AGE_BANDS = ["before 1900", "1900-1929", "1930-1949", "1950-1966", "1967-1975", "1976-1982", "1983-1990", "1991-1995", "1996-2002", "2003-2006", "2007 onwards"]
_TENURES = ["owner_occupied", "rented_private", "rented_social"]
_YES_NO_UNKNOWN = ["yes", "no", "unknown"]
# No "unknown": these are things a homeowner can see for themselves.
_YES_NO = ["yes", "no"]
_INSULATION = ["poor", "moderate", "good", "unknown"]
# A sub-type, not a separate technology: MIS 3005-I p.248 folds solar assisted
# heat pumps into air source.
_HEAT_PUMP_TYPES = ["ground_air_source", "solar_assisted"]
# MIS 3005-I §3.6.7 sends pitched-roof absorbers to MIS 3001, §3.6.8 covers the rest.
_ABSORBER_MOUNTINGS = ["pitched_roof", "flat_roof", "wall", "ground_mounted", "unsure"]
_EMITTER_TYPES = ["radiators", "underfloor_heating", "mixed", "unknown"]
_ROOF_TYPES = ["pitched", "flat", "mixed"]
_ROOF_ORIENTATIONS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest", "unknown"]
_ROOF_PITCHES = ["flat", "shallow", "moderate", "steep", "unknown"]
_ROOF_SHADING = ["none", "low", "moderate", "high"]
_DAYTIME_USAGE = ["low", "medium", "high"]
_HW_HEATING_TYPES = [
    "combi_boiler", "system_boiler_with_cylinder",
    "heat_only_boiler_with_cylinder", "electric_immersion",
    "heat_pump", "unsure",
]
_BATTERY_LOCATIONS = [
    "garage", "utility_room", "outdoor_protected", "loft",
    "inside_living_space", "unsure",
]
_BATTERY_CAPACITIES = ["small_under_5kwh", "medium_5_to_10kwh", "large_over_10kwh", "unsure"]
_BATTERY_PURPOSES = ["backup_power", "self_consumption", "both", "unsure"]
_PV_STATUS = ["existing", "planned", "none", "unknown"]
_COLLECTOR_TYPES = ["flat_plate", "evacuated_tube", "unsure"]
_AUX_HEATING = ["existing_boiler", "immersion_only", "new_system", "unsure"]
_HEATING_SYSTEMS = [
    "gas_boiler_combi", "gas_boiler_system", "gas_boiler_heat_only",
    "oil_boiler", "lpg_boiler",
    "electric_storage_heaters", "electric_panel_or_direct",
    "air_source_heat_pump", "ground_source_heat_pump",
    "solid_fuel", "district_heating", "other_or_unsure",
]
_FUEL_TYPES = [
    "mains_gas", "heating_oil", "lpg", "electricity",
    "biomass_or_wood", "coal_or_solid_fuel", "district_heating",
    "other_or_unsure",
]


FIELDS: dict[str, dict[str, dict]] = {
    "common": {
        "technology_requested": {
            "label": "What would you like to install?",
            "type": "select",
            "options": ["heat_pump", "solar_pv", "battery", "solar_thermal"],
            "required": True,
        },
        "preferred_contact_method": {
            "label": "How would you prefer to be contacted?",
            "type": "select",
            "options": _CONTACT_METHODS,
            "required": True,
        },
        "contact_email": {
            "label": "Your email address",
            "type": "email",
            "options": None,
            "required": True,
        },
        "contact_phone": {
            "label": "Your phone number",
            "type": "tel",
            "options": None,
            "required": True,
        },
        "desired_installation_timeline": {
            "label": "When would you like the installation?",
            "type": "select",
            "options": _TIMELINES,
            "required": True,
        },
        "motivation": {
            "label": "What is your main reason for this enquiry?",
            "type": "textarea",
            "options": None,
            "required": True,
        },
        "budget_range": {
            "label": "What is your approximate budget? (optional)",
            "type": "text",
            "options": None,
            "required": False,
        },
        "additional_notes": {
            "label": "Anything else we should know? (optional)",
            "type": "textarea",
            "options": None,
            "required": False,
        },
    },
    "property": {
        "property_type": {
            "label": "What type of property is it?",
            "type": "select",
            "options": _PROPERTY_TYPES,
            "required": True,
        },
        "built_form": {
            "label": "What is the built form?",
            "type": "select",
            "options": _BUILT_FORMS,
            "required": False,
        },
        # Optional; auto-filled from the EPC, so only ever offered on Case B.
        "floor_area_m2": {
            "label": "Approximate floor area in m² (optional)",
            "type": "number",
            "options": None,
            "required": False,
            "prompt_if_empty": True,
        },
        "construction_age_band": {
            "label": "Approximate construction age band",
            "type": "select",
            "options": _AGE_BANDS,
            "required": False,
        },
        "current_heating_system": {
            "label": "Current heating system",
            "type": "select",
            "options": _HEATING_SYSTEMS,
            "required": True,
        },
        "current_fuel_type": {
            "label": "Current fuel type",
            "type": "select",
            "options": _FUEL_TYPES,
            "required": True,
        },
        "hot_water_system": {
            "label": "Hot water system (e.g. combi, cylinder)",
            "type": "text",
            "options": None,
            "required": False,
        },
        "walls_description": {
            "label": "Wall construction (e.g. cavity insulated, solid brick)",
            "type": "text",
            "options": None,
            "required": False,
        },
        "roof_description": {
            "label": "Roof construction and insulation",
            "type": "text",
            "options": None,
            "required": False,
        },
        "windows_description": {
            "label": "Window glazing",
            "type": "text",
            "options": None,
            "required": False,
        },
        "occupancy_status": {
            "label": "Occupancy status",
            "type": "select",
            "options": _TENURES,
            "required": False,
        },
        "access_constraints": {
            "label": "Any access constraints? (e.g. narrow side passage)",
            "type": "textarea",
            "options": None,
            "required": False,
        },
    },
    "heat_pump": {
        # First: ground source needs land, air source needs external unit space.
        "heat_pump_type_interest": {
            "label": "Which kind of heat pump?",
            "type": "select",
            "options": _HEAT_PUMP_TYPES,
            "required": True,
        },
        "emitter_type": {
            "label": "What kind of heat emitters do you have?",
            "type": "select",
            "options": _EMITTER_TYPES,
            "required": True,
        },
        "hot_water_cylinder_space_available": {
            "label": "Is there space indoors for a hot water cylinder?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": True,
        },
        "external_unit_space": {
            "label": "Is there space outside for the heat pump unit?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": True,
        },
        "garden_or_side_access": {
            "label": "Is there garden or side access for installation?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": True,
        },
        # Heat load and cylinder sizing follow property scale and household
        # hot-water demand (the same reason solar thermal asks). Sizing itself
        # is in MIS 3005-D, the Design standard, not MIS 3005-I.
        "number_of_bedrooms": {
            "label": "How many bedrooms does the property have?",
            "type": "number",
            "options": None,
            "required": True,
        },
        "number_of_bathrooms": {
            "label": "How many bathrooms does the property have?",
            "type": "number",
            "options": None,
            "required": True,
        },
        "number_of_occupants": {
            "label": "How many people live in the property?",
            "type": "number",
            "options": None,
            "required": True,
        },
        # Supply capacity feeds the DNO notification required by MIS 3005-I §3.1.7.
        "smart_meter_installed": {
            "label": "Do you have a smart meter?",
            "type": "select",
            "options": _YES_NO,
            "required": True,
        },
        "smart_meter_cutout_fuse_label": {
            "label": "What does the label on your main cutout fuse say? (optional)",
            "type": "text",
            "options": None,
            "required": False,
            "prompt_if_empty": True,
        },
        # Optional: solar_assisted only. FIELDS has no conditional-required
        # mechanism, so requiring it would ask ground-source enquiries too.
        "absorber_mounting_location": {
            "label": "Where would the solar absorber panels be mounted?",
            "type": "select",
            "options": _ABSORBER_MOUNTINGS,
            "required": False,
        },
        "radiator_suitability_known": {
            "label": "Do you know if your radiators are suitable for a heat pump?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": False,
        },
        "insulation_status_basic": {
            "label": "How would you describe your insulation?",
            "type": "select",
            "options": _INSULATION,
            "required": False,
        },
        "boiler_age_years": {
            "label": "How old is your current boiler? (years, optional)",
            "type": "number",
            "options": None,
            "required": False,
        },
        "noise_or_planning_constraints": {
            "label": "Any noise or planning constraints? (optional)",
            "type": "textarea",
            "options": None,
            "required": False,
        },
        "bus_interest": {
            "label": "Are you interested in the Boiler Upgrade Scheme grant?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": False,
        },
        "cooling_interest": {
            "label": "Are you also interested in cooling (reversible heat pump)?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": False,
        },
    },
    "solar_pv": {
        "roof_orientation": {
            "label": "Which direction does your main roof face?",
            "type": "select",
            "options": _ROOF_ORIENTATIONS,
            "required": True,
        },
        "usable_roof_area_m2": {
            "label": "Approximate usable roof area in m² (optional)",
            "type": "number",
            "options": None,
            "required": False,
            "prompt_if_empty": True,
        },
        "roof_shading_level": {
            "label": "How shaded is your roof?",
            "type": "select",
            "options": _ROOF_SHADING,
            "required": True,
        },
        # A roof needing work before panels go on changes the quote, so ask.
        "roof_condition": {
            "label": "Is your roof in good condition?",
            "type": "select",
            "options": _YES_NO,
            "required": True,
        },
        # Bedrooms key the indicative cost table in cost_tables.py. Bathrooms
        # and occupants are asked for parity with the heat pump form.
        "number_of_bedrooms": {
            "label": "How many bedrooms does the property have?",
            "type": "number",
            "options": None,
            "required": True,
        },
        "number_of_bathrooms": {
            "label": "How many bathrooms does the property have?",
            "type": "number",
            "options": None,
            "required": True,
        },
        "number_of_occupants": {
            "label": "How many people live in the property?",
            "type": "number",
            "options": None,
            "required": True,
        },
        "roof_type": {
            "label": "Roof type",
            "type": "select",
            "options": _ROOF_TYPES,
            "required": False,
        },
        "roof_pitch": {
            "label": "Roof pitch",
            "type": "select",
            "options": _ROOF_PITCHES,
            "required": False,
        },
        "number_of_roof_faces": {
            "label": "How many roof faces could be used?",
            "type": "number",
            "options": None,
            "required": False,
        },
        # Monthly, not annual: it is the figure a homeowner actually knows.
        "monthly_electricity_bill_gbp": {
            "label": "Approximate monthly electricity cost in GBP (optional)",
            "type": "number",
            "options": None,
            "required": False,
            "prompt_if_empty": True,
        },
        "daytime_electricity_usage": {
            "label": "Daytime electricity usage",
            "type": "select",
            "options": _DAYTIME_USAGE,
            "required": False,
        },
        "battery_interest": {
            "label": "Are you interested in battery storage?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": False,
        },
        "ev_present_or_planned": {
            "label": "Do you have or plan to get an electric vehicle?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": False,
        },
        "roof_under_warranty": {
            "label": "Is your roof currently under warranty?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": False,
        },
    },
    "battery": {
        "existing_solar_pv": {
            "label": "Do you already have or are planning solar PV?",
            "type": "select",
            "options": _PV_STATUS,
            "required": True,
        },
        "battery_purpose": {
            "label": "What is the main purpose of the battery?",
            "type": "select",
            "options": _BATTERY_PURPOSES,
            "required": True,
        },
        "backup_power_required": {
            "label": "Do you need backup power during grid outages?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": True,
        },
        # Precondition for the location question below.
        "battery_space_available": {
            "label": "Is there space for the battery (for example in a garage)?",
            "type": "select",
            "options": _YES_NO,
            "required": True,
        },
        "battery_location_preference": {
            "label": "Where would you prefer the battery to be installed?",
            "type": "select",
            "options": _BATTERY_LOCATIONS,
            "required": True,
        },
        "existing_solar_pv_capacity_kwp": {
            "label": "If you have solar PV, what is its capacity in kWp? (optional)",
            "type": "number",
            "options": None,
            "required": False,
        },
        "desired_capacity_band": {
            "label": "Roughly what battery capacity are you considering?",
            "type": "select",
            "options": _BATTERY_CAPACITIES,
            "required": False,
        },
        "monthly_electricity_bill_gbp": {
            "label": "Approximate monthly electricity cost in GBP (optional)",
            "type": "number",
            "options": None,
            "required": False,
            "prompt_if_empty": True,
        },
        "daytime_electricity_usage": {
            "label": "Daytime electricity usage",
            "type": "select",
            "options": _DAYTIME_USAGE,
            "required": False,
        },
        "ev_present_or_planned": {
            "label": "Do you have or plan to get an electric vehicle?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": False,
        },
        "time_of_use_tariff": {
            "label": "Are you on (or planning to switch to) a time-of-use tariff?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": False,
        },
    },
    "solar_thermal": {
        "roof_orientation": {
            "label": "Which direction does your main roof face?",
            "type": "select",
            "options": _ROOF_ORIENTATIONS,
            "required": True,
        },
        "usable_roof_area_m2": {
            "label": "Approximate usable roof area in m² (optional)",
            "type": "number",
            "options": None,
            "required": False,
            "prompt_if_empty": True,
        },
        "roof_shading_level": {
            "label": "How shaded is your roof?",
            "type": "select",
            "options": _ROOF_SHADING,
            "required": True,
        },
        "hot_water_cylinder_space_available": {
            "label": "Is there space indoors for a hot water cylinder?",
            "type": "select",
            "options": _YES_NO_UNKNOWN,
            "required": True,
        },
        "number_of_occupants": {
            "label": "How many people live in the property?",
            "type": "number",
            "options": None,
            "required": True,
        },
        "number_of_bathrooms": {
            "label": "How many bathrooms does the property have?",
            "type": "number",
            "options": None,
            "required": True,
        },
        "current_water_heating_type": {
            "label": "How is your hot water currently heated?",
            "type": "select",
            "options": _HW_HEATING_TYPES,
            "required": False,
        },
        "existing_cylinder_size_litres": {
            "label": "If you have a hot water cylinder, what is its capacity in litres? (optional)",
            "type": "number",
            "options": None,
            "required": False,
        },
        "roof_type": {
            "label": "Roof type",
            "type": "select",
            "options": _ROOF_TYPES,
            "required": False,
        },
        "roof_pitch": {
            "label": "Roof pitch",
            "type": "select",
            "options": _ROOF_PITCHES,
            "required": False,
        },
        # Same key as solar_pv, so same wording. Optional here: only solar PV asks it.
        "roof_condition": {
            "label": "Is your roof in good condition?",
            "type": "select",
            "options": _YES_NO,
            "required": False,
        },
        "collector_type_preference": {
            "label": "Do you have a preference for collector type?",
            "type": "select",
            "options": _COLLECTOR_TYPES,
            "required": False,
        },
        "auxiliary_heating_intended": {
            "label": "How will hot water be heated when there isn't enough sun?",
            "type": "select",
            "options": _AUX_HEATING,
            "required": False,
        },
    },
}

# Single source of truth for technology → form section name. Add a new
# technology by adding its section dict in FIELDS above and one entry here.
_TECH_SECTIONS: dict[str, str] = {
    "heat_pump": "heat_pump",
    "solar_pv": "solar_pv",
    "battery": "battery",
    "solar_thermal": "solar_thermal",
}


# --------------------------------------------------------------------------
# Address selection
# --------------------------------------------------------------------------

def select_certificate(
    epc_data: dict,
    house_number: Optional[str] = None,
    lmk_key: Optional[str] = None,
) -> Optional[dict]:
    """Pick one property from an EPC postcode lookup.

    Resolution order:
    1. If `lmk_key` is given, return the exact record (used after the user
       has picked from a previously-returned candidate list). Returns None
       if no record matches.
    2. If `house_number` is given, filter records whose address's first token
       matches exactly (so "1" matches "1 Old Penns Yard" but not "10..." or
       "1A...").
    3. Group remaining records by `address`; keep only the latest inspection
       per address.
    4. If exactly one address remains, return it.
    5. If none remain, return None — caller falls back to Case B (user
       manually enters property details).
    6. If multiple distinct addresses remain, raise `AmbiguousAddress` whose
       `candidates` list carries each address's `lmk_key` and inspection
       date for the caller to surface to the user.
    """
    properties = epc_data.get("properties", [])

    # 1. Direct lmk_key resolution
    if lmk_key:
        for p in properties:
            if p.get("certificate", {}).get("lmk-key") == lmk_key:
                return p
        return None

    # 2. House-number filter (exact first-token match)
    if house_number:
        properties = [
            p for p in properties
            if _matches_house_number(
                p.get("certificate", {}).get("address", ""), house_number
            )
        ]

    if not properties:
        return None

    # 3. Group by address, take latest inspection per address
    by_address: dict[str, list] = {}
    for p in properties:
        addr = p.get("certificate", {}).get("address", "")
        by_address.setdefault(addr, []).append(p)

    latest_per_address = [
        max(group, key=lambda p: p["certificate"].get("inspection-date", ""))
        for group in by_address.values()
    ]

    # 4-6. Resolve / fall back / raise with candidate list
    if len(latest_per_address) == 1:
        return latest_per_address[0]
    if not latest_per_address:
        return None

    candidates = [_candidate_summary(p) for p in latest_per_address]
    candidates.sort(key=lambda c: c["address"])
    raise AmbiguousAddress(candidates)


# --------------------------------------------------------------------------
# Field mapping
# --------------------------------------------------------------------------

def map_property_section(cert: dict) -> dict:
    return {
        "epc_found": True,
        "property_type": _normalize(cert.get("property-type")),
        "built_form": _normalize(cert.get("built-form")),
        "floor_area_m2": _safe_float(cert.get("total-floor-area")),
        "construction_age_band": _strip_age_band(cert.get("construction-age-band")),
        "epc_rating": _clean(cert.get("current-energy-rating")),
        "epc_score": _safe_int(cert.get("current-energy-efficiency")),
        "current_heating_system": _clean(cert.get("mainheat-description")),
        "current_fuel_type": _clean(cert.get("main-fuel")),
        "hot_water_system": _clean(cert.get("hotwater-description")),
        "walls_description": _clean(cert.get("walls-description")),
        "roof_description": _clean(cert.get("roof-description")),
        "windows_description": _clean(cert.get("windows-description")),
        "occupancy_status": _normalize_tenure(cert.get("tenure")),
        "access_constraints": None,
    }


def _money(value) -> Optional[int]:
    """'2,700' or '£2,700' -> 2700. None when absent or unparseable.

    Single figures only. EPC costs are often ranges, so use _money_range there.
    """
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None


def _money_range(value) -> Optional[tuple[int, int]]:
    """Parse an EPC cost, which may be a single figure or a range.

    '£4,000 - £14,000' -> (4000, 14000);  '£2,700' -> (2700, 2700)

    Most EPC indicative-cost values are ranges. Stripping non-digits and
    concatenating turns '£4,000 - £14,000' into 400014000, so each figure has
    to be read separately. min/max tolerates '£4,000-£14,000' with no spaces
    and any stray third figure.
    """
    if value is None:
        return None
    nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]*\d", str(value))]
    if not nums:
        return None
    return min(nums), max(nums)


def map_recommendation_section(recommendations: list) -> dict:
    items = []
    details = []
    for r in recommendations or []:
        text = r.get("improvement-summary-text") or r.get("improvement-descr-text")
        if not text:
            continue
        items.append(text)
        cost = _money_range(r.get("indicative-cost"))
        # typical-saving is always a single figure, never a range.
        saving = _money(r.get("typical-saving"))
        entry = {"item": text}
        if cost is not None:
            entry["indicative_cost_low_gbp"] = cost[0]
            entry["indicative_cost_high_gbp"] = cost[1]
        if saving is not None:
            entry["typical_yearly_saving_gbp"] = saving
        details.append(entry)
    return {
        "epc_recommendations_available": bool(items),
        "recommendation_source": "official_epc",
        # Must stay plain strings: faithfulness.preservation_score str()s each.
        "raw_recommendation_items": items,
        "recommendation_details": details,
    }


# --------------------------------------------------------------------------
# Proxy EPC — aggregate neighbouring certificates when the homeowner's own
# property has no EPC (or they prefer a street-level estimate). Approach A
# (probabilistic prior) from the Renbee design discussion.
# --------------------------------------------------------------------------

def _proxy_confidence(n: int) -> str:
    if n >= 5:
        return "high"
    if n >= 2:
        return "medium"
    return "low"


def _mode(certs: list[dict], field: str):
    from collections import Counter
    values = [c.get(field) for c in certs if c.get(field) not in (None, "")]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _median(certs: list[dict], field: str):
    values: list[float] = []
    for c in certs:
        v = c.get(field)
        if v in (None, ""):
            continue
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    values.sort()
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def _latest_field(certs: list[dict], field: str, date_field: str = "inspection-date"):
    """Return the value of `field` from the most recently inspected certificate
    where that field is populated. Useful for EPC rating where the most-recent
    assessment is more representative than a mode."""
    dated = [
        (c.get(date_field) or "", c.get(field))
        for c in certs
        if c.get(field) not in (None, "")
    ]
    if not dated:
        return None
    return max(dated, key=lambda x: x[0])[1]


def build_proxy_property_section(epc_data: dict, overrides: dict | None = None) -> dict | None:
    """Aggregate every certificate in `epc_data` into a single proxy property
    section. Returns None if there are zero certificates (caller falls back to
    Case B). Same shape as `map_property_section()` plus three proxy markers:
      epc_source              — always "proxy"
      proxy_comparator_count  — number of neighbouring certificates aggregated
      proxy_confidence        — low / medium / high
    """
    properties = (epc_data or {}).get("properties") or []
    if not properties:
        return None

    certs = [p.get("certificate") or {} for p in properties]
    overrides = overrides or {}

    section = {
        "epc_found": False,
        "epc_source": "proxy",
        "proxy_comparator_count": len(certs),
        "proxy_confidence": _proxy_confidence(len(certs)),
        "property_type": _normalize(_mode(certs, "property-type")),
        "built_form": _normalize(_mode(certs, "built-form")),
        "floor_area_m2": _median(certs, "total-floor-area"),
        "construction_age_band": _strip_age_band(_mode(certs, "construction-age-band")),
        "epc_rating": _clean(_latest_field(certs, "current-energy-rating")),
        "epc_score": None,
        "current_heating_system": _clean(_mode(certs, "mainheat-description")),
        "current_fuel_type": _clean(_mode(certs, "main-fuel")),
        "hot_water_system": _clean(_mode(certs, "hotwater-description")),
        "walls_description": _clean(_mode(certs, "walls-description")),
        "roof_description": _clean(_mode(certs, "roof-description")),
        "windows_description": _clean(_mode(certs, "windows-description")),
        "occupancy_status": _normalize_tenure(_mode(certs, "tenure")),
        "access_constraints": overrides.get("access_constraints"),
    }
    return section


def build_nearby_candidate_list(epc_data: dict) -> list[dict]:
    """Surface each nearby EPC certificate as a picker candidate.

    Used by the Layer 2 picker UX: instead of silently aggregating nearby
    EPCs, the API hands the homeowner a list of candidates so they can either
    pick the one they feel is closest to their property or opt for the
    aggregate of all of them. The four fields shown per candidate mirror the
    Case-B manual-entry form.
    """
    properties = (epc_data or {}).get("properties") or []
    out: list[dict] = []
    for p in properties:
        cert = p.get("certificate") or {}
        out.append({
            "lmk_key": cert.get("lmk-key", ""),
            "postcode": cert.get("postcode", ""),
            "address": cert.get("address", ""),
            "inspection_date": cert.get("inspection-date", ""),
            "property_type": _normalize(cert.get("property-type")),
            "floor_area_m2": _safe_float(cert.get("total-floor-area")),
            "current_heating_system": _clean(cert.get("mainheat-description")),
            "current_fuel_type": _clean(cert.get("main-fuel")),
        })
    return out


def build_picked_property_section(certificate: dict, source_postcode: str) -> dict:
    """Build a single-EPC property section overlaid with proxy markers.

    Used when the homeowner picks one of the nearby EPCs as "closest match".
    The data comes from a real EPC (so all fields are concrete, not aggregated)
    but it's not the homeowner's own home, so we keep the proxy markers on the
    section to make the LLM prompts soften their phrasing accordingly.
    """
    section = map_property_section(certificate)
    section["epc_found"] = False
    section["epc_source"] = "proxy_nearby"
    section["proxy_picked"] = True
    section["proxy_comparator_count"] = 1
    section["proxy_confidence"] = _proxy_confidence(1)
    section["proxy_postcodes"] = [source_postcode] if source_postcode else []
    return section


def build_proxy_recommendation_section(epc_data: dict) -> dict:
    """Aggregate the most-common improvement recommendations across all
    certificates in `epc_data`. Top 5 most-cited items are surfaced — the
    LLM prompt is constrained to use only these, same as the EPC-found path,
    but with `recommendation_source = "proxy_aggregate"` so the prose can
    soften ("based on similar properties on your street").
    """
    from collections import Counter
    properties = (epc_data or {}).get("properties") or []

    all_items: list[str] = []
    for p in properties:
        for r in p.get("recommendations") or []:
            text = r.get("improvement-summary-text") or r.get("improvement-descr-text")
            if text:
                all_items.append(text)

    if not all_items:
        return {
            "epc_recommendations_available": False,
            "recommendation_source": "proxy_aggregate",
            "raw_recommendation_items": [],
        }

    most_common = [text for text, _ in Counter(all_items).most_common(5)]
    return {
        "epc_recommendations_available": True,
        "recommendation_source": "proxy_aggregate",
        "raw_recommendation_items": most_common,
    }


# Named designations that restrict Permitted Development rights. article_4 and
# listed_building are booleans, so they are handled separately below.
_PD_RESTRICTING_DESIGNATIONS = (
    ("conservation_area_name", "conservation area"),
    ("aonb_name", "Area of Outstanding Natural Beauty"),
    ("whs_name", "World Heritage Site"),
    ("national_park_name", "National Park"),
)


def derive_planning_consequences(planning: dict) -> dict:
    """Spell out what a planning designation implies, so the claim is carried by
    the data and not only by the prompt.

    The source data gives a conservation area name but nothing about what it
    means, while the prompts instruct the model to state that Permitted
    Development rights are restricted. The fabrication judge sees only the input,
    so it flags a claim the prompt demanded.

    Returns {} when no designation applies. Keys are omitted rather than set to
    False, because `_prune_planning_flags` strips falsy planning values before
    the model sees them and the prompts forbid stating a constraint's absence.
    """
    if not planning:
        return {}

    basis: list[str] = []
    for key, label in _PD_RESTRICTING_DESIGNATIONS:
        if planning.get(key):
            basis.append(f"{label}: {planning[key]}")
    if planning.get("article_4"):
        basis.append("Article 4 direction in force")
    if planning.get("listed_building"):
        grade = planning.get("listed_grade")
        basis.append(f"listed building (Grade {grade})" if grade else "listed building")

    if not basis:
        return {}
    # One key per claim the prompts make, in their own words so the judge can
    # match claim to field.
    return {
        "permitted_development_restricted": True,       # "PD rights are restricted"
        "planning_permission_likely_required": True,    # "will likely need planning permission"
        "consent_may_be_required": True,                # "confirm what consent applies"
        "planning_consequence_basis": "; ".join(basis),
    }


def build_site_context(postcode: str) -> dict:
    """Cross-check the postcode against external open-data sources and
    return a single `site_context` section for the RFQ.

    Composition:
      planning: dict       — listed building, conservation, Article 4, AONB,
                             WHS, National Park (from planning.data.gov.uk),
                             plus the derived consequence keys added by
                             `derive_planning_consequences`
      grid: dict | None    — UKPN headroom for nearest PRIMARY substation
                             (None when outside UKPN territory or no API key)
      data_sources: list   — credits surfaced on the demo page (CC BY 4.0)

    Failure-tolerant — if either upstream fails, the corresponding section
    is empty/None and the caller keeps going.
    """
    from external_data import fetch_planning_constraints, fetch_ukpn_constraints

    planning = fetch_planning_constraints(postcode)
    if planning:
        # Carry the legal consequence in the data, not just in the prompt.
        planning = {**planning, **derive_planning_consequences(planning)}
    grid = fetch_ukpn_constraints(postcode)

    sources: list[str] = []
    if planning:
        sources.append("planning.data.gov.uk")
    if grid:
        sources.append("UK Power Networks Open Data")

    return {
        "planning": planning,
        "grid": grid,
        "data_sources": sources,
    }


def assemble_rfq_input(
    epc_data: dict,
    user_form: dict,
    house_number: Optional[str] = None,
    enquiry_id: Optional[str] = None,
    lmk_key: Optional[str] = None,
    use_proxy: bool = False,
) -> dict:
    """Build full RFQ input dict matching `rfq_cases_real_v1.json[i].input`.

    user_form keys (all optional except `common.technology_requested`):
      common: dict — preferred_contact_method, desired_installation_timeline,
              budget_range, motivation, additional_notes, technology_requested
      heat_pump: dict — used when technology_requested == "heat_pump"
      solar_pv: dict — used when technology_requested == "solar_pv"
      property_overrides: dict — manual property entries used when EPC not found
              (Case B fallback; also fills `access_constraints` when EPC found)
    """
    overrides = user_form.get("property_overrides") or {}

    property_section: Optional[dict] = None
    recommendation_section: Optional[dict] = None

    if use_proxy:
        # Skip address selection; aggregate every certificate in the postcode
        # into a proxy property + recommendation section. If no certificates
        # exist at all, fall through to the Case B manual-entry shape below.
        proxy_property = build_proxy_property_section(epc_data, overrides)
        if proxy_property is not None:
            property_section = proxy_property
            recommendation_section = build_proxy_recommendation_section(epc_data)
    else:
        chosen = select_certificate(
            epc_data, house_number=house_number, lmk_key=lmk_key
        )
        if chosen:
            property_section = map_property_section(chosen["certificate"])
            recommendation_section = map_recommendation_section(
                chosen.get("recommendations", [])
            )
            if overrides.get("access_constraints"):
                property_section["access_constraints"] = overrides["access_constraints"]

    if property_section is None:
        property_section = {
            "epc_found": False,
            "property_type": overrides.get("property_type"),
            "built_form": overrides.get("built_form"),
            "floor_area_m2": overrides.get("floor_area_m2"),
            "construction_age_band": overrides.get("construction_age_band"),
            "epc_rating": None,
            "epc_score": None,
            "current_heating_system": overrides.get("current_heating_system"),
            "current_fuel_type": overrides.get("current_fuel_type"),
            "hot_water_system": overrides.get("hot_water_system"),
            "walls_description": overrides.get("walls_description"),
            "roof_description": overrides.get("roof_description"),
            "windows_description": overrides.get("windows_description"),
            "occupancy_status": overrides.get("occupancy_status"),
            "access_constraints": overrides.get("access_constraints"),
        }
        recommendation_section = {
            "epc_recommendations_available": False,
            "recommendation_source": "official_epc",
            "raw_recommendation_items": [],
        }

    common = dict(user_form.get("common", {}))
    common.setdefault("enquiry_id", enquiry_id or f"RFQ_{uuid.uuid4().hex[:8].upper()}")
    common.setdefault("submission_date", date.today().isoformat())
    common.setdefault("postcode", epc_data.get("postcode"))

    result = {
        "common": common,
        "property": property_section,
        "recommendation": recommendation_section,
    }
    for section_name in _TECH_SECTIONS.values():
        result[section_name] = user_form.get(section_name, {})
    return result


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def missing_fields(rfq_input: dict) -> dict:
    """Return required fields that are missing from the assembled RFQ input.

    Walks `common`, `property`, and the technology-specific section (chosen
    by `common.technology_requested`). For each section, checks every field
    declared `required` in `FIELDS` and reports those whose value is missing.

    Return shape (ready to drop into an API response):
      {
        "common":   [{name, label, type, options}, ...],
        "property": [...],
        "<technology>": [...]   # heat_pump / solar_pv / battery / solar_thermal
      }

    Empty sections are omitted. If nothing is missing, returns `{}`.
    """
    technology = (rfq_input.get("common") or {}).get("technology_requested")

    sections = ["common", "property"]
    if technology in _TECH_SECTIONS:
        sections.append(_TECH_SECTIONS[technology])

    result: dict[str, list[dict]] = {}
    for section_name in sections:
        section_data = rfq_input.get(section_name) or {}
        section_fields = FIELDS.get(section_name, {})
        missing: list[dict] = []
        for field_name, meta in section_fields.items():
            if not meta.get("required"):
                continue
            if _is_empty(section_data.get(field_name)):
                missing.append({
                    "name": field_name,
                    "label": meta["label"],
                    "type": meta["type"],
                    "options": meta.get("options"),
                })
        if missing:
            result[section_name] = missing

    return result


def optional_fields(rfq_input: dict) -> dict:
    """Optional fields worth offering the homeowner anyway, same shape as
    `missing_fields()`.

    Only fields flagged `prompt_if_empty` are returned, so the form stays
    minimal. Kept separate from `missing_fields()` because that result gates
    the LLM call — a blank here must never block.
    """
    technology = (rfq_input.get("common") or {}).get("technology_requested")

    sections = ["common", "property"]
    if technology in _TECH_SECTIONS:
        sections.append(_TECH_SECTIONS[technology])

    result: dict[str, list[dict]] = {}
    for section_name in sections:
        section_data = rfq_input.get(section_name) or {}
        offer: list[dict] = []
        for field_name, meta in FIELDS.get(section_name, {}).items():
            if meta.get("required") or not meta.get("prompt_if_empty"):
                continue
            if _is_empty(section_data.get(field_name)):
                offer.append({
                    "name": field_name,
                    "label": meta["label"],
                    "type": meta["type"],
                    "options": meta.get("options"),
                })
        if offer:
            result[section_name] = offer

    return result


def completeness_score(rfq_input: dict) -> dict:
    """Return a §III.8-style completeness score.

    Counts required fields across applicable sections, computes the fraction
    that are populated. Useful both as a runtime diagnostic and as the
    quantitative metric for the thesis evaluation.
    """
    technology = (rfq_input.get("common") or {}).get("technology_requested")
    sections = ["common", "property"]
    if technology in _TECH_SECTIONS:
        sections.append(_TECH_SECTIONS[technology])

    required_total = 0
    populated = 0
    per_section: dict[str, dict[str, int]] = {}

    for section_name in sections:
        section_data = rfq_input.get(section_name) or {}
        section_fields = FIELDS.get(section_name, {})
        sec_required = 0
        sec_populated = 0
        for field_name, meta in section_fields.items():
            if not meta.get("required"):
                continue
            sec_required += 1
            if not _is_empty(section_data.get(field_name)):
                sec_populated += 1
        per_section[section_name] = {
            "required": sec_required,
            "populated": sec_populated,
        }
        required_total += sec_required
        populated += sec_populated

    return {
        "score": (populated / required_total) if required_total else 1.0,
        "populated": populated,
        "required": required_total,
        "per_section": per_section,
    }
