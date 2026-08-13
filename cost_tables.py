"""Renbee indicative cost tables (solar PV and heat pump only).

Two static lookup tables from Renbee's homeowner cost guidance, transcribed
verbatim. Every figure the recommendation prose quotes is copied from here,
never computed — the same contract as the EPC cost figures produced by
`map_recommendation_section` in epc_to_rfq.py.

These are commercial guide prices. They are not MCS figures, not EPC figures,
and not derived from the homeowner's own property, so the recommendation
disclaimer names them as a separate source.

Deliberately imports nothing from the project, so it can be pulled into app.py,
generate_rfq.py, evaluation/build_cases.py and the tests without touching the
epc_to_rfq import graph.
"""
from __future__ import annotations

from typing import Any, Optional

SOURCE = "Renbee indicative cost tables"

# The prompt requires the cost block to close with this caveat. It is carried as
# data for the same reason derive_planning_consequences exists: the fabrication
# judge receives rfq_input as the only facts that are true, so a caveat demanded
# by the prompt but absent from the input is scored as unsupported. In the first
# scored run this single sentence accounted for 41 of 51 fabrication flags.
GUIDE_PRICE_NOTE = (
    "These are typical guide prices for a comparable property, not a quotation. "
    "The installer's survey confirms the actual cost."
)

# Boiler Upgrade Scheme. Carried so the prose can name the grant. The net
# ranges below are transcribed from the table, not derived from this.
BUS_GRANT_GBP = 7500

# --------------------------------------------------------------------------
# Table 1 — "Solar Panel Costs by System Size"
# Rows ascend by system size; _match relies on that for its final tie-break.
# --------------------------------------------------------------------------
SOLAR_PV_ROWS: list[dict[str, Any]] = [
    {"band": "1-2 bed flat/terrace", "beds": (1, 2), "styles": {"flat", "terrace"},
     "kw": 3, "panels": (7, 8), "cost": (5000, 6500), "saving": (350, 550)},
    {"band": "3 bed semi-detached", "beds": (3, 3), "styles": {"semi"},
     "kw": 4, "panels": (10, 10), "cost": (6000, 8000), "saving": (500, 800)},
    {"band": "3-4 bed detached", "beds": (3, 4), "styles": {"detached"},
     "kw": 5, "panels": (12, 13), "cost": (7000, 9500), "saving": (650, 950)},
    {"band": "4-5 bed detached", "beds": (4, 5), "styles": {"detached"},
     "kw": 6, "panels": (15, 15), "cost": (8000, 11000), "saving": (800, 1200)},
]

# --------------------------------------------------------------------------
# Table 2 — "Quick Cost Summary by Property Size"
# --------------------------------------------------------------------------
HEAT_PUMP_ROWS: list[dict[str, Any]] = [
    {"band": "1-2 bed flat/terrace", "beds": (1, 2), "styles": {"flat", "terrace"},
     "ashp": (8000, 10000), "gshp": (18000, 22000), "net": (500, 2500)},
    {"band": "3-4 bed detached", "beds": (3, 4), "styles": {"detached"},
     "ashp": (11000, 15000), "gshp": (22000, 30000), "net": (3500, 7500)},
]

# The third row is qualitative, not bedroom-keyed, so it is matched by
# `is_large_period` rather than by the ranking function.
HEAT_PUMP_LARGE_ROW: dict[str, Any] = {
    "band": "Large period property",
    "ashp": (14000, 18000), "gshp": (28000, 35000), "net": (6500, 10500),
}

TABLE_NAMES = {
    "solar_pv": "Solar Panel Costs by System Size",
    "heat_pump": "Quick Cost Summary by Property Size",
}

# Solid-wall fabric is the mechanism the "Large period property" row prices.
# Cavity construction becomes normal from roughly 1920, so pre-1919 is the
# meaningful cut; the band vocabulary has no 1919 split, making 1900-1929 the
# closest available boundary. 1930-1949 is cavity-era and must not count.
_PERIOD_AGE_BANDS = frozenset({"before 1900", "1900-1929"})
# The row above tops out at a typical 4-bed detached (~120-140 m²).
_LARGE_FLOOR_AREA_M2 = 150.0
_LARGE_BEDROOMS = 5


def _flatten(value: Optional[str]) -> str:
    """Lowercase and collapse EPC's inconsistent separators to spaces.

    The cache holds both `enclosed-mid-terrace_house` and
    `enclosed_end-terrace_house`, so neither '-' nor '_' can be trusted as the
    token separator. Also normalises the Case B manual-form vocabulary
    (`semi-detached`, `terraced`) onto the same surface.
    """
    return " ".join(str(value or "").lower().replace("-", " ").replace("_", " ").split())


def classify_style(property_type, built_form) -> Optional[str]:
    """'flat' | 'terrace' | 'semi' | 'detached', or None when undeterminable.

    `property_type` is authoritative for the flat/house split; `built_form` is
    consulted only when `property_type` names no form at all.

    That ordering is load-bearing. In EPC data `built-form` describes the
    building, not the dwelling, so 160 of the 355 flat and maisonette
    certificates in output/ carry a non-terrace built form. Leading with
    built_form prices a mid-floor flat in a detached block as a detached house.
    """
    kind = _flatten(property_type)
    if "flat" in kind or "maisonette" in kind:
        return "flat"
    for source in (kind, _flatten(built_form)):
        if not source or source == "not recorded":
            continue
        if "terrace" in source:            # mid / end / enclosed / terraced
            return "terrace"
        if "semi detached" in source:      # must precede the bare detached test
            return "semi"
        if "detached" in source:
            return "detached"
    return None


def _coerce_int(value) -> Optional[int]:
    """Accept 3 and "3" alike.

    The demo builds additional_fields from FormData, so every answer arrives as
    a string, while the tests and evaluation cases pass real ints. Comparing a
    str numerically would 500 the request.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _coerce_float(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _rank(row: dict, beds: int, style: Optional[str], order: int):
    """Sort key for band matching. Lower is better, every component decided.

    Bedrooms lead style deliberately: bedrooms are the homeowner's own answer,
    while style is derived from 15+ EPC property_type spellings. Letting the
    lower-confidence signal dominate would let a vocabulary miss move the price.
    """
    low, high = row["beds"]
    bed_distance = 0 if low <= beds <= high else min(abs(beds - low), abs(beds - high))
    style_penalty = 0 if (style is not None and style in row["styles"]) else 1
    midpoint_gap = abs(beds - (low + high) / 2)
    return (bed_distance, style_penalty, midpoint_gap, order)


def _match(rows: list[dict], beds: int, style: Optional[str]) -> tuple[dict, bool]:
    """Pick the best row and report whether it is an exact match.

    `order` is the final tie-break and rows ascend by size, so a genuine tie
    resolves to the cheaper system. It is needed rather than decorative: a
    4-bed detached matches both "3-4 bed detached" and "4-5 bed detached", and
    without it the result would silently depend on list order.
    """
    index = min(range(len(rows)), key=lambda i: _rank(rows[i], beds, style, i))
    row = rows[index]
    low, high = row["beds"]
    exact = low <= beds <= high and style is not None and style in row["styles"]
    return row, exact


def is_large_period(style, beds, floor_area_m2, construction_age_band) -> bool:
    """Large AND period, both required, because that is what the row says.

    Period is a membership test, never a parsed year: the vocabulary contains
    "before 1900" and "2012 onwards", and out-of-vocabulary bands such as
    "2007-2011" occur, so an unknown band fails closed.

    A flat is never a large period property — it is a dwelling inside one.
    Without that guard a maisonette whose EPC recorded the whole building's
    floor area prices as a mansion.
    """
    if style == "flat":
        return False
    if (construction_age_band or "").strip().lower() not in _PERIOD_AGE_BANDS:
        return False
    if beds is not None and beds >= _LARGE_BEDROOMS:
        return True
    return floor_area_m2 is not None and floor_area_m2 >= _LARGE_FLOOR_AREA_M2


def _describe_basis(beds: int, style: Optional[str]) -> str:
    label = {
        "flat": "flat or maisonette",
        "terrace": "terraced",
        "semi": "semi-detached",
        "detached": "detached",
    }.get(style)
    bedrooms = f"{beds} bedroom{'s' if beds != 1 else ''}"
    return f"{bedrooms}; {label}" if label else f"{bedrooms}; property style not known"


def build_cost_estimate(rfq_input: dict) -> Optional[dict]:
    """Derived `cost_estimate` section, or None when no band can be matched.

    None rather than a section of nulls is the contract: the prompt is told to
    say nothing about technology cost when the section is absent, so a
    half-populated section would be worse than no section.

    Returns None when the technology has no table (battery, solar thermal) or
    when the bedroom count is missing or unparseable. Bedrooms are never
    inferred from floor area — that is exactly the estimation this design
    forbids, and it would be invisible to the fabrication judge.
    """
    technology = (rfq_input.get("common") or {}).get("technology_requested")
    if technology not in TABLE_NAMES:
        return None

    # Read from the requested technology's section only, so a battery enquiry
    # carrying a stray heat_pump dict cannot produce a heat pump price.
    beds = _coerce_int((rfq_input.get(technology) or {}).get("number_of_bedrooms"))
    if beds is None:
        return None

    prop = rfq_input.get("property") or {}
    style = classify_style(prop.get("property_type"), prop.get("built_form"))

    estimate: dict[str, Any] = {
        "technology": technology,
        "source": SOURCE,
        "table": TABLE_NAMES[technology],
        "guide_price_note": GUIDE_PRICE_NOTE,
    }

    if technology == "solar_pv":
        row, exact = _match(SOLAR_PV_ROWS, beds, style)
        estimate.update({
            "matched_band": row["band"],
            "system_size_kw": row["kw"],
            "panels_low": row["panels"][0],
            "panels_high": row["panels"][1],
            "installed_cost_low_gbp": row["cost"][0],
            "installed_cost_high_gbp": row["cost"][1],
            "annual_saving_low_gbp": row["saving"][0],
            "annual_saving_high_gbp": row["saving"][1],
        })
    else:
        if is_large_period(style, beds, _coerce_float(prop.get("floor_area_m2")),
                           prop.get("construction_age_band")):
            row, exact = HEAT_PUMP_LARGE_ROW, True
        else:
            row, exact = _match(HEAT_PUMP_ROWS, beds, style)
        estimate.update({
            "matched_band": row["band"],
            "air_source_cost_low_gbp": row["ashp"][0],
            "air_source_cost_high_gbp": row["ashp"][1],
            "ground_source_cost_low_gbp": row["gshp"][0],
            "ground_source_cost_high_gbp": row["gshp"][1],
            "net_cost_after_grant_low_gbp": row["net"][0],
            "net_cost_after_grant_high_gbp": row["net"][1],
            "grant_name": "Boiler Upgrade Scheme",
            "grant_amount_gbp": BUS_GRANT_GBP,
            "grant_applies_to": "air source",
        })

    # A string, never a boolean: _strip_nulls keeps False, and a small model
    # narrates `approximate: false` as "this is an exact match".
    estimate["match_type"] = "exact" if exact else "nearest_band"
    estimate["match_basis"] = _describe_basis(beds, style)
    if not exact:
        # Carry the hedge as data, not only as a prompt instruction. Same
        # reason derive_planning_consequences exists: the fabrication judge
        # sees only rfq_input, so a prompt-demanded caveat with no matching
        # field reads as an invented claim.
        estimate["match_note"] = (
            "No band matches this property exactly; the figures shown are for "
            "the closest comparable band."
        )
    return estimate
