"""External data sources for site-intelligence enrichment.

Two pure functions, both failure-tolerant (any network or HTTP error returns
an empty/None result so the rest of the pipeline keeps working):

  fetch_planning_constraints(postcode) -> dict
      Hits planning.data.gov.uk (free, Open Government Licence) for the
      postcode's centroid and returns flags for listed buildings,
      conservation areas, Article 4 directions, AONBs, World Heritage Sites,
      and National Parks.

  fetch_ukpn_constraints(postcode) -> dict | None
      Hits the UKPN Open Data Portal (CC BY 4.0 / OGL UK 3.0, registration
      required for the headroom dataset) for the nearest PRIMARY substation
      and returns its demand + generation headroom figures. Returns None when
      the postcode isn't in UKPN's licence area or when no API key is set.

Both functions cache responses in-process keyed by postcode.

Attribution required by UKPN's CC BY 4.0 licence; the demo page surfaces this
in the site-intelligence panel footer.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import requests


PLANNING_BASE = "https://www.planning.data.gov.uk/entity.json"
UKPN_BASE = (
    "https://ukpowernetworks.opendatasoft.com/api/explore/v2.1/catalog/datasets"
)
POSTCODES_IO = "https://api.postcodes.io/postcodes"

# In-process cache. Keyed by normalised postcode.
_PLANNING_CACHE: dict[str, dict] = {}
_UKPN_CACHE: dict[str, Optional[dict]] = {}
_LATLONG_CACHE: dict[str, Optional[tuple[float, float]]] = {}


def _normalise(postcode: str) -> str:
    return (postcode or "").strip().upper().replace(" ", "")


def _resolve_latlong(postcode: str) -> Optional[tuple[float, float]]:
    """Postcode -> (lat, lon) via postcodes.io. Cached, failure-tolerant."""
    key = _normalise(postcode)
    if not key:
        return None
    if key in _LATLONG_CACHE:
        return _LATLONG_CACHE[key]
    try:
        resp = requests.get(f"{POSTCODES_IO}/{key}", timeout=10)
        if resp.status_code != 200:
            _LATLONG_CACHE[key] = None
            return None
        body = resp.json().get("result") or {}
        lat = body.get("latitude")
        lon = body.get("longitude")
        out = (float(lat), float(lon)) if lat is not None and lon is not None else None
    except (requests.RequestException, ValueError, TypeError):
        out = None
    _LATLONG_CACHE[key] = out
    return out


# --------------------------------------------------------------------------
# planning.data.gov.uk
# --------------------------------------------------------------------------

# Datasets we query. Each one is queried by lat/long and we report whether
# the centroid is inside any returned entity's geometry.
_PLANNING_DATASETS = [
    "listed-building",
    "conservation-area",
    "article-4-direction-area",
    "area-of-outstanding-natural-beauty",
    "world-heritage-site",
    "national-park",
]


def _planning_query(dataset: str, lat: float, lon: float) -> list[dict]:
    """One dataset query. Returns the entity list, or [] on any failure."""
    try:
        resp = requests.get(
            PLANNING_BASE,
            params={
                "latitude": lat,
                "longitude": lon,
                "dataset": dataset,
                "limit": 5,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("entities") or []
    except (requests.RequestException, ValueError):
        return []


def fetch_planning_constraints(postcode: str) -> dict:
    """Return a dict of planning-constraint flags at the postcode's centroid.

    Always returns the full set of keys (so the caller doesn't have to deal
    with absent keys). Missing/no-hit values are False/None.
    """
    key = _normalise(postcode)
    if key in _PLANNING_CACHE:
        return _PLANNING_CACHE[key]

    empty = {
        "listed_building": False,
        "listed_grade": None,
        "conservation_area_name": None,
        "article_4": False,
        "aonb_name": None,
        "whs_name": None,
        "national_park_name": None,
    }

    coords = _resolve_latlong(postcode)
    if coords is None:
        _PLANNING_CACHE[key] = empty
        return empty

    lat, lon = coords
    out = dict(empty)

    for ds in _PLANNING_DATASETS:
        entities = _planning_query(ds, lat, lon)
        if not entities:
            continue
        first = entities[0]
        name = first.get("name") or None
        if ds == "listed-building":
            out["listed_building"] = True
            # planning.data.gov.uk exposes grade as "listed-building-grade"
            # (hyphenated, like the rest of its fields).
            out["listed_grade"] = first.get("listed-building-grade") or None
        elif ds == "conservation-area":
            out["conservation_area_name"] = name
        elif ds == "article-4-direction-area":
            out["article_4"] = True
        elif ds == "area-of-outstanding-natural-beauty":
            out["aonb_name"] = name
        elif ds == "world-heritage-site":
            out["whs_name"] = name
        elif ds == "national-park":
            out["national_park_name"] = name

    _PLANNING_CACHE[key] = out
    return out


# --------------------------------------------------------------------------
# UKPN Open Data Portal
# --------------------------------------------------------------------------

def _ukpn_auth_headers() -> Optional[dict]:
    """UKPN headroom dataset is gated. Read the key from env at call time so
    test runs can patch the env without re-importing the module."""
    key = os.getenv("UKPN_API_KEY")
    if not key:
        return None
    return {"Authorization": f"apikey {key}"}


def _ukpn_query(dataset: str, params: dict, headers: dict) -> dict:
    """Single UKPN query. Returns the JSON body, or {} on any failure."""
    try:
        resp = requests.get(
            f"{UKPN_BASE}/{dataset}/records",
            params=params,
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            return {}
        return resp.json() or {}
    except (requests.RequestException, ValueError):
        return {}


def fetch_ukpn_constraints(postcode: str, year: str = "2026") -> Optional[dict]:
    """Return UKPN grid headroom for the nearest PRIMARY substation.

    None when the postcode is outside UKPN's licence area, when no API key is
    set, or on any network/parse failure. Otherwise a dict with the headroom
    figures for the nearest primary substation under the Counterfactual
    scenario (the do-nothing baseline; most defensible default).
    """
    key = _normalise(postcode)
    if key in _UKPN_CACHE:
        return _UKPN_CACHE[key]

    headers = _ukpn_auth_headers()
    if headers is None:
        _UKPN_CACHE[key] = None
        return None

    coords = _resolve_latlong(postcode)
    if coords is None:
        _UKPN_CACHE[key] = None
        return None

    lat, lon = coords

    # 1. Nearest PRIMARY substation, Demand Headroom (Counterfactual, this year).
    where_demand = (
        f"within_distance(spatial_coordinates, geom'POINT({lon} {lat})', 5000m)"
        f" and physical_level='PRIMARY' and scenario='Counterfactual'"
        f" and category='Demand Headroom' and year=date'{year}'"
    )
    order_by = f"distance(spatial_coordinates, geom'POINT({lon} {lat})') asc"
    demand_body = _ukpn_query(
        "dfes-network-headroom-report",
        {"where": where_demand, "order_by": order_by, "limit": 1},
        headers,
    )
    demand_rows = demand_body.get("results") or []
    if not demand_rows:
        # No PRIMARY substation within 5 km → not UKPN territory (or no data).
        _UKPN_CACHE[key] = None
        return None

    nearest = demand_rows[0]
    substation = nearest.get("substation_name") or ""

    # 2. Generation inverter headroom for the same substation, same year/scenario.
    #    These records sit at a different physical_level (None) but the
    #    substation name is shared.
    where_gen = (
        f"substation_name='{substation}'"
        f" and scenario='Counterfactual'"
        f" and category='Gen inverter headroom'"
        f" and year=date'{year}'"
    )
    gen_body = _ukpn_query(
        "dfes-network-headroom-report",
        {"where": where_gen, "limit": 1},
        headers,
    )
    gen_rows = gen_body.get("results") or []
    gen_row = gen_rows[0] if gen_rows else {}

    out = {
        "licence_area": nearest.get("licencearea"),
        "primary_substation": substation,
        "demand_headroom_mw": nearest.get("headroom_mw"),
        "demand_headroom_percent": nearest.get("headroom_percent"),
        "demand_rag": nearest.get("demand_rag"),
        "generation_headroom_mw": gen_row.get("headroom_mw"),
        "year": year,
        "scenario": "Counterfactual",
    }
    _UKPN_CACHE[key] = out
    return out


# --------------------------------------------------------------------------
# Caches (exposed for tests)
# --------------------------------------------------------------------------

def _clear_caches() -> None:
    _PLANNING_CACHE.clear()
    _UKPN_CACHE.clear()
    _LATLONG_CACHE.clear()
