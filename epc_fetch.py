import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# New gov.uk service. The legacy host (epc.opendatacommunities.org) is
# being retired on 2026-05-30; the new endpoint uses bearer-token auth
# and a different field schema. epc_to_rfq.py still reads the legacy
# hyphenated shape, so this module translates new → legacy.
BASE_URL = "https://api.get-energy-performance-data.communities.gov.uk"

# Code-lookup tables are fetched once per process and cached.
_CODE_CACHE: dict[str, dict[str, str]] = {}


def get_auth_header():
    token = os.getenv("EPC_BEARER_TOKEN")
    if not token:
        # Back-compat: support legacy EPC_API_KEY var if user hasn't renamed it yet.
        token = os.getenv("EPC_API_KEY")
    if not token:
        # Raise (not sys.exit): this module is imported by the API, and a
        # SystemExit would bypass app.py's `except Exception` and crash the worker.
        raise RuntimeError("EPC_BEARER_TOKEN must be set in .env")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _lookup_code(code_type: str, key, headers) -> Optional[str]:
    if key is None or key == "":
        return None
    key_str = str(key)
    table = _CODE_CACHE.get(code_type)
    if table is None:
        try:
            resp = requests.get(
                f"{BASE_URL}/api/codes/info",
                params={"code": code_type},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            entries = resp.json().get("data", []) or []
        except (requests.HTTPError, requests.RequestException, ValueError):
            # Transient failure: return None WITHOUT caching, so a later call
            # retries instead of the process permanently serving an empty table
            # (which would silently blank fuel/recommendations for every property).
            return None
        table = {}
        for entry in entries:
            ekey = str(entry.get("key", ""))
            values = entry.get("values") or []
            if values and ekey not in table:
                table[ekey] = values[0].get("value")
        _CODE_CACHE[code_type] = table
    return table.get(key_str)


def search_by_postcode(postcode, headers, page_size: int = 100):
    url = f"{BASE_URL}/api/domestic/search"
    params = {"postcode": postcode, "page_size": page_size, "current_page": 1}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    if not resp.text:
        return []
    return resp.json().get("data", []) or []


def fetch_certificate(certificate_number, headers):
    url = f"{BASE_URL}/api/certificate"
    resp = requests.get(
        url, headers=headers, params={"certificate_number": certificate_number}, timeout=30
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    if not resp.text:
        return None
    return resp.json().get("data") or None


def _build_address(line1, line2, line3, line4) -> str:
    parts = [p.strip() for p in (line1, line2, line3, line4) if p and str(p).strip()]
    return ", ".join(parts)


def _unwrap_value(x):
    """Some gov.uk EPC fields return a localised dict like
    `{"value": "Mid-floor flat", "language": "1"}` instead of a plain string.
    Older / single-language EPCs return plain strings. Normalise both shapes
    to a plain string so downstream code (Counter, JSON serialisation, the
    LLM prompt) doesn't have to worry about it."""
    if isinstance(x, dict):
        return x.get("value")
    return x


def _normalize_certificate(detail: dict, search_row: dict, headers: dict) -> dict:
    detail = detail or {}
    search_row = search_row or {}

    address = _build_address(
        search_row.get("addressLine1") or detail.get("address_line_1"),
        search_row.get("addressLine2") or detail.get("address_line_2"),
        search_row.get("addressLine3") or detail.get("address_line_3"),
        search_row.get("addressLine4") or detail.get("address_line_4"),
    )

    # Construction age band lives on the first sap_building_part as a code letter.
    cab_code = None
    parts_arr = detail.get("sap_building_parts") or []
    if parts_arr and isinstance(parts_arr, list):
        cab_code = parts_arr[0].get("construction_age_band")
    cab_human = _lookup_code("construction_age_band", cab_code, headers) or cab_code

    # dwelling_type is already a human-readable string ("Top-floor flat") and
    # captures more detail than the numeric property_type code, so prefer it.
    property_type = _unwrap_value(detail.get("dwelling_type")) or _lookup_code(
        "property_type", detail.get("property_type"), headers
    )
    built_form = _lookup_code("built_form", detail.get("built_form"), headers)
    tenure = _lookup_code("tenure", detail.get("tenure"), headers)

    main_fuel_code = (
        ((detail.get("sap_heating") or {}).get("main_heating_details") or [{}])[0]
        .get("main_fuel_type")
    )
    main_fuel = _lookup_code("main_fuel", main_fuel_code, headers)

    main_heat = _unwrap_value(((detail.get("main_heating") or [{}])[0] or {}).get("description"))
    walls = _unwrap_value(((detail.get("walls") or [{}])[0] or {}).get("description"))
    roofs = _unwrap_value(((detail.get("roofs") or [{}])[0] or {}).get("description"))
    window = _unwrap_value((detail.get("window") or {}).get("description"))
    hot_water = _unwrap_value((detail.get("hot_water") or {}).get("description"))

    return {
        "lmk-key": search_row.get("certificateNumber") or detail.get("certificate_number"),
        "address": address,
        "postcode": search_row.get("postcode") or detail.get("postcode", ""),
        "inspection-date": detail.get("inspection_date") or search_row.get("registrationDate", ""),
        "current-energy-rating": (
            search_row.get("currentEnergyEfficiencyBand")
            or detail.get("current_energy_efficiency_band")
        ),
        "current-energy-efficiency": detail.get("energy_rating_current"),
        "property-type": property_type,
        "built-form": built_form,
        "total-floor-area": detail.get("total_floor_area"),
        "construction-age-band": cab_human,
        "mainheat-description": main_heat,
        "main-fuel": main_fuel,
        "hotwater-description": hot_water,
        "walls-description": walls,
        "roof-description": roofs,
        "windows-description": window,
        "tenure": tenure,
    }


def _normalize_recommendations(detail: dict, headers: dict) -> list[dict]:
    if not detail:
        return []
    out = []
    for imp in detail.get("suggested_improvements") or []:
        details = imp.get("improvement_details") or {}
        improvement_number = details.get("improvement_number")
        text = _lookup_code("improvement_summary", improvement_number, headers) or ""
        # typical_saving comes back as either {"value": 309, "currency": "GBP"}
        # or just an int — handle both. Same for indicative_cost (sometimes
        # an int, sometimes a string like "2,700", sometimes a dict).
        ts_raw = imp.get("typical_saving")
        typical_saving = ts_raw.get("value") if isinstance(ts_raw, dict) else ts_raw
        ic_raw = imp.get("indicative_cost")
        indicative_cost = ic_raw.get("value") if isinstance(ic_raw, dict) else ic_raw
        out.append({
            "improvement-summary-text": text,
            "indicative-cost": str(indicative_cost) if indicative_cost is not None else "",
            "typical-saving": str(typical_saving) if typical_saving is not None else "",
            "improvement-id": improvement_number,
        })
    return out


def find_nearby_postcodes(postcode: str, max_count: int = 10) -> list[str]:
    """Return up to `max_count` geographically nearest UK postcodes via the
    free postcodes.io API. The input postcode itself is excluded from the
    result. Returns an empty list on network/API failure rather than raising
    so the caller can degrade gracefully to Case B (manual entry).
    """
    if not postcode:
        return []
    try:
        # postcodes.io accepts no-space and with-space variants; strip spaces
        # to be safe.
        key = postcode.strip().replace(" ", "")
        resp = requests.get(
            f"https://api.postcodes.io/postcodes/{key}/nearest",
            params={"limit": max_count + 1, "radius": 2000},  # +1 because index 0 is the input itself; 2000m is API max
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        body = resp.json() or {}
        results = body.get("result") or []
    except (requests.RequestException, ValueError):
        return []

    normalized_input = postcode.strip().upper().replace(" ", "")
    out: list[str] = []
    for r in results:
        pc = (r.get("postcode") or "").strip()
        if not pc:
            continue
        if pc.upper().replace(" ", "") == normalized_input:
            continue
        out.append(pc)
        if len(out) >= max_count:
            break
    return out


def _safe_fetch_epc_data(postcode: str) -> dict:
    """Wrapper around fetch_epc_data that swallows network errors and
    format-validation 400s, returning an empty-properties result instead.
    Used by the Layer 2 fallback so a bad postcode doesn't 500 the request."""
    try:
        return fetch_epc_data(postcode)
    except (requests.RequestException, ValueError):
        return {"postcode": postcode, "count": 0, "properties": []}



def fetch_epc_data_with_neighbour_fallback(
    postcode: str, target_count: int = 5, max_neighbours: int = 10
) -> dict:
    """Fetch EPCs for `postcode`. If zero records are returned, iterate the
    geographically nearest postcodes (via postcodes.io) and accumulate their
    EPCs until we have at least `target_count` properties, or run out of
    neighbours.

    Returns the same shape as `fetch_epc_data()`, plus two extra keys:
      proxy_used: bool      — True if any neighbouring EPCs were merged in
      proxy_postcodes: list — postcodes that actually contributed certificates
    """
    primary = _safe_fetch_epc_data(postcode)
    if primary.get("count", 0) > 0:
        primary["proxy_used"] = False
        primary["proxy_postcodes"] = [postcode]
        return primary

    # Original postcode has no EPCs — walk nearby postcodes.
    nearby = find_nearby_postcodes(postcode, max_count=max_neighbours)
    accumulated: list[dict] = []
    contributing: list[str] = []
    for pc in nearby:
        data = _safe_fetch_epc_data(pc)
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


def fetch_epc_data(postcode):
    headers = get_auth_header()
    print(f"Searching EPC records for postcode: {postcode}")

    rows = search_by_postcode(postcode, headers)
    print(f"Found {len(rows)} properties\n")

    properties = []
    for row in rows:
        cert_num = row.get("certificateNumber")
        detail = fetch_certificate(cert_num, headers) if cert_num else None
        cert = _normalize_certificate(detail, row, headers)
        recommendations = _normalize_recommendations(detail, headers)

        addr = cert.get("address") or "Unknown"
        rating = cert.get("current-energy-rating") or "?"
        print(f"  {addr} — Rating: {rating}")

        properties.append({
            "certificate": cert,
            "recommendations": recommendations,
        })

    result = {
        "postcode": postcode,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(properties),
        "properties": properties,
    }

    os.makedirs("output", exist_ok=True)
    filename = f"output/{postcode.replace(' ', '_')}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {filename}")

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pc = " ".join(sys.argv[1:])
    else:
        pc = input("Enter postcode: ")
    fetch_epc_data(pc.strip())
