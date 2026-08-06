"""End-to-end smoke test for the API, with EPC fetcher and LLM mocked.

Run:
  python test_api.py

Validates:
  /health
  /api/initiate (postcode + tech)
  /api/generate (Step 2 fields)
  /api/generate-rfq (vendor selection)
  /api/session/{id}
  Error paths: ambiguous postcode, unknown session, missing required fields
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


# Cached EPC fixture from the existing /EPC/output dump.
FIXTURE = Path(__file__).parent.parent / "EPC" / "output" / "E1_6AN.json"


def _make_synthetic_property(address: str, lmk_key: str, inspection_date: str) -> dict:
    """Minimal valid EPC property record for ambiguity testing."""
    return {
        "certificate": {
            "lmk-key": lmk_key,
            "address": address,
            "inspection-date": inspection_date,
            "property-type": "House",
            "built-form": "Detached",
            "total-floor-area": "100.0",
            "construction-age-band": "England and Wales: 1976-1982",
            "current-energy-rating": "C",
            "current-energy-efficiency": "70",
            "mainheat-description": "Boiler and radiators, mains gas",
            "main-fuel": "mains gas (not community)",
            "hotwater-description": "From main system",
            "walls-description": "Cavity wall, insulated",
            "roof-description": "Pitched, insulated",
            "windows-description": "Double glazed",
            "tenure": "owner-occupied",
        },
        "recommendations": [],
    }


# CV12 8UE-style fixture: many distinct yards/streets, some sharing house numbers
SYNTHETIC_MULTI_ADDRESS = {
    "postcode": "CV12 8UE",
    "fetched_at": "fake",
    "count": 9,
    "properties": [
        _make_synthetic_property("1 Old Penns Yard", "lmk-old1-2025", "2025-11-21"),
        _make_synthetic_property("5 Old Penns Yard", "lmk-old5-2023", "2023-08-18"),
        _make_synthetic_property("1 Bucklers Yard", "lmk-buc1-2023", "2023-11-01"),
        _make_synthetic_property("4 Bucklers Yard", "lmk-buc4-2023", "2023-10-13"),
        _make_synthetic_property("2 Sleets Yard", "lmk-sle2-2023", "2023-01-17"),
        _make_synthetic_property("4 Sleets Yard", "lmk-sle4-2022", "2022-07-11"),
        _make_synthetic_property("2 Lye Corner", "lmk-lye2-2024", "2024-04-25"),
        _make_synthetic_property("2 Emes Walk", "lmk-eme2-2021", "2021-05-06"),
        _make_synthetic_property("10 Old Penns Yard", "lmk-old10-2024", "2024-06-01"),
    ],
}


def _fake_epc_fetch_factory(fixture_path: Path):
    cached = json.loads(fixture_path.read_text())

    def fake_fetch(postcode: str):
        normalized = postcode.upper().replace(" ", "")
        if normalized == cached["postcode"].upper().replace(" ", ""):
            return cached
        if normalized == "CV128UE":
            return SYNTHETIC_MULTI_ADDRESS
        return {"postcode": postcode, "fetched_at": "fake", "count": 0, "properties": []}

    return fake_fetch


class FakeRecommendation:
    @staticmethod
    def __call__(rfq_input):
        items = rfq_input.get("recommendation", {}).get("raw_recommendation_items", [])
        if not items:
            return {
                "recommendation_summary": (
                    "Your property has no official EPC improvement recommendations on record."
                ),
                "recommendation_disclaimer": "Based only on official EPC recommendation data.",
            }
        return {
            "recommendation_summary": (
                f"Your home's EPC report suggests these upgrades: "
                f"{', '.join(items[:3])}. Following them could improve your rating."
            ),
            "recommendation_disclaimer": "Based only on official EPC recommendation data.",
        }


class FakeRfq:
    @staticmethod
    def __call__(rfq_input):
        tech = rfq_input.get("common", {}).get("technology_requested", "?")
        ptype = rfq_input.get("property", {}).get("property_type", "unknown")
        return {
            "rfq_summary": (
                f"The homeowner is requesting a {tech} quotation for a {ptype} property. "
                f"(mock RFQ summary for testing)"
            )
        }


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Patch BEFORE importing app, so app picks up the mocks at import time.
        cls.fetch_patcher = patch("epc_fetch.fetch_epc_data", _fake_epc_fetch_factory(FIXTURE))
        cls.fetch_patcher.start()

        # Default: no nearby postcodes. Tests exercising the auto-fallback
        # path override this with an inline `with patch(...)`.
        cls.nearby_patcher = patch("epc_fetch.find_nearby_postcodes", return_value=[])
        cls.nearby_patcher.start()

        # Default: site_context returns empty stubs so tests stay offline.
        # Individual tests override these with `with patch(...)` blocks.
        cls.planning_patcher = patch(
            "external_data.fetch_planning_constraints",
            return_value={
                "listed_building": False, "listed_grade": None,
                "conservation_area_name": None, "article_4": False,
                "aonb_name": None, "whs_name": None, "national_park_name": None,
            },
        )
        cls.planning_patcher.start()
        cls.ukpn_patcher = patch("external_data.fetch_ukpn_constraints", return_value=None)
        cls.ukpn_patcher.start()

        # Import after patching epc_fetch (app.py re-imports it).
        import app
        cls.app_module = app

        # Patch the LLM functions (looked up dynamically inside handlers), so
        # no test ever reaches the hosted endpoint.
        cls.gen_rec_patcher = patch("generate_rfq.generate_recommendation", FakeRecommendation())
        cls.gen_rfq_patcher = patch("generate_rfq.generate_rfq_summary", FakeRfq())
        cls.gen_rec_patcher.start()
        cls.gen_rfq_patcher.start()

        # Clear sessions
        cls.app_module._SESSIONS.clear()
        cls.client = TestClient(cls.app_module.app)

    @classmethod
    def tearDownClass(cls):
        cls.gen_rfq_patcher.stop()
        cls.gen_rec_patcher.stop()
        cls.ukpn_patcher.stop()
        cls.planning_patcher.stop()
        cls.nearby_patcher.stop()
        cls.fetch_patcher.stop()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertIn("demo_mock_llm", r.json())

    def test_full_happy_path(self):
        # Step 1: initiate
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "house_number": "6", "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        sid = body["session_id"]

        self.assertTrue(body["epc_found"])
        self.assertEqual(body["auto_filled"]["property_type"], "flat")
        self.assertEqual(body["auto_filled"]["epc_rating"], "D")

        missing = body["missing_fields"]
        self.assertIn("common", missing)
        self.assertIn("heat_pump", missing)
        # property is fully populated by EPC for this fixture
        self.assertNotIn("property", missing)

        # The form would render these in Step 2:
        common_field_names = {f["name"] for f in missing["common"]}
        self.assertEqual(common_field_names, {
            "preferred_contact_method", "contact_email", "contact_phone",
            "desired_installation_timeline", "motivation",
        })
        heat_pump_field_names = {f["name"] for f in missing["heat_pump"]}
        self.assertEqual(heat_pump_field_names, {
            "emitter_type", "hot_water_cylinder_space_available",
            "external_unit_space", "garden_or_side_access",
            "heat_pump_type_interest",
        })

        # Step 2: generate (with all required fields filled in)
        r = self.client.post("/api/generate", json={
            "session_id": sid,
            "additional_fields": {
                "common": {
                    "preferred_contact_method": "email",
                    "contact_email": "homeowner@example.com",
                    "contact_phone": "07700 900123",
                    "desired_installation_timeline": "within_6_months",
                    "motivation": "reduce energy bills and replace gas heating",
                },
                "heat_pump": {
                    "emitter_type": "radiators",
                    "hot_water_cylinder_space_available": "yes",
                    "external_unit_space": "yes",
                    "garden_or_side_access": "yes",
                    "heat_pump_type_interest": "ground_air_source",
                },
            },
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("recommendation_summary", body)
        self.assertIn("Room-in-roof insulation", body["recommendation_summary"])
        self.assertEqual(body["completeness"]["score"], 1.0)

        # Vendor selection -> RFQ summary
        r = self.client.post("/api/generate-rfq", json={
            "session_id": sid, "vendor_id": "installer_42",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("rfq_summary", body)
        self.assertIn("heat_pump", body["rfq_summary"])
        self.assertTrue(body["ready_to_submit"])

        # Session inspector
        r = self.client.get(f"/api/session/{sid}")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIsNotNone(body["recommendation"])
        self.assertIsNotNone(body["rfq_summary"])
        self.assertEqual(body["vendor_id"], "installer_42")

    def test_missing_session(self):
        r = self.client.post("/api/generate", json={
            "session_id": "does-not-exist", "additional_fields": {},
        })
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["detail"]["error"], "unknown_session")

    def test_blocks_generate_when_missing_fields(self):
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "house_number": "6", "technology": "heat_pump",
        })
        sid = r.json()["session_id"]

        # /generate without filling Step 2 -> should 422
        r = self.client.post("/api/generate", json={"session_id": sid, "additional_fields": {}})
        self.assertEqual(r.status_code, 422)
        detail = r.json()["detail"]
        self.assertEqual(detail["error"], "still_missing_required_fields")
        self.assertIn("common", detail["missing_fields"])

    def test_invalid_technology(self):
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "spaceship",
        })
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"]["error"], "invalid_technology")

    def test_battery_initiate(self):
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "battery",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("battery", body["missing_fields"])
        names = {f["name"] for f in body["missing_fields"]["battery"]}
        for required_field in (
            "existing_solar_pv", "battery_purpose",
            "backup_power_required", "battery_location_preference",
        ):
            self.assertIn(required_field, names)

    def test_proxy_aggregates_from_multiple_neighbours(self):
        # CV12 8UE has 9 synthetic addresses in the test fixture. With use_proxy=True
        # the server should skip address selection and aggregate them into a single
        # proxy property section tagged epc_source="proxy".
        r = self.client.post("/api/initiate", json={
            "postcode": "CV12 8UE", "technology": "heat_pump", "use_proxy": True,
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertFalse(body["epc_found"])
        auto = body["auto_filled"]
        self.assertEqual(auto["epc_source"], "proxy")
        self.assertEqual(auto["proxy_comparator_count"], 9)
        self.assertEqual(auto["proxy_confidence"], "high")  # n >= 5
        # Aggregated fields should be populated (mode/median of fixture data)
        self.assertEqual(auto["property_type"], "house")
        self.assertEqual(auto["epc_rating"], "C")
        # No required property fields should be missing now
        self.assertNotIn("property", body["missing_fields"])

    def test_nearby_postcode_proxy_returns_picker_candidates(self):
        # ZZ1 1ZZ has no EPCs in the fixture. The Layer 2 path should now call
        # postcodes.io for nearby postcodes, find EPCs there, and return them
        # as a picker candidate list (409) instead of silently aggregating.
        with patch(
            "epc_fetch.find_nearby_postcodes",
            return_value=["E1 6AN", "CV12 8UE"],
        ):
            r = self.client.post("/api/initiate", json={
                "postcode": "ZZ1 1ZZ", "technology": "heat_pump",
            })
        self.assertEqual(r.status_code, 409, r.text)
        detail = r.json()["detail"]
        self.assertEqual(detail["error"], "proxy_nearby_candidates")
        candidates = detail["candidates"]
        self.assertGreaterEqual(len(candidates), 1)
        # Every candidate must carry the four form-vocabulary fields the picker UI displays
        for c in candidates:
            for key in (
                "lmk_key", "postcode", "address", "inspection_date",
                "property_type", "floor_area_m2",
                "current_heating_system", "current_fuel_type",
            ):
                self.assertIn(key, c)
        # The CV12 8UE synthetic fixture should contribute its lmk_keys
        lmks = {c["lmk_key"] for c in candidates}
        self.assertIn("lmk-old1-2025", lmks)

    def test_nearby_pick_one_returns_proxy_picked(self):
        # Follow-up after picker: user picked "1 Old Penns Yard" (lmk-old1-2025)
        # in nearby postcode CV12 8UE. Server should fetch that specific EPC
        # and mark the resulting property section as proxy_picked=True.
        r = self.client.post("/api/initiate", json={
            "postcode": "ZZ1 1ZZ",
            "technology": "heat_pump",
            "lmk_key": "lmk-old1-2025",
            "proxy_postcode": "CV12 8UE",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertFalse(body["epc_found"])
        auto = body["auto_filled"]
        self.assertEqual(auto["epc_source"], "proxy_nearby")
        self.assertTrue(auto["proxy_picked"])
        self.assertEqual(auto["proxy_comparator_count"], 1)
        self.assertEqual(auto["proxy_postcodes"], ["CV12 8UE"])
        # The chosen EPC was for a "House", so the picked property section
        # should reflect that (not an aggregate).
        self.assertEqual(auto["property_type"], "house")
        # Property required fields are now populated, so they shouldn't appear in missing.
        self.assertNotIn("property", body["missing_fields"])

    def test_nearby_use_average_returns_aggregate(self):
        # Follow-up after picker: user clicked "Use average of all". Server
        # should re-run the nearby search and aggregate the EPCs (existing
        # proxy_nearby behaviour, now triggered explicitly).
        with patch(
            "epc_fetch.find_nearby_postcodes",
            return_value=["E1 6AN", "CV12 8UE"],
        ):
            r = self.client.post("/api/initiate", json={
                "postcode": "ZZ1 1ZZ",
                "technology": "heat_pump",
                "use_proxy_nearby_average": True,
            })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertFalse(body["epc_found"])
        auto = body["auto_filled"]
        self.assertEqual(auto["epc_source"], "proxy_nearby")
        # The aggregate path does NOT set proxy_picked
        self.assertNotIn("proxy_picked", auto)
        self.assertGreater(auto["proxy_comparator_count"], 1)
        self.assertNotIn("property", body["missing_fields"])

    def test_nearby_fallback_skipped_when_use_proxy_explicit(self):
        # When the user explicitly requested same-postcode proxy, we don't
        # expand the radius behind their back. Even if postcodes.io is mocked
        # to return neighbours, the explicit use_proxy=True path should stay
        # within the original postcode.
        with patch(
            "epc_fetch.find_nearby_postcodes",
            return_value=["E1 6AN"],
        ) as nearby_mock:
            r = self.client.post("/api/initiate", json={
                "postcode": "CV12 8UE", "technology": "heat_pump",
                "use_proxy": True,
            })
        self.assertEqual(r.status_code, 200, r.text)
        # find_nearby_postcodes must not have been called: the same-postcode
        # path returns 9 comparators from the fixture, so the fallback path
        # is bypassed entirely.
        nearby_mock.assert_not_called()
        self.assertEqual(r.json()["auto_filled"]["epc_source"], "proxy")

    def test_proxy_with_zero_epcs_falls_back_to_case_b(self):
        # ZZ1 1ZZ has no EPCs in the fixture. use_proxy=True should fall through
        # to Case B (manual entry) rather than synthesising values from nothing.
        r = self.client.post("/api/initiate", json={
            "postcode": "ZZ1 1ZZ", "technology": "heat_pump", "use_proxy": True,
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertFalse(body["epc_found"])
        # No proxy marker — we fell through to Case B
        self.assertNotIn("epc_source", body["auto_filled"])
        # Property required fields are missing (Case B form)
        self.assertIn("property", body["missing_fields"])
        names = {f["name"] for f in body["missing_fields"]["property"]}
        # floor_area_m2 is optional, so Case B asks three questions.
        self.assertEqual(names, {
            "property_type",
            "current_heating_system", "current_fuel_type",
        })

    def test_site_context_planning_constraints(self):
        # Planning lookup returns a listed-building + conservation hit.
        # Expect those to surface on the InitiateResponse.site_context.
        with patch(
            "external_data.fetch_planning_constraints",
            return_value={
                "listed_building": True, "listed_grade": "II",
                "conservation_area_name": "Bath", "article_4": False,
                "aonb_name": None, "whs_name": "City of Bath",
                "national_park_name": None,
            },
        ):
            r = self.client.post("/api/initiate", json={
                "postcode": "BA1 1LZ", "technology": "heat_pump",
            })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        ctx = body["site_context"]
        self.assertEqual(ctx["planning"]["listed_grade"], "II")
        self.assertEqual(ctx["planning"]["conservation_area_name"], "Bath")
        self.assertEqual(ctx["planning"]["whs_name"], "City of Bath")
        # Grid is None (no UKPN mock override) — Bath isn't in UKPN territory
        self.assertIsNone(ctx["grid"])
        # data_sources lists planning since planning hit is non-empty
        self.assertIn("planning.data.gov.uk", ctx["data_sources"])

    def test_site_context_ukpn_present_for_in_area_postcode(self):
        # UKPN lookup returns headroom for a London postcode — expect it on
        # the response. Tests that the grid section reaches the client.
        with patch(
            "external_data.fetch_ukpn_constraints",
            return_value={
                "licence_area": "London Power Networks (LPN)",
                "primary_substation": "Tooley St 11kV",
                "demand_headroom_mw": 12.6,
                "demand_headroom_percent": 34.43,
                "demand_rag": "Green (over 5% headroom)",
                "generation_headroom_mw": 42.82,
                "year": "2026",
                "scenario": "Counterfactual",
            },
        ):
            r = self.client.post("/api/initiate", json={
                "postcode": "SE1 9SG", "technology": "solar_pv",
            })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        grid = body["site_context"]["grid"]
        self.assertIsNotNone(grid)
        self.assertEqual(grid["primary_substation"], "Tooley St 11kV")
        self.assertEqual(grid["demand_headroom_mw"], 12.6)
        self.assertIn("UK Power Networks Open Data", body["site_context"]["data_sources"])

    def test_site_context_external_failure_is_tolerated(self):
        # Both external fetchers raise — the API must still return 200 with
        # an empty site_context, never 500.
        with patch(
            "external_data.fetch_planning_constraints",
            side_effect=RuntimeError("planning unreachable"),
        ), patch(
            "external_data.fetch_ukpn_constraints",
            side_effect=RuntimeError("ukpn unreachable"),
        ):
            r = self.client.post("/api/initiate", json={
                "postcode": "E1 6AN", "technology": "heat_pump",
            })
        self.assertEqual(r.status_code, 200, r.text)
        ctx = r.json()["site_context"]
        # Empty/default shape, no crash
        self.assertEqual(ctx["grid"], None)
        self.assertEqual(ctx["data_sources"], [])

    def test_solar_thermal_initiate(self):
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "solar_thermal",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("solar_thermal", body["missing_fields"])
        names = {f["name"] for f in body["missing_fields"]["solar_thermal"]}
        for required_field in (
            "roof_orientation", "roof_shading_level",
            "hot_water_cylinder_space_available",
            "number_of_occupants", "number_of_bathrooms",
        ):
            self.assertIn(required_field, names)
        self.assertNotIn("usable_roof_area_m2", names)  # optional, must not gate

    def test_contact_details_asked_but_kept_out_of_prompts(self):
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 200, r.text)
        fields = {f["name"]: f for f in r.json()["missing_fields"]["common"]}
        self.assertIn("contact_email", fields)
        self.assertIn("contact_phone", fields)
        self.assertEqual(
            fields["preferred_contact_method"]["options"], ["email", "phone"],
        )

        import json as _json
        import generate_rfq
        from evaluation.faithfulness import _relevant_rfq_fields
        rfq_input = {
            "common": {
                "technology_requested": "heat_pump",
                "preferred_contact_method": "email",
                "contact_email": "homeowner@example.com",
                "contact_phone": "07700 900123",
            },
        }
        for build in (generate_rfq.build_rfq_prompt,
                      generate_rfq.build_recommendation_prompt):
            blob = _json.dumps(build(rfq_input))
            self.assertNotIn("homeowner@example.com", blob)
            self.assertNotIn("900123", blob)
        self.assertEqual(rfq_input["common"]["contact_email"], "homeowner@example.com")

        checked = {f for _, f, _ in _relevant_rfq_fields(rfq_input)}
        self.assertNotIn("contact_email", checked)
        self.assertNotIn("contact_phone", checked)

    def test_optional_measurements_still_scored_when_supplied(self):
        """Optional on the form, but preservation still checks them when supplied."""
        from epc_to_rfq import FIELDS, completeness_score
        from evaluation.faithfulness import _relevant_rfq_fields

        self.assertFalse(FIELDS["property"]["floor_area_m2"]["required"])
        for section in ("solar_pv", "solar_thermal"):
            self.assertFalse(FIELDS[section]["usable_roof_area_m2"]["required"])

        populated = {
            "common": {"technology_requested": "solar_pv"},
            "property": {"floor_area_m2": 78.0},
            "solar_pv": {"usable_roof_area_m2": 24.0},
        }
        checked = {f for _, f, _ in _relevant_rfq_fields(populated)}
        self.assertIn("floor_area_m2", checked)
        self.assertIn("usable_roof_area_m2", checked)
        fields = [f for _, f, _ in _relevant_rfq_fields(populated)]
        self.assertEqual(len(fields), len(set(fields)))

        # Absent means not checked, so a blank is never counted as a miss.
        blank = {"common": {"technology_requested": "solar_pv"},
                 "property": {}, "solar_pv": {}}
        checked_blank = {f for _, f, _ in _relevant_rfq_fields(blank)}
        self.assertNotIn("floor_area_m2", checked_blank)
        self.assertNotIn("usable_roof_area_m2", checked_blank)

        full = {
            "common": {
                "technology_requested": "solar_pv",
                "preferred_contact_method": "email",
                "contact_email": "a@example.com", "contact_phone": "07700 900000",
                "desired_installation_timeline": "flexible", "motivation": "bills",
            },
            "property": {"property_type": "flat", "current_heating_system": "gas_boiler_combi",
                         "current_fuel_type": "mains_gas"},
            "solar_pv": {"roof_orientation": "south", "roof_shading_level": "low"},
        }
        self.assertEqual(completeness_score(full)["score"], 1.0)

    def test_heat_pump_type_is_asked(self):
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 200, r.text)
        fields = {f["name"]: f for f in r.json()["missing_fields"]["heat_pump"]}
        self.assertIn("heat_pump_type_interest", fields)
        self.assertEqual(
            fields["heat_pump_type_interest"]["options"],
            ["ground_air_source", "solar_assisted"],
        )
        # Solar-assisted only, so it must stay optional.
        self.assertNotIn("absorber_mounting_location", fields)

    def test_epc_not_found_falls_back(self):
        # Postcode that our fake fetcher returns 0 results for
        r = self.client.post("/api/initiate", json={
            "postcode": "ZZ1 1ZZ", "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertFalse(body["epc_found"])
        # In Case B, all property fields are missing too
        self.assertIn("property", body["missing_fields"])

    def test_substring_match_does_not_confuse_1_with_10(self):
        # CV12 8UE has both "1 Old Penns Yard" and "10 Old Penns Yard".
        # Filtering by house_number="1" must match only the former, not "10".
        # Even so, "1" still appears at "1 Bucklers Yard", so this is ambiguous.
        r = self.client.post("/api/initiate", json={
            "postcode": "CV12 8UE", "house_number": "1", "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 409, r.text)
        detail = r.json()["detail"]
        self.assertEqual(detail["error"], "ambiguous_address")
        addrs = {c["address"] for c in detail["candidates"]}
        self.assertIn("1 Old Penns Yard", addrs)
        self.assertIn("1 Bucklers Yard", addrs)
        # Critical: "10 Old Penns Yard" must NOT match house_number="1"
        self.assertNotIn("10 Old Penns Yard", addrs)

    def test_ambiguous_includes_lmk_key_for_picking(self):
        r = self.client.post("/api/initiate", json={
            "postcode": "CV12 8UE", "house_number": "2", "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 409, r.text)
        detail = r.json()["detail"]
        # All three "2 X" addresses should be returned with their lmk_keys
        addrs = {c["address"] for c in detail["candidates"]}
        self.assertEqual(addrs, {"2 Sleets Yard", "2 Lye Corner", "2 Emes Walk"})
        for c in detail["candidates"]:
            self.assertTrue(c["lmk_key"], "every candidate must carry an lmk_key")
            self.assertTrue(c["inspection_date"], "...and an inspection_date")

    def test_single_address_postcode_needs_no_house_number(self):
        # E1 6AN has 3 EPC records but all for "6, Brushfield Street" —
        # group-by-address collapses to one. No house_number needed.
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["epc_found"])
        self.assertEqual(body["auto_filled"]["epc_rating"], "D")

    def test_multi_address_postcode_returns_candidates_without_hint(self):
        # CV12 8UE has 9 distinct addresses — without house_number or lmk_key
        # the API should return all 9 for the user to pick from.
        r = self.client.post("/api/initiate", json={
            "postcode": "CV12 8UE", "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 409, r.text)
        candidates = r.json()["detail"]["candidates"]
        self.assertEqual(len(candidates), 9)

    def test_lmk_key_resolves_after_user_picks(self):
        # User got the 409, picked "2 Sleets Yard" -> client re-calls with its lmk_key
        r = self.client.post("/api/initiate", json={
            "postcode": "CV12 8UE",
            "lmk_key": "lmk-sle2-2023",
            "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["epc_found"])
        # The chosen record's auto-fill should reflect that specific property
        # (we can't check address directly because property section doesn't carry it;
        #  but we can check the EPC rating made it through)
        self.assertEqual(body["auto_filled"]["epc_rating"], "C")


if __name__ == "__main__":
    unittest.main(verbosity=2)
