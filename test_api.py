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
        # Select the disclaimer the way production does, so the two cannot drift.
        from generate_rfq import recommendation_disclaimer

        items = rfq_input.get("recommendation", {}).get("raw_recommendation_items", [])
        if not items:
            return {
                "recommendation_summary": (
                    "Your property has no official EPC improvement recommendations on record."
                ),
                "recommendation_disclaimer": recommendation_disclaimer(rfq_input),
            }
        return {
            "recommendation_summary": (
                f"Your home's EPC report suggests these upgrades: "
                f"{', '.join(items[:3])}. Following them could improve your rating."
            ),
            "recommendation_disclaimer": recommendation_disclaimer(rfq_input),
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


class MockedAPITestCase(unittest.TestCase):
    """EPC fetch, external data and both LLM calls mocked.

    Carries no test methods, so a subclass inherits the environment without
    re-running anything.
    """

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


class TestAPI(MockedAPITestCase):
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
            "number_of_bedrooms", "number_of_bathrooms",
            "number_of_occupants", "smart_meter_installed",
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
                    "number_of_bedrooms": 3,
                    "number_of_bathrooms": 2,
                    "number_of_occupants": 4,
                    "smart_meter_installed": "yes",
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
            "backup_power_required", "battery_space_available",
            "battery_location_preference",
        ):
            self.assertIn(required_field, names)
        self.assertNotIn("monthly_electricity_bill_gbp", names)  # optional, must not gate

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
            "solar_pv": {"roof_orientation": "south", "roof_shading_level": "low",
                         "roof_condition": "yes", "number_of_bedrooms": 3,
                         "number_of_bathrooms": 2, "number_of_occupants": 4},
        }
        self.assertEqual(completeness_score(full)["score"], 1.0)

    def test_optional_fields_offered_but_never_gate(self):
        """Optional measurements are offered in the form, separately from
        missing_fields, so a blank cannot block /api/generate."""
        from epc_to_rfq import missing_fields, optional_fields

        # Case B: no EPC, so both measurements are offered.
        with patch("epc_fetch.fetch_epc_data",
                   return_value={"postcode": "SW1A 2AA", "count": 0, "properties": []}), \
             patch("epc_fetch.find_nearby_postcodes", return_value=[]):
            r = self.client.post("/api/initiate", json={
                "postcode": "SW1A 2AA", "technology": "solar_pv",
            })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        offered = {s: {f["name"] for f in v} for s, v in body["optional_fields"].items()}
        self.assertEqual(offered.get("property"), {"floor_area_m2"})
        self.assertEqual(offered.get("solar_pv"),
                         {"usable_roof_area_m2", "monthly_electricity_bill_gbp"})
        # Never duplicated into the gating list.
        for section, fields in body["missing_fields"].items():
            names = {f["name"] for f in fields}
            self.assertNotIn("floor_area_m2", names)
            self.assertNotIn("usable_roof_area_m2", names)

        # EPC found: floor area is auto-filled, so it is not offered again.
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "solar_pv",
        })
        offered = r.json()["optional_fields"]
        self.assertNotIn("property", offered)
        self.assertEqual({f["name"] for f in offered["solar_pv"]},
                         {"usable_roof_area_m2", "monthly_electricity_bill_gbp"})

        # A populated optional value leaves nothing to offer, and never gates.
        filled = {
            "common": {"technology_requested": "solar_pv"},
            "property": {"floor_area_m2": 78.0},
            "solar_pv": {"usable_roof_area_m2": 24.0,
                         "monthly_electricity_bill_gbp": 95},
        }
        self.assertEqual(optional_fields(filled), {})
        blank = {"common": {"technology_requested": "solar_pv"}}
        gating = {f["name"] for v in missing_fields(blank).values() for f in v}
        self.assertNotIn("floor_area_m2", gating)
        self.assertNotIn("usable_roof_area_m2", gating)

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

    def test_heat_pump_sizing_and_supply_fields(self):
        """Household size and supply questions gate; the fuse label only offers."""
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "heat_pump",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        gating = {f["name"] for f in body["missing_fields"]["heat_pump"]}
        for required_field in (
            "number_of_bedrooms", "number_of_bathrooms",
            "number_of_occupants", "smart_meter_installed",
        ):
            self.assertIn(required_field, gating)
        self.assertNotIn("smart_meter_cutout_fuse_label", gating)

        offered = {f["name"] for f in body["optional_fields"]["heat_pump"]}
        self.assertEqual(offered, {"smart_meter_cutout_fuse_label"})

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


class TestEvalCases(unittest.TestCase):
    """The committed evaluation cases must stay in step with FIELDS.

    Promoting a field to `required: True` without backfilling the cases drops
    completeness for reasons that have nothing to do with the model. Catch that
    here rather than in a scored run.
    """

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / "rfq_cases_real_v1.json"
        cls.cases = json.loads(path.read_text(encoding="utf-8"))

    def test_every_case_complete_except_the_deliberate_one(self):
        from epc_to_rfq import completeness_score

        incomplete = {
            c["case_id"]: round(completeness_score(c["input"])["score"], 2)
            for c in self.cases
            if completeness_score(c["input"])["score"] < 1.0
        }
        # REAL_021 omits number_of_occupants on purpose, so the completeness
        # metric and the missing_fields path both get exercised.
        self.assertEqual(incomplete, {"REAL_021": 0.93})

    def test_form_answers_are_valid_options(self):
        from epc_to_rfq import FIELDS, _TECH_SECTIONS

        # `property` is auto-filled from the EPC in EPC vocabulary rather than
        # form vocabulary, so only the form-answered sections are checkable.
        bad = []
        for case in self.cases:
            tech = case["input"]["common"]["technology_requested"]
            for section in ("common", _TECH_SECTIONS[tech]):
                for name, value in (case["input"].get(section) or {}).items():
                    meta = FIELDS.get(section, {}).get(name)
                    if not meta or meta["type"] != "select" or not meta["options"]:
                        continue
                    if value not in meta["options"]:
                        bad.append(f"{case['case_id']}.{section}.{name}={value!r}")
        self.assertEqual(bad, [])

    def test_cases_match_the_generator_config(self):
        """Hand-edited cases must not drift from build_cases.py, or the next
        rebuild silently reverts them."""
        import ast

        from epc_to_rfq import _TECH_SECTIONS

        src = (Path(__file__).parent / "evaluation" / "build_cases.py").read_text(
            encoding="utf-8"
        )
        marker = "CONFIG: list[dict] = "
        start = src.index(marker)
        config = ast.literal_eval(src[start + len(marker):src.index("\n]\n", start) + 2])
        by_id = {c["case_id"]: c for c in config}

        self.assertEqual({c["case_id"] for c in self.cases}, set(by_id))
        drift = []
        for case in self.cases:
            section = _TECH_SECTIONS[case["input"]["common"]["technology_requested"]]
            form = by_id[case["case_id"]]["form"].get(section, {})
            built = case["input"][section]
            for name in set(form) | set(built):
                if form.get(name) != built.get(name):
                    drift.append(
                        f"{case['case_id']}.{section}.{name}: "
                        f"config={form.get(name)!r} case={built.get(name)!r}"
                    )
        self.assertEqual(drift, [])


class TestCostEstimate(unittest.TestCase):
    """Renbee cost tables and band matching. Pure functions, no HTTP.

    The fabrication judge receives rfq_input as "the only facts that are true",
    so a mistyped price reads to it as supported. Transcription correctness has
    to be pinned here, the same lesson as the £400,014,000 cost bug.
    """

    def test_solar_pv_table_transcribed_exactly(self):
        from cost_tables import SOLAR_PV_ROWS

        self.assertEqual(
            [(r["kw"], r["panels"], r["cost"], r["saving"], r["band"])
             for r in SOLAR_PV_ROWS],
            [
                (3, (7, 8), (5000, 6500), (350, 550), "1-2 bed flat/terrace"),
                (4, (10, 10), (6000, 8000), (500, 800), "3 bed semi-detached"),
                (5, (12, 13), (7000, 9500), (650, 950), "3-4 bed detached"),
                (6, (15, 15), (8000, 11000), (800, 1200), "4-5 bed detached"),
            ],
        )

    def test_heat_pump_table_transcribed_exactly(self):
        from cost_tables import HEAT_PUMP_LARGE_ROW, HEAT_PUMP_ROWS

        rows = HEAT_PUMP_ROWS + [HEAT_PUMP_LARGE_ROW]
        self.assertEqual(
            [(r["band"], r["ashp"], r["gshp"], r["net"]) for r in rows],
            [
                ("1-2 bed flat/terrace", (8000, 10000), (18000, 22000), (500, 2500)),
                ("3-4 bed detached", (11000, 15000), (22000, 30000), (3500, 7500)),
                ("Large period property", (14000, 18000), (28000, 35000), (6500, 10500)),
            ],
        )

    def test_net_cost_is_air_source_minus_the_grant(self):
        """A transcription check, not a derivation: if this breaks, one of the
        two columns was typed wrong."""
        from cost_tables import BUS_GRANT_GBP, HEAT_PUMP_LARGE_ROW, HEAT_PUMP_ROWS

        for row in HEAT_PUMP_ROWS + [HEAT_PUMP_LARGE_ROW]:
            self.assertEqual(row["net"][0], row["ashp"][0] - BUS_GRANT_GBP, row["band"])
            self.assertEqual(row["net"][1], row["ashp"][1] - BUS_GRANT_GBP, row["band"])

    def test_property_type_beats_built_form_for_flats(self):
        """EPC built-form describes the building, not the dwelling: 160 of the
        355 flat certificates in output/ carry a non-terrace built form. This
        is the REAL_017 case (mid-floor flat in a detached block)."""
        from cost_tables import classify_style

        self.assertEqual(classify_style("mid-floor_flat", "detached"), "flat")
        self.assertEqual(classify_style("ground-floor_maisonette", "semi-detached"), "flat")
        self.assertEqual(classify_style("top-floor_flat", "mid-terrace"), "flat")

    def test_classify_style_handles_the_real_vocabulary(self):
        from cost_tables import classify_style

        cases = {
            # Inconsistent separators come straight out of the EPC data.
            ("enclosed-mid-terrace_house", None): "terrace",
            ("enclosed_end-terrace_house", None): "terrace",
            ("mid-terrace_house", None): "terrace",
            ("terraced", None): "terrace",
            ("semi-detached_house", None): "semi",
            ("detached_house", None): "detached",
            # Bungalows fall out of the same tests.
            ("semi-detached_bungalow", None): "semi",
            ("detached_bungalow", None): "detached",
            # property_type names no form, so built_form is consulted.
            ("bungalow", "detached"): "detached",
            (None, "end-terrace"): "terrace",
        }
        for (ptype, bform), expected in cases.items():
            self.assertEqual(classify_style(ptype, bform), expected, (ptype, bform))

        self.assertIsNone(classify_style(None, "not_recorded"))
        self.assertIsNone(classify_style(None, None))

    def _estimate(self, technology, beds, **prop):
        from cost_tables import build_cost_estimate

        return build_cost_estimate({
            "common": {"technology_requested": technology},
            "property": prop,
            technology: {"number_of_bedrooms": beds},
        })

    def test_nearest_band_resolutions(self):
        """The combinations the tables do not cover, pinned for both tables."""
        cases = [
            # (property_type, beds, expected solar band, expected heat pump band)
            ("mid-terrace_house", 3, "3 bed semi-detached", "3-4 bed detached"),
            ("semi-detached_house", 4, "3-4 bed detached", "3-4 bed detached"),
            ("top-floor_flat", 5, "4-5 bed detached", "3-4 bed detached"),
            ("detached_house", 2, "1-2 bed flat/terrace", "1-2 bed flat/terrace"),
        ]
        for ptype, beds, solar_band, hp_band in cases:
            solar = self._estimate("solar_pv", beds, property_type=ptype)
            self.assertEqual(solar["matched_band"], solar_band, (ptype, beds))
            self.assertEqual(solar["match_type"], "nearest_band", (ptype, beds))

            hp = self._estimate("heat_pump", beds, property_type=ptype)
            self.assertEqual(hp["matched_band"], hp_band, (ptype, beds))
            self.assertEqual(hp["match_type"], "nearest_band", (ptype, beds))

    def test_four_bed_detached_overlap_resolves_to_the_cheaper_row(self):
        """4 beds detached matches both "3-4 bed detached" and "4-5 bed
        detached" exactly, so the tie-break must be pinned rather than left to
        list order."""
        solar = self._estimate("solar_pv", 4, property_type="detached_house")
        self.assertEqual(solar["matched_band"], "3-4 bed detached")
        self.assertEqual(solar["match_type"], "exact")
        self.assertEqual(solar["system_size_kw"], 5)

    def test_exact_matches_carry_no_hedge(self):
        solar = self._estimate("solar_pv", 3, property_type="semi-detached_house")
        self.assertEqual(solar["matched_band"], "3 bed semi-detached")
        self.assertEqual(solar["match_type"], "exact")
        self.assertNotIn("match_note", solar)

        hp = self._estimate("heat_pump", 2, property_type="mid-terrace_house")
        self.assertEqual(hp["matched_band"], "1-2 bed flat/terrace")
        self.assertEqual(hp["match_type"], "exact")
        self.assertNotIn("match_note", hp)

    def test_large_period_property_rule(self):
        from cost_tables import is_large_period

        # REAL_030: 152 m², before 1900, semi-detached, 4 beds.
        self.assertTrue(is_large_period("semi", 4, 152.0, "before 1900"))
        # Pins the 150 m² threshold.
        self.assertFalse(is_large_period("semi", 4, 149.9, "before 1900"))
        # REAL_011: large but not period.
        self.assertFalse(is_large_period("detached", 4, 160.0, "1967-1975"))
        # Out-of-vocabulary bands fail closed.
        self.assertFalse(is_large_period("detached", 4, 200.0, "2007-2011"))
        self.assertFalse(is_large_period("detached", 4, 200.0, None))
        # REAL_023 regression: a maisonette whose EPC recorded the whole
        # building's 436 m² must not price as a mansion.
        self.assertFalse(is_large_period("flat", 3, 436.0, "before 1900"))
        # Bedrooms alone can qualify it.
        self.assertTrue(is_large_period("detached", 5, 90.0, "1900-1929"))

    def test_large_period_reaches_the_estimate(self):
        hp = self._estimate("heat_pump", 4, property_type="semi-detached_house",
                            floor_area_m2=152.0, construction_age_band="before 1900")
        self.assertEqual(hp["matched_band"], "Large period property")
        self.assertEqual(hp["air_source_cost_low_gbp"], 14000)
        self.assertEqual(hp["ground_source_cost_high_gbp"], 35000)

    def test_no_estimate_without_a_table_or_a_bedroom_count(self):
        for technology in ("battery", "solar_thermal"):
            self.assertIsNone(self._estimate(technology, 3, property_type="detached_house"))
        for beds in (None, "", "abc", -1):
            self.assertIsNone(
                self._estimate("solar_pv", beds, property_type="detached_house"), beds)

    def test_bedrooms_are_read_from_the_requested_technology_only(self):
        """A battery enquiry carrying a stray heat_pump dict must not produce a
        heat pump price."""
        from cost_tables import build_cost_estimate

        self.assertIsNone(build_cost_estimate({
            "common": {"technology_requested": "battery"},
            "property": {"property_type": "detached_house"},
            "heat_pump": {"number_of_bedrooms": 4},
        }))

    def test_string_bedrooms_behave_like_ints(self):
        """The demo builds additional_fields from FormData, so every answer
        arrives as a string while the tests and eval cases send ints."""
        as_int = self._estimate("solar_pv", 3, property_type="semi-detached_house")
        as_str = self._estimate("solar_pv", "3", property_type="semi-detached_house")
        self.assertEqual(as_int, as_str)

    def test_no_falsy_values_in_the_output(self):
        """_strip_nulls keeps False and 0, and a small model narrates
        `approximate: false` as "this is an exact match"."""
        for estimate in (
            self._estimate("solar_pv", 3, property_type="semi-detached_house"),
            self._estimate("heat_pump", 9, property_type="detached_house"),
        ):
            for key, value in estimate.items():
                self.assertNotIsInstance(value, bool, key)
                self.assertTrue(value, key)

    def test_guide_price_caveat_is_carried_as_data(self):
        """The prompt requires the cost block to close with this caveat, so the
        field has to exist or the fabrication judge scores a prompt-mandated
        sentence as unsupported. In the first scored run that one sentence was
        41 of 51 recommendation fabrication flags."""
        from cost_tables import GUIDE_PRICE_NOTE

        for tech, beds in (("solar_pv", 3), ("heat_pump", 4)):
            est = self._estimate(tech, beds, property_type="detached_house")
            self.assertEqual(est["guide_price_note"], GUIDE_PRICE_NOTE)

        # The prompt must quote the field, not paraphrase it, or the sentence
        # the model writes will not be the sentence the data supports.
        import generate_rfq
        self.assertIn("cost_estimate.guide_price_note",
                      generate_rfq.RECOMMENDATION_SYSTEM_PROMPT)
        # Twice per worked example: once in the input excerpt, once in the
        # example output the model is shown.
        self.assertEqual(
            generate_rfq.RECOMMENDATION_SYSTEM_PROMPT.count(GUIDE_PRICE_NOTE), 4,
            "both worked examples and their input excerpts must use the exact note",
        )

    def test_studio_flat_is_not_clamped(self):
        """0 bedrooms is legitimate and must resolve, not be rounded up."""
        solar = self._estimate("solar_pv", 0, property_type="top-floor_flat")
        self.assertEqual(solar["matched_band"], "1-2 bed flat/terrace")
        self.assertEqual(solar["match_type"], "nearest_band")

    def test_cost_estimate_never_reaches_the_installer_prompt(self):
        """The installer sets their own price; quoting homeowner guide prices
        back at them is wrong and would move the RFQ fabrication denominator."""
        from generate_rfq import build_rfq_prompt

        rfq_input = {
            "common": {"technology_requested": "heat_pump", "postcode": "E1 6AN"},
            "property": {"property_type": "detached_house"},
            "heat_pump": {"number_of_bedrooms": 4},
            "cost_estimate": self._estimate("heat_pump", 4,
                                            property_type="detached_house"),
        }
        message = build_rfq_prompt(rfq_input)[1]["content"]
        self.assertNotIn("cost_estimate", message)
        self.assertNotIn("11,000", message)
        self.assertNotIn("Boiler Upgrade Scheme", message)

    def test_cost_estimate_reaches_the_recommendation_prompt(self):
        from generate_rfq import build_recommendation_prompt

        rfq_input = {
            "common": {"technology_requested": "solar_pv"},
            "property": {"property_type": "semi-detached_house", "epc_rating": "D"},
            "solar_pv": {"number_of_bedrooms": 3},
            "recommendation": {"raw_recommendation_items": ["Loft insulation"]},
            "cost_estimate": self._estimate("solar_pv", 3,
                                            property_type="semi-detached_house"),
        }
        message = build_recommendation_prompt(rfq_input)[1]["content"]
        self.assertIn("cost_estimate", message)
        self.assertIn("3 bed semi-detached", message)


class TestCostEstimateWiring(MockedAPITestCase):
    """The /api/generate path: derivation, staleness, injection, disclaimer."""

    def _run(self, technology, extra, common_extra=None):
        sid = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": technology,
        }).json()["session_id"]
        common = {
            "preferred_contact_method": "email",
            "contact_email": "a@example.com",
            "contact_phone": "07700 900000",
            "desired_installation_timeline": "flexible",
            "motivation": "cut bills",
        }
        common.update(common_extra or {})
        r = self.client.post("/api/generate", json={
            "session_id": sid,
            "additional_fields": {"common": common, technology: extra},
        })
        return r

    HEAT_PUMP = {
        "heat_pump_type_interest": "ground_air_source", "emitter_type": "radiators",
        "hot_water_cylinder_space_available": "yes", "external_unit_space": "yes",
        "garden_or_side_access": "yes", "number_of_bedrooms": 4,
        "number_of_bathrooms": 2, "number_of_occupants": 4,
        "smart_meter_installed": "yes",
    }
    SOLAR_PV = {
        "roof_orientation": "south", "roof_shading_level": "low",
        "roof_condition": "yes", "number_of_bedrooms": 3,
        "number_of_bathrooms": 2, "number_of_occupants": 4,
    }
    BATTERY = {
        "existing_solar_pv": "none", "battery_purpose": "both",
        "backup_power_required": "yes", "battery_space_available": "yes",
        "battery_location_preference": "garage",
    }

    def test_heat_pump_enquiry_gets_a_band(self):
        r = self._run("heat_pump", self.HEAT_PUMP)
        self.assertEqual(r.status_code, 200, r.text)
        est = r.json()["rfq_input"]["cost_estimate"]
        self.assertEqual(est["technology"], "heat_pump")
        self.assertEqual(est["air_source_cost_low_gbp"], 11000)
        self.assertEqual(est["net_cost_after_grant_low_gbp"], 3500)

    def test_battery_enquiry_gets_no_band(self):
        r = self._run("battery", self.BATTERY)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertNotIn("cost_estimate", r.json()["rfq_input"])

    def test_disclaimer_names_the_sources_actually_used(self):
        from generate_rfq import (RECOMMENDATION_DISCLAIMER,
                                  RECOMMENDATION_DISCLAIMER_WITH_COSTS)

        with_costs = self._run("solar_pv", self.SOLAR_PV).json()
        self.assertEqual(with_costs["recommendation_disclaimer"],
                         RECOMMENDATION_DISCLAIMER_WITH_COSTS)
        self.assertIn("Renbee", with_costs["recommendation_disclaimer"])

        epc_only = self._run("battery", self.BATTERY).json()
        self.assertEqual(epc_only["recommendation_disclaimer"],
                         RECOMMENDATION_DISCLAIMER)

    def test_bedrooms_posted_as_strings(self):
        """The demo builds additional_fields from FormData, so every numeric
        answer arrives as a string. Without coercion this 500s in production
        while every int-based test stays green."""
        as_strings = {k: (str(v) if isinstance(v, int) else v)
                      for k, v in self.HEAT_PUMP.items()}
        r = self._run("heat_pump", as_strings)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["rfq_input"]["cost_estimate"]["matched_band"],
                         "3-4 bed detached")

    def test_a_second_call_does_not_leave_a_stale_band(self):
        sid = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "heat_pump",
        }).json()["session_id"]
        common = {
            "preferred_contact_method": "email", "contact_email": "a@example.com",
            "contact_phone": "07700 900000",
            "desired_installation_timeline": "flexible", "motivation": "cut bills",
        }
        bands = []
        for beds in (4, 1):
            body = self.client.post("/api/generate", json={
                "session_id": sid,
                "additional_fields": {
                    "common": common,
                    "heat_pump": {**self.HEAT_PUMP, "number_of_bedrooms": beds},
                },
            }).json()
            bands.append(body["rfq_input"]["cost_estimate"]["matched_band"])
        self.assertEqual(bands, ["3-4 bed detached", "1-2 bed flat/terrace"])

    def test_client_cannot_inject_a_cost_estimate(self):
        sid = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "heat_pump",
        }).json()["session_id"]
        body = self.client.post("/api/generate", json={
            "session_id": sid,
            "additional_fields": {
                "common": {
                    "preferred_contact_method": "email",
                    "contact_email": "a@example.com",
                    "contact_phone": "07700 900000",
                    "desired_installation_timeline": "flexible",
                    "motivation": "cut bills",
                },
                "heat_pump": self.HEAT_PUMP,
                "cost_estimate": {"air_source_cost_low_gbp": 1},
            },
        }).json()
        self.assertEqual(body["rfq_input"]["cost_estimate"]["air_source_cost_low_gbp"],
                         11000)

    def test_solar_pv_household_fields_are_asked(self):
        r = self.client.post("/api/initiate", json={
            "postcode": "E1 6AN", "technology": "solar_pv",
        })
        gating = {f["name"] for f in r.json()["missing_fields"]["solar_pv"]}
        for name in ("number_of_bedrooms", "number_of_bathrooms", "number_of_occupants"):
            self.assertIn(name, gating)


if __name__ == "__main__":
    unittest.main(verbosity=2)
