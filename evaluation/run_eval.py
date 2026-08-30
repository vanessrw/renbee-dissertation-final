"""Evaluation orchestrator. Generates the two outputs for each case with the
hosted Llama model, scores them, and writes the results.

Outputs (under --out, default eval_outputs/):
  generated.json   raw generated outputs (cached so the slow step runs once)
  scores.json      per-case, per-repeat metric breakdown + run metadata
  summary.csv      aggregate table ready to quote in Chapter IV
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from epc_to_rfq import completeness_score
from evaluation import faithfulness, rubric
from evaluation.gemini_client import GeminiClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CASES = ROOT / "rfq_cases_real_v1.json"
DEFAULT_OUT = ROOT / "eval_outputs"


# --------------------------------------------------------------------------
# Generation (hosted model, or mock)
# --------------------------------------------------------------------------

def _mock_generate(rfq_input: dict) -> dict:
    """Deterministic stand-in generator: weaves the real field values into prose
    so the scoring loop (especially preservation) can be exercised offline."""
    common = rfq_input.get("common") or {}
    prop = rfq_input.get("property") or {}
    tech = common.get("technology_requested", "low-carbon technology")
    tech_data = rfq_input.get(tech) or {}
    rec = rfq_input.get("recommendation") or {}

    def render(v):
        return str(v).replace("_", " ")

    parts = [
        f"The homeowner is requesting a {render(tech)} quotation."
    ]
    for k, v in prop.items():
        if k in faithfulness._EXCLUDE_FIELDS or v in (None, "", []):
            continue
        parts.append(f"The {k.replace('_',' ')} is {render(v)}.")
    for k, v in tech_data.items():
        if v in (None, "", []):
            continue
        parts.append(f"For installation, {k.replace('_',' ')} is {render(v)}.")
    for k, v in common.items():
        if k in faithfulness._EXCLUDE_FIELDS or v in (None, "", []):
            continue
        parts.append(f"{k.replace('_',' ').capitalize()}: {render(v)}.")
    rfq_summary = " ".join(parts) + " [mock generator]"

    rating = prop.get("epc_rating", "not specified")
    items = rec.get("raw_recommendation_items") or []
    if items:
        rec_summary = (
            f"Your home has an EPC rating of {rating}. Suggested improvements "
            f"include: {', '.join(render(i) for i in items)}. [mock generator]"
        )
    else:
        rec_summary = (
            "Your home does not have EPC improvement recommendations on record. "
            "[mock generator]"
        )
    return {"rfq_summary": rfq_summary, "recommendation_summary": rec_summary}


class Generator:
    """Wraps either the mock generator or the hosted model."""

    def __init__(self, mock: bool):
        self.mock = mock
        if mock:
            self.label = "mock"
        else:
            import generate_rfq
            # The label is provenance: it keys the generation cache and is
            # written into scores.json metadata, so it must name what actually
            # produced the text. Keep this format stable — changing it
            # invalidates every cached generation from a previous paid run.
            self.label = f"cloud:{os.getenv('LLM_MODEL') or generate_rfq.MODEL_NAME}"

    def generate(self, rfq_input: dict) -> dict:
        if self.mock:
            return _mock_generate(rfq_input)
        import generate_rfq as gen
        rfq = gen.generate_rfq_summary(rfq_input)
        rec = gen.generate_recommendation(rfq_input)
        return {
            "rfq_summary": rfq.get("rfq_summary", ""),
            "recommendation_summary": rec.get("recommendation_summary", ""),
            "rfq_parse_status": rfq.get("parse_status"),
            "rec_parse_status": rec.get("parse_status"),
        }


# --------------------------------------------------------------------------
# Quality judging (RQ2 / RQ3)
# --------------------------------------------------------------------------

def _coerce_scores(res: dict, criteria: dict) -> dict:
    out = {}
    for k in criteria:
        try:
            v = int(round(float(res.get(k))))
        except (TypeError, ValueError):
            v = None
        out[k] = max(1, min(5, v)) if v is not None else None
    # Per-criterion reasons (rubric v2). Tolerate a judge that still emits the
    # v1 single `justification` string rather than dropping the text entirely.
    reasons = res.get("reasons")
    if not isinstance(reasons, dict):
        legacy = res.get("justification", "")
        reasons = {k: legacy for k in criteria} if legacy else {}
    out["reasons"] = {k: str(reasons.get(k, "")) for k in criteria}
    return out


def judge_quality(client: GeminiClient, output_text: str, output_type: str,
                  technology: str) -> dict:
    if output_type == "rfq":
        system = rubric.RFQ_JUDGE_SYSTEM
        user = rubric.rfq_judge_user(output_text, technology)
        criteria = rubric.RFQ_CRITERIA
    else:
        system = rubric.RECOMMENDATION_JUDGE_SYSTEM
        user = rubric.recommendation_judge_user(output_text)
        criteria = rubric.RECOMMENDATION_CRITERIA
    res = client.judge_json(system, user,
                            mock_response=lambda: rubric.mock_quality_response(criteria))
    return _coerce_scores(res, criteria)


# --------------------------------------------------------------------------
# Caching of generated outputs
# --------------------------------------------------------------------------

def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _gen_outputs(cases: list[dict], gen: Generator, repeats: int,
                 cache_path: Path, regenerate: bool) -> dict:
    cache = {} if regenerate else _load_cache(cache_path)
    cache.setdefault("model_label", gen.label)
    cache.setdefault("items", {})
    if cache.get("model_label") != gen.label:
        # Different model than cached: start fresh to avoid mixing.
        cache = {"model_label": gen.label, "items": {}}
    items = cache["items"]
    for case in cases:
        cid = case["case_id"]
        for r in range(repeats):
            key = f"{gen.label}::{cid}::{r}"
            if key in items:
                continue
            print(f"  generating {key} ...")
            items[key] = gen.generate(case["input"])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    return cache


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_output(client: GeminiClient, rfq_input: dict, generated: dict) -> dict:
    tech = (rfq_input.get("common") or {}).get("technology_requested", "")
    rfq_text = generated.get("rfq_summary", "")
    rec_text = generated.get("recommendation_summary", "")

    rfq_block = {
        "preservation": faithfulness.preservation_score(rfq_input, rfq_text, "rfq"),
        "site_context": faithfulness.site_context_coverage(rfq_input, rfq_text),
        "fabrication": faithfulness.fabrication_score(client, rfq_input, rfq_text),
        "quality": judge_quality(client, rfq_text, "rfq", tech),
    }
    rec_block = {
        "preservation": faithfulness.preservation_score(rfq_input, rec_text, "recommendation"),
        "fabrication": faithfulness.fabrication_score(client, rfq_input, rec_text),
        "quality": judge_quality(client, rec_text, "recommendation", tech),
    }
    return {"rfq": rfq_block, "recommendation": rec_block}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _summarise_metric(per_case_values: list[list[float]]) -> dict:
    """Reduce each case to its mean, then report dispersion two ways:
      std         spread across the case means (`n` = number of cases).
      std_within  mean spread across repeats of a case; None at repeats=1.
    At repeats=1 this equals the old behaviour (std over the cases)."""
    case_means: list[float] = []
    within_stds: list[float] = []
    n_obs = 0
    for vals in per_case_values:
        clean = [v for v in vals if v is not None]
        if not clean:
            continue
        n_obs += len(clean)
        case_means.append(statistics.mean(clean))
        if len(clean) > 1:
            within_stds.append(statistics.pstdev(clean))
    if not case_means:
        return {"mean": None, "std": None, "std_within": None, "n": 0, "n_obs": 0}
    return {
        "mean": round(statistics.mean(case_means), 3),
        "std": round(statistics.pstdev(case_means), 3) if len(case_means) > 1 else 0.0,
        "std_within": round(statistics.mean(within_stds), 3) if within_stds else None,
        "n": len(case_means),   # number of cases the between-case std spans
        "n_obs": n_obs,         # total scored observations (cases x repeats)
    }


def aggregate(case_results: list[dict]) -> dict:
    # metric -> list (one per case) of lists (one value per repeat)
    buckets: dict[str, list[list[float]]] = {}

    def push(metric, case_values):
        buckets.setdefault(metric, []).append(case_values)

    for cr in case_results:
        # completeness is a single per-case value (no repeats).
        push("completeness.score", [cr["completeness"]["score"]])

        # one entry per case, holding that case's per-repeat values
        per_case: dict[str, list[float]] = {}

        def add(metric, value):
            per_case.setdefault(metric, []).append(value)

        for rep in cr["repeats"]:
            r = rep["scores"]
            add("rfq.preservation_rate", r["rfq"]["preservation"]["preservation_rate"])
            add("rfq.fabrication_rate", r["rfq"]["fabrication"]["fabrication_rate"])
            add("rfq.fabricated_statements", r["rfq"]["fabrication"].get("fabricated_statements", 0))
            add("rfq.total_statements", r["rfq"]["fabrication"].get("total_statements", 0))
            for k in rubric.RFQ_CRITERIA:
                add(f"rfq.{k}", r["rfq"]["quality"].get(k))
            if r["rfq"]["site_context"]:
                add("rfq.site_context_coverage", r["rfq"]["site_context"]["coverage_rate"])
            add("recommendation.preservation_rate", r["recommendation"]["preservation"]["preservation_rate"])
            add("recommendation.fabrication_rate", r["recommendation"]["fabrication"]["fabrication_rate"])
            add("recommendation.fabricated_statements", r["recommendation"]["fabrication"].get("fabricated_statements", 0))
            add("recommendation.total_statements", r["recommendation"]["fabrication"].get("total_statements", 0))
            for k in rubric.RECOMMENDATION_CRITERIA:
                add(f"recommendation.{k}", r["recommendation"]["quality"].get(k))

        for metric, vals in per_case.items():
            push(metric, vals)

    return {metric: _summarise_metric(per_case_values)
            for metric, per_case_values in buckets.items()}


def write_summary_csv(aggregates: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    order = [
        "completeness.score",
        "rfq.preservation_rate",
        "rfq.fabrication_rate", "rfq.fabricated_statements", "rfq.total_statements",
        "rfq.site_context_coverage",
        "rfq.clarity", "rfq.usability", "rfq.perceived_efficiency",
        "recommendation.preservation_rate",
        "recommendation.fabrication_rate", "recommendation.fabricated_statements",
        "recommendation.total_statements",
        "recommendation.clarity", "recommendation.understandability",
        "recommendation.perceived_helpfulness",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std_between_case", "std_within_case",
                    "n_cases", "n_obs"])
        for m in order:
            if m in aggregates:
                a = aggregates[m]
                w.writerow([m, a["mean"], a["std"], a.get("std_within"),
                            a["n"], a.get("n_obs")])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _print_aggregates(aggregates: dict) -> None:
    print("\n=== AGGREGATE SUMMARY ===")
    print("  (std = between-case; within = mean spread across repeats of a case)")
    for m, a in aggregates.items():
        within = a.get("std_within")
        within_s = "  within=--" if within is None else f"  within={within}"
        n_obs = a.get("n_obs", a["n"])
        print(f"  {m:42} mean={a['mean']}  std={a['std']}{within_s}"
              f"  n_cases={a['n']}  n_obs={n_obs}")


def reaggregate(scores_path: Path) -> None:
    """Recompute aggregates from an existing scores.json in place (no generation
    or judging) and rewrite scores.json + summary.csv."""
    results = json.loads(scores_path.read_text())
    if "cases" not in results:
        raise SystemExit(f"{scores_path} has no 'cases' block to re-aggregate.")
    results["aggregates"] = aggregate(results["cases"])
    results.setdefault("metadata", {})["reaggregated_at"] = (
        datetime.now(timezone.utc).isoformat()
    )
    scores_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_summary_csv(results["aggregates"], scores_path.parent / "summary.csv")
    print(f"Re-aggregated {scores_path} and {scores_path.parent/'summary.csv'} "
          f"(no re-generation or re-judging).")
    _print_aggregates(results["aggregates"])


def main():
    ap = argparse.ArgumentParser(description="RFQ evaluation harness")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--mock-gen", action="store_true", help="use the mock generator (no network)")
    ap.add_argument("--mock-judge", action="store_true", help="use the mock judge (no API key)")
    ap.add_argument("--regenerate", action="store_true", help="ignore cached generations")
    ap.add_argument("--reaggregate", metavar="SCORES_JSON",
                    help="recompute aggregates from an existing scores.json "
                         "(no generation or judging) and rewrite it + summary.csv")
    args = ap.parse_args()

    if args.reaggregate:
        reaggregate(Path(args.reaggregate))
        return

    out_dir = Path(args.out)
    cases = json.loads(Path(args.cases).read_text())
    gen = Generator(mock=args.mock_gen)
    client = GeminiClient(mock=args.mock_judge)

    print(f"Cases: {len(cases)} | generator: {gen.label} | judge: {client.backend} "
          f"| repeats: {args.repeats}")

    print("Generating outputs...")
    cache = _gen_outputs(cases, gen, args.repeats, out_dir / "generated.json", args.regenerate)
    items = cache["items"]

    print("Scoring...")
    case_results = []
    for case in cases:
        cid = case["case_id"]
        rfq_input = case["input"]
        comp = completeness_score(rfq_input)
        reps = []
        for r in range(args.repeats):
            generated = items[f"{gen.label}::{cid}::{r}"]
            scores = score_output(client, rfq_input, generated)
            reps.append({"repeat": r, "generated": generated, "scores": scores})
        case_results.append({
            "case_id": cid,
            "technology": (rfq_input.get("common") or {}).get("technology_requested"),
            "completeness": comp,
            "repeats": reps,
        })

    aggregates = aggregate(case_results)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": gen.label,
        "judge": client.backend,
        # Which rubric wording produced these quality scores. Runs under
        # different rubric versions are not comparable.
        "rubric_version": rubric.RUBRIC_VERSION,
        "repeats": args.repeats,
        "num_cases": len(cases),
        "cases_file": str(args.cases),
    }
    results = {"metadata": metadata, "aggregates": aggregates, "cases": case_results}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_summary_csv(aggregates, out_dir / "summary.csv")

    print(f"\nWrote {out_dir/'scores.json'} and {out_dir/'summary.csv'}")
    _print_aggregates(aggregates)


if __name__ == "__main__":
    main()
