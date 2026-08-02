"""Build a human-readable review of an evaluation run.

This is NOT a scoring step. It reads an existing eval_outputs/scores.json and
renders a single HTML page so you can eyeball how the two scoring methods behave
and where they agree or diverge:

  - Deterministic information preservation (objective token-overlap): which input
    fields were kept in the generated text and which were dropped, field by
    field.
  - LLM judge (Gemini): the fabrication statements it flagged and the 1-5 quality
    scores with the justification it wrote.

The two are shown side by side per output, with a divergence note that points
out cases worth a closer look (for example, the deterministic metric says
information was dropped while the judge still rated clarity highly, or the judge
flagged fabrication). No human ratings are involved.

Usage:
    python -m evaluation.review
    python -m evaluation.review --scores eval_outputs/scores.json --out eval_outputs/review.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SCORES = ROOT / "eval_outputs" / "scores.json"
DEFAULT_OUT = ROOT / "eval_outputs" / "review.html"


def _esc(v) -> str:
    return html.escape("" if v is None else str(v))


def _load_inputs(cases_file: str | None) -> dict:
    """Map case_id -> input dict, so the review can show postcode / site context.
    Best-effort: returns {} if the cases file is missing."""
    if not cases_file:
        return {}
    p = Path(cases_file)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {}
    try:
        cases = json.loads(p.read_text())
        return {c["case_id"]: c.get("input", {}) for c in cases}
    except Exception:
        return {}


def _pres_table(pres: dict) -> str:
    """Render a per-field preservation table from a preservation_score result."""
    rows = []
    for d in pres.get("details", []):
        cls = "ok" if d["preserved"] else "miss"
        mark = "kept" if d["preserved"] else "DROPPED"
        rows.append(
            f"<tr class='{cls}'><td>{_esc(d['field'])}</td>"
            f"<td>{_esc(d['value'])}</td><td>{mark}</td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='3'><em>no fields to preserve</em></td></tr>")
    rate = pres.get("preservation_rate")
    head = (f"<p class='metric'>Preservation rate "
            f"<b>{rate:.2f}</b> ({pres.get('preserved')}/{pres.get('total')} fields)</p>"
            if rate is not None else "")
    return head + (
        "<table class='detail'><tr><th>field</th><th>input value</th>"
        "<th>in output?</th></tr>" + "".join(rows) + "</table>"
    )


def _site_table(sc: dict | None) -> str:
    if not sc:
        return ""
    rows = []
    for d in sc.get("details", []):
        cls = "ok" if d["covered"] else "miss"
        mark = "mentioned" if d["covered"] else "OMITTED"
        rows.append(f"<tr class='{cls}'><td>{_esc(d['item'])}</td><td>{mark}</td></tr>")
    return (f"<p class='metric'>Site-context coverage <b>{sc['coverage_rate']:.2f}</b> "
            f"({sc['covered']}/{sc['total']})</p>"
            "<table class='detail'><tr><th>site-context item</th><th>in output?</th></tr>"
            + "".join(rows) + "</table>")


def _fab_block(fab: dict) -> str:
    rate = fab.get("fabrication_rate", 0.0)
    cls = "miss" if (fab.get("fabricated_statements") or 0) > 0 else "ok"
    items = fab.get("fabricated_list") or []
    li = "".join(f"<li>{_esc(x)}</li>" for x in items) or "<li><em>none flagged</em></li>"
    return (f"<p class='metric {cls}'>Fabrication rate <b>{rate:.2f}</b> "
            f"({fab.get('fabricated_statements')}/{fab.get('total_statements')} statements)</p>"
            f"<ul class='fab'>{li}</ul>")


def _quality_block(q: dict) -> str:
    """Render the 1-5 scores with the judge's per-criterion reason beside each.

    Rubric v2 returns `reasons` keyed by criterion. Older scores.json files
    carry a single `justification` string instead, so fall back to that and
    show it once below the table rather than losing it.
    """
    reasons = q.get("reasons") or {}
    rows = []
    for k, v in q.items():
        if k in ("justification", "reasons"):
            continue
        why = reasons.get(k, "")
        rows.append(
            f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td>"
            f"<td class='just'><em>{_esc(why)}</em></td></tr>"
        )
    legacy = "" if reasons else q.get("justification", "")
    return ("<table class='detail'>"
            "<tr><th>criterion</th><th>score (1-5)</th><th>why</th></tr>"
            + "".join(rows) + "</table>"
            + (f"<p class='just'><em>{_esc(legacy)}</em></p>" if legacy else ""))


def _divergence_note(pres: dict, fab: dict, quality: dict) -> str:
    """A short flag pointing out where the two methods may disagree."""
    notes = []
    rate = pres.get("preservation_rate")
    clarity = quality.get("clarity")
    if rate is not None and rate < 0.7 and isinstance(clarity, int) and clarity >= 4:
        notes.append(f"Deterministic preservation is low ({rate:.2f}) but the judge "
                     f"rated clarity {clarity}/5. Information may be dropped while the "
                     f"prose still reads well.")
    if (fab.get("fabricated_statements") or 0) > 0:
        notes.append(f"The judge flagged {fab['fabricated_statements']} fabricated "
                     f"statement(s). Check these against the input.")
    if rate is not None and rate >= 0.95 and (fab.get("fabricated_statements") or 0) == 0:
        notes.append("Both methods agree: information preserved and nothing fabricated.")
    if not notes:
        return ""
    return "<div class='note'>" + "<br>".join(notes) + "</div>"


def _output_section(title: str, text: str, scores: dict, output_type: str) -> str:
    pres = scores["preservation"]
    fab = scores["fabrication"]
    quality = scores["quality"]
    site = scores.get("site_context") if output_type == "rfq" else None
    return f"""
    <div class='output'>
      <h4>{_esc(title)}</h4>
      <div class='gentext'>{_esc(text) or '<em>(empty)</em>'}</div>
      {_divergence_note(pres, fab, quality)}
      <div class='cols'>
        <div class='col'>
          <h5>Deterministic (objective)</h5>
          {_pres_table(pres)}
          {_site_table(site)}
        </div>
        <div class='col'>
          <h5>LLM judge (Gemini)</h5>
          {_fab_block(fab)}
          {_quality_block(quality)}
        </div>
      </div>
    </div>"""


def _case_section(case: dict, inputs: dict) -> str:
    cid = case["case_id"]
    inp = inputs.get(cid, {})
    common = inp.get("common") or {}
    prop = inp.get("property") or {}
    sc = (inp.get("site_context") or {}).get("planning") or {}
    flags = [k for k, on in [
        ("listed", sc.get("listed_building")),
        ("conservation", sc.get("conservation_area_name")),
        ("article 4", sc.get("article_4")),
        ("AONB", sc.get("aonb_name")),
        ("WHS", sc.get("whs_name")),
        ("national park", sc.get("national_park_name")),
    ] if on]
    meta = (f"technology: <b>{_esc(case.get('technology'))}</b> &middot; "
            f"EPC rating: <b>{_esc(prop.get('epc_rating') or 'n/a')}</b> &middot; "
            f"completeness: <b>{case['completeness']['score']:.2f}</b> &middot; "
            f"site: <b>{_esc(', '.join(flags) or 'none')}</b>")
    blocks = []
    for rep in case["repeats"]:
        gen = rep["generated"]
        s = rep["scores"]
        if len(case["repeats"]) > 1:
            blocks.append(f"<h3>Repeat {rep['repeat']}</h3>")
        blocks.append(_output_section("Installer RFQ summary",
                                      gen.get("rfq_summary", ""), s["rfq"], "rfq"))
        blocks.append(_output_section("Homeowner recommendation summary",
                                      gen.get("recommendation_summary", ""),
                                      s["recommendation"], "recommendation"))
    return (f"<section class='case'><h2>{_esc(cid)}</h2>"
            f"<p class='meta'>{meta}</p>" + "".join(blocks) + "</section>")


def _aggregate_section(results: dict) -> str:
    agg = results.get("aggregates", {})
    rows = []
    for m, a in agg.items():
        within = a.get("std_within")
        rows.append(f"<tr><td>{_esc(m)}</td><td>{_esc(a['mean'])}</td>"
                    f"<td>{_esc(a['std'])}</td>"
                    f"<td>{'&ndash;' if within is None else _esc(within)}</td>"
                    f"<td>{_esc(a['n'])}</td>"
                    f"<td>{_esc(a.get('n_obs', a['n']))}</td></tr>")
    meta = results.get("metadata", {})
    return (f"<section class='agg'><h2>Run summary</h2>"
            f"<p class='meta'>generator: <b>{_esc(meta.get('generator'))}</b> &middot; "
            f"judge: <b>{_esc(meta.get('judge'))}</b> &middot; "
            f"cases: <b>{_esc(meta.get('num_cases'))}</b> &middot; "
            f"repeats: <b>{_esc(meta.get('repeats'))}</b> &middot; "
            f"generated: {_esc(meta.get('generated_at'))}</p>"
            "<p class='meta'>std = between-case dispersion (over n cases); "
            "within = mean spread across repeats of the same case.</p>"
            "<table class='detail'><tr><th>metric</th><th>mean</th>"
            "<th>std (between-case)</th><th>within-case</th>"
            "<th>n cases</th><th>n obs</th></tr>"
            + "".join(rows) + "</table></section>")


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  max-width:1100px;margin:24px auto;padding:0 16px;color:#1a1a1a;line-height:1.45}
h1{font-size:22px} h2{font-size:18px;border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:32px}
h4{margin:18px 0 6px;font-size:15px} h5{margin:8px 0;font-size:13px;color:#555}
.meta{color:#555;font-size:13px} .intro{background:#f6f8fa;padding:12px 16px;border-radius:8px;font-size:13px}
.case{margin-bottom:28px} .output{border:1px solid #e2e2e2;border-radius:8px;padding:12px 16px;margin:12px 0}
.gentext{background:#fafafa;border-left:3px solid #bbb;padding:8px 12px;margin:6px 0;
  font-size:13px;white-space:pre-wrap}
.cols{display:flex;gap:20px;flex-wrap:wrap} .col{flex:1;min-width:340px}
table.detail{border-collapse:collapse;width:100%;font-size:12px;margin:6px 0}
table.detail th,table.detail td{border:1px solid #e0e0e0;padding:3px 6px;text-align:left;vertical-align:top}
table.detail th{background:#f2f2f2}
tr.ok td{background:#f1fbf2} tr.miss td{background:#fdf1f1}
.metric{font-size:13px;margin:6px 0} .metric.ok{color:#1a7f37} .metric.miss{color:#b42318}
.note{background:#fff8e6;border:1px solid #f0d98a;border-radius:6px;padding:8px 12px;
  font-size:12px;margin:8px 0}
ul.fab{margin:4px 0 8px 18px;font-size:12px} .just{font-size:12px;color:#444;margin:4px 0}
"""


def build_html(results: dict, inputs: dict) -> str:
    cases_html = "".join(_case_section(c, inputs) for c in results.get("cases", []))
    intro = (
        "<div class='intro'><b>How to read this.</b> Each output is scored two ways. "
        "The left column is the <b>deterministic</b> method: an objective check of "
        "whether each input field survived into the text (green kept, red dropped). "
        "The right column is the <b>LLM judge</b>: the statements it flagged as "
        "fabricated and its 1 to 5 quality scores with justification. The amber note "
        "highlights where the two methods may disagree. No human ratings are used.</div>"
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>RFQ evaluation review</title><style>{_CSS}</style></head><body>
<h1>RFQ evaluation review</h1>
{intro}
{_aggregate_section(results)}
{cases_html}
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a review of an eval run.")
    ap.add_argument("--scores", default=str(DEFAULT_SCORES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise SystemExit(f"No scores file at {scores_path}. Run run_eval first.")
    results = json.loads(scores_path.read_text())
    inputs = _load_inputs(results.get("metadata", {}).get("cases_file"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(results, inputs), encoding="utf-8")
    print(f"Wrote review to {out_path}")
    print(f"Open it with:  open {out_path}")


if __name__ == "__main__":
    main()
