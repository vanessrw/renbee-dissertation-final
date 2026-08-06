# How to run

Commands only. For architecture see `CLAUDE.md`; for the file map see `README.md`.
Run everything from the `main/` directory with the venv activated.

## One-time setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # small: no torch, no model weights

cp .env.example .env                    # then fill in the tokens below
```

`.env` keys:

- `EPC_BEARER_TOKEN` — from <https://get-energy-performance-data.communities.gov.uk>.
  Required for any real EPC lookup.
- Gemini judge (only for a real scored eval): `GOOGLE_CLOUD_PROJECT`,
  `GOOGLE_APPLICATION_CREDENTIALS` (path to the Vertex service-account key),
  `GEMINI_MODEL=gemini-3.5-flash-lite`, `GOOGLE_CLOUD_LOCATION=global`.
  The location must be `global`: Flash-Lite 3.5 is not served from regional
  Vertex endpoints, so `us-central1` returns a 404.
- `UKPN_API_KEY` — no longer functional (the UKPN headroom dataset was retired), safe
  to omit.
- `LLM_API_BASE` — **required for any generation.** The OpenAI-compatible
  `/chat/completions` host. All generation is hosted; there is no local path.
  The configured target is **Llama 3.3 70B on Vertex**
  (`meta/llama-3.3-70b-instruct-maas`, `us-central1` only), which must be enabled
  once in Model Garden on the *API service* card, not the self-deploy one.
  `LLM_MODEL` overrides the model (it defaults to `MODEL_NAME` in
  `generate_rfq.py`); `LLM_API_KEY` is optional, and leaving it unset on Vertex
  makes auth fall back to Google ADC or the service-account key.
  `GET /health` reports the active target.

## Run the app

```bash
# mocked LLM: fast, no network, no credentials. Best for demos and screen recordings.
DEMO_MOCK_LLM=1 uvicorn app:app --port 8000

# real hosted generation (Llama 3.3 70B on Vertex).
uvicorn app:app --port 8000
```

Then open one of:

- <http://localhost:8000/demo> — the Webflow-style demo UI.
- <http://localhost:8000/docs> — Swagger UI for the API.

The demo asks for postcode + technology, shows the site-intelligence panel and the
dynamic form, then produces the recommendation and the installer RFQ. Both the RFQ prose
and the structured `rfq_input` are editable for human review before **Save RFQ** writes
the final JSON to `rfq_outputs/`.

## Sanity check

```bash
python test_api.py          # end-to-end smoke tests, ~0.1 s, EPC + LLM + external data mocked
```

## Quick data checks (no LLM)

```bash
python epc_fetch.py "E1 6AN"     # confirm the EPC token works; prints addresses + ratings
python pipeline.py "E1 6AN" 6    # full assembly + missing-fields diagnostics, no model
```

## Site intelligence (planning)

Every `/api/initiate` cross-references the postcode against planning.data.gov.uk and
surfaces the result as the "Site intelligence" panel. Test postcodes:

| Postcode | What you should see |
|---|---|
| `BA1 1LZ` | Conservation area (Bath) + World Heritage Site |
| `OX7 3EL` | Cotswolds AONB |
| `SW1A 2AA` | Westminster conservation area (renders even with 0 EPCs) |

Note: `site_context.grid` (UKPN headroom) is always `null` — the UKPN dataset the code
used has been retired. Planning constraints are unaffected.

## Proxy EPC fallbacks

When the homeowner's own EPC is missing, the pipeline aggregates other EPCs instead of
dropping straight to manual entry:

- **Layer 1** (same postcode, user picks "use street average"): re-call `/api/initiate`
  with `"use_proxy": true`. Tagged `epc_source: "proxy"`.
- **Layer 2** (input postcode has 0 EPCs): the server returns a **409
  `proxy_nearby_candidates`** picker. The client either picks one
  (`lmk_key` + `proxy_postcode`, tagged `proxy_picked: true`) or takes the average
  (`use_proxy_nearby_average: true`). Both tagged `epc_source: "proxy_nearby"`.
- **Case B**: if no nearby EPCs exist either, the 3-question manual form fires.

`test_api.py` covers all three paths with mocked postcodes.io. `GL54 2BP` (Layer 2) and
`CV12 8UE` (Layer 1) are handy live examples.

## Evaluation harness

`evaluation/run_eval.py` generates both LLM outputs for each case in
`rfq_cases_real_v1.json` (30 real-postcode cases), scores them with the Gemini judge, and
writes `eval_outputs/`.

```bash
# offline smoke test — no network, no credentials
python -m evaluation.run_eval --mock-gen --mock-judge

# full scored run (hosted Llama 3.3 70B + real Gemini judge)
python -m evaluation.run_eval --repeats 3 --regenerate
```

Use **`--regenerate`** whenever the prompts or parsing changed, so cached outputs from a
previous build are not reused.

Flags: `--cases <path>`, `--out <dir>` (default `eval_outputs/`), `--repeats N`,
`--mock-gen`, `--mock-judge`, `--regenerate`, and `--reaggregate <scores.json>` (recompute
the aggregate table from an existing run, no generation or judging).

Outputs (in `eval_outputs/`):

- `generated.json` — cached raw generations (the slow step runs once).
- `scores.json` — per-case, per-repeat metrics + run metadata.
- `summary.csv` — aggregate table: `mean`, `std_between_case`, `std_within_case`,
  `n_cases`, `n_obs`.

Then render the qualitative cross-check:

```bash
python -m evaluation.review && open eval_outputs/review.html
```

**Judge credentials**: `GeminiClient` uses Vertex AI when `GOOGLE_CLOUD_PROJECT` is set and
`GEMINI_API_KEY` is not. It reads the service-account key from
`GOOGLE_APPLICATION_CREDENTIALS`, or falls back to `gcloud auth application-default login`.
Use `--mock-judge` to sanity-check generation without any credentials.

## Troubleshooting

| Error / symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv: `source venv/bin/activate` |
| `EPC_BEARER_TOKEN must be set` | Add `EPC_BEARER_TOKEN=...` to `.env` |
| `Generation requires LLM_API_BASE` | Add `LLM_API_BASE=...` to `.env`, or use `DEMO_MOCK_LLM=1` |
| `LLM endpoint returned 404 ... does not have access` | Enable Llama MaaS in Vertex Model Garden on the *API service* card |
| `LLM endpoint returned 429` after all retries | Shared-capacity throttling — raise `LLM_MAX_RETRIES` or retry later |
| `409 ambiguous_address` | Multi-address postcode — re-call with an `lmk_key` from `candidates` |
| `409 proxy_nearby_candidates` | 0-EPC postcode — pick a candidate or pass `use_proxy_nearby_average: true` |
| `unknown_session` 404 | Sessions are in-memory and die on restart — re-run `/api/initiate` |
