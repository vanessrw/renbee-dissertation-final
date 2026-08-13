# Constrained LLM-Based RFQ Generation for Residential Decarbonisation

Production-shape codebase for the MSc thesis *"Design and Evaluation of Constrained
LLM-Based RFQ Generation in Residential Decarbonisation"* (Warwick), in partnership
with **Renbee**, a UK B2C decarbonisation installer directory. The pipeline turns a
homeowner's postcode and chosen technology into a structured, installer-ready Request
for Quote (RFQ), using real UK open data and a hosted Llama 3.3 70B model under
constrained prompting (no fine-tuning, no RAG).

Four technologies are supported: **heat pump**, **solar PV**, **battery storage**, and
**solar thermal**

## How it works

```
postcode + technology
      |
      v
  EPC API  +  planning.data.gov.uk        (real property + site data)
      |
      v
  assemble structured RFQ input  ->  missing_fields form (Webflow Step 2)
      |
      v
  two LLM moments (Llama 3.3 70B, constrained prompts):
    1. homeowner-facing EPC recommendation   (engagement)
    2. installer-facing RFQ summary           (HITL review, then sent)
```

When the homeowner's own EPC is unavailable, the pipeline falls back through two proxy
layers (same-postcode aggregate, then nearby-postcode aggregate or pick-one) before the
final manual-entry form. See `CLAUDE.md` for the full architecture.

## Repository structure

| Path | What it is |
|---|---|
| `epc_fetch.py` | Postcode to UK EPC API. Reads `EPC_BEARER_TOKEN`. |
| `external_data.py` | Postcode to planning constraints (listed / conservation / AONB / WHS / National Park / Article 4). Powers `site_context`. |
| `epc_to_rfq.py` | Assembles EPC + form answers into the RFQ schema. Owns `FIELDS`, `missing_fields()`, `completeness_score()`, proxy builders. |
| `generate_rfq.py` | Hosted Llama 3.3 70B + the two constrained system prompts (recommendation / RFQ). |
| `pipeline.py` | CLI orchestrator (assemble, generate, or both). |
| `app.py` | FastAPI server. `/api/initiate`, `/api/generate`, `/api/generate-rfq`, `/api/save-rfq`, `/demo`, `/health`. |
| `webflow_demo.html` | Standalone Webflow-style demo, served at `/demo`. |
| `test_api.py` | End-to-end smoke tests (EPC + LLM + external data mocked). |
| `evaluation/` | Evaluation harness (see below). |
| `notebooks/` | `renbee_demo_journey.ipynb`, the homeowner journey runnable on Vertex AI Workbench. Setup in `notebooks/README.md`. |
| `rfq_cases_real_v1.json` | 30 real-postcode evaluation cases (per-case breakdown in `evaluation/CASES.md`). |
| `CLAUDE.md` | Full architecture and design notes. |
| `HOW_TO_RUN.md` | Setup and run commands. |
| `TECHNICAL.md` | Extended technical write-up. |

`evaluation/`:

| File | Role |
|---|---|
| `build_cases.py` | Builds the real-postcode case set from live APIs. |
| `run_eval.py` | Generates both LLM outputs per case, scores them, writes `eval_outputs/`. |
| `faithfulness.py` | Deterministic preservation + site-context coverage; judge-based fabrication. |
| `rubric.py` | Judge prompts and the 1-5 quality criteria. |
| `gemini_client.py` | Gemini 3.5 Flash-Lite (Vertex AI) judge wrapper, with a mock mode. |
| `review.py` | Renders an HTML side-by-side of the deterministic metrics vs the judge. |

The quotable run is **`eval_outputs_cloud70b_v2/`**. `eval_outputs_cloud70b/` is superseded,
and `eval_outputs/` (local 3B, an older judge, the unanchored v1 rubric) is orphaned and not
comparable. See the run-provenance table in `CLAUDE.md`.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in EPC_BEARER_TOKEN etc.

# fastest way to see the flow (no network, no credentials):
DEMO_MOCK_LLM=1 uvicorn app:app --port 8000   # open http://localhost:8000/demo
```

Full setup and the evaluation commands are in [`HOW_TO_RUN.md`](HOW_TO_RUN.md).

## Generation

All generation is hosted. There is no local-weights path, so the repo needs neither
torch nor transformers, and installs in seconds.

| Mode | Set | Used for |
|---|---|---|
| hosted (default) | `LLM_API_BASE` in `.env` | Everything: evaluation, demos, production. |
| mock | `DEMO_MOCK_LLM=1` | UI walkthroughs with no model and no network. |

The model of record is **Llama 3.3 70B Instruct on Vertex AI Model-as-a-Service**
(`meta/llama-3.3-70b-instruct-maas`), hardcoded as `MODEL_NAME` in `generate_rfq.py`
and overridable with `LLM_MODEL`. Decoding is fixed in code rather than read from the
environment, so a run cannot silently differ from the paper: `temperature=0.2`,
`top_p=0.9`, and 800/400 token caps for the RFQ and recommendation respectively.

```bash
uvicorn app:app --port 8000                    # hosted generation
DEMO_MOCK_LLM=1 uvicorn app:app --port 8000    # canned responses, no network
```

`LLM_API_BASE` points at any OpenAI-compatible `/chat/completions` host. Leave
`LLM_API_KEY` unset on Vertex and auth falls back to Google ADC or the service-account
key; setting it switches to a static bearer token, which is what Together, Groq,
Fireworks and OpenRouter expect. `GET /health` reports the active target.

Two caveats worth knowing:

- **Vertex Llama Model-as-a-Service needs one-time enablement** in Model Garden, on
  the *API service* card rather than the self-deploy one. Until then every call
  returns 404 with *"not found or your project does not have access to it"*. Only
  `llama-3.3-70b-instruct-maas` and `llama-4-maverick-17b-128e-instruct-maas` are
  offered as managed endpoints, both `us-central1` only.
- **429s are routine, not failures.** MaaS runs on shared capacity. A two-call demo
  never sees one; a 180-call evaluation run sees several. `generate_output()` honours
  `Retry-After` and otherwise backs off exponentially with jitter, up to
  `LLM_MAX_RETRIES` (default 6).

## Approach and constraints

Per the thesis method, generation is **constrained prompt engineering only**: no
fine-tuning and no retrieval. Behaviour is changed by editing the system prompts in
`generate_rfq.py`. The hosted Llama 3.3 70B is the system under evaluation, and the
evaluation harness records the model in the generation-cache key and in
`scores.json` metadata so results always carry their provenance.

Because generation is hosted, property attributes and postcode leave the machine on
every call. Contact details do not: `_redact_contact_details()` strips
`contact_email` and `contact_phone` before either prompt sees the input.

## Known limitations

- **Grid headroom is unavailable.** The UKPN open-data headroom dataset the code
  queried has been retired, so `site_context.grid` is currently `null` for every
  postcode. Planning constraints are unaffected.
- **Listed-building detection is centroid-based** and effectively never resolves a
  listed building from a postcode centroid.
- **Sessions are in-memory** (`app.py`), so they do not survive a server restart.
