# Constrained LLM-Based RFQ Generation for Residential Decarbonisation

> **MSc Dissertation Repository**  
> *Design and Evaluation of Constrained LLM-Based RFQ Generation in Residential Decarbonisation*  

This repository contains the full production-shape codebase and evaluation harness for the dissertation. The pipeline transforms a UK homeowner's postcode and technology choice into a structured, installer-ready Request for Quote (RFQ), leveraging real UK open data (EPC & planning constraints) and a constrained Llama 3.3 70B model.

Four decarbonisation technologies are supported: **heat pump**, **solar PV**, **battery storage**, and **solar thermal**.

---
> ### Note Regarding API Keys & Mock vs. Paper Results
> Live execution connects to external APIs (**UK EPC Open Data API** and **Google Cloud Vertex AI** for Llama 3.3 70B and Gemini 3.5 Flash-Lite). To prevent unauthorized usage and comply with security practices, API keys and credentials are **not published in the GitHub repository**.
>
> **You can fully run and evaluate the codebase without API keys** using the built-in **Mock Modes**:
> - **Mock Generation (`DEMO_MOCK_LLM=1` / `--mock-gen`)**: Uses deterministic mock generation to exercise the entire pipeline offline without API calls.
> - **Mock Evaluation (`--mock-judge`)**: Exercises the full evaluation scoring loop and generates metric reports without requiring GCP Vertex AI credentials.
>
> ⚠️ **Note on Results**: Evaluation scores produced in Mock Mode verify code execution, but their numerical values will differ from the empirical results reported in the dissertation paper (which require the live Llama 3.3 70B model and Gemini judge). The exact live evaluation data reported in the paper is pre-computed and preserved in **`eval_outputs/`**.

---

## 🚀 How to Run the Code

You can run the demonstration pipeline either via an interactive **Jupyter Notebook** (1-click execution) or through the **Terminal**.

### Option A: Run via Jupyter Notebook
1. Open [`renbee_demo_journey.ipynb`](renbee_demo_journey.ipynb).
2. Run all cells. This executes the entire pipeline: postcode lookup $\rightarrow$ EPC fetch $\rightarrow$ planning constraints check $\rightarrow$ RFQ input assembly $\rightarrow$ output generation.

### Option B: Run via Interactive Web Demo (Terminal)
To run the interactive web application demo locally:

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Launch local web server in Mock Mode (no network/credentials required)
DEMO_MOCK_LLM=1 uvicorn app:app --port 8000
```
Open **`http://localhost:8000/demo`** in your browser to walk through the interactive Webflow homeowner journey.

### Option C: Run Pipeline via Command Line (CLI)
```bash
# Process a single case via CLI
python3 pipeline.py --postcode "E1 6AN" --tech heat_pump --mock
```

---

## 📊 How to Run the Evaluation Suite

The evaluation harness tests 30 real UK postcode cases across structural completeness, LLM faithfulness (preservation & fabrication rates), and qualitative rubric criteria (clarity, usability, helpfulness).

### Option A: Run Evaluation via Jupyter Notebook
Open and run **[`evaluation_runner.ipynb`](evaluation_runner.ipynb)**. It executes the evaluation suite, loads the results table, and plots metric visualisations.

### Option B: Run Evaluation via Terminal

```bash
# 1. Run full evaluation offline (no API keys required)
python3 evaluation/run_eval.py --mock-gen --mock-judge

# 2. Run live evaluation on Vertex AI (Requires GCP credentials in .env)
python3 evaluation/run_eval.py --repeats 3 --regenerate
```

---

## 📁 Project Structure & Key Components

The codebase is organized into four simple functional modules:

### 1. ⚙️ Core Generation Pipeline
- **`epc_to_rfq.py`**: Assembles property & form data into the RFQ schema and handles missing field detection.
- **`generate_rfq.py`**: Constrained Llama 3.3 70B generation engine using dedicated system prompts.
- **`pipeline.py`**: Python CLI orchestrator connecting data retrieval, assembly, and generation.

### 2. 🌐 Open Data Integration
- **`epc_fetch.py`**: Connects to the UK EPC Open Data API (with proxy fallback logic for missing certificates).
- **`external_data.py`**: Queries UK planning constraints.

### 3. 💻 Web App & Interactive Notebooks
- **`app.py`**: FastAPI web server hosting API endpoints (`/api/initiate`, `/api/generate`) and the local demo UI.
- **`renbee_demo_journey.ipynb`**: 1-click interactive notebook demonstrating the complete customer journey.

### 4. 📊 Evaluation Harness (`evaluation/`)
- **`evaluation_runner.ipynb`**: Notebook runner to execute evaluation runs and render visual plots.
- **`evaluation/run_eval.py`**: Main orchestrator running evaluations across 30 real postcode test cases.
- **`evaluation/faithfulness.py`**: Calculates information preservation, site-context coverage, and fabrication rates.
- **`evaluation/rubric.py`**: LLM judge scoring for qualitative criteria (Clarity, Usability, Helpfulness).
- **`evaluation/format_table.py`**: Formats evaluation outputs into the thesis Markdown summary table (RQ1-RQ3).
- **`rfq_cases_real_v1.json`**: Benchmark dataset containing 30 real UK postcode evaluation test cases.
- **`eval_outputs/`**: Contains pre-computed evaluation output files (`summary.csv`, `scores.json`, `generated.json`).

---

## 🛠 System Architecture & Approach

```text
Postcode + Technology Choice
      │
      ▼
EPC Open Data API  +  planning.data.gov.uk (Real Property & Site Constraints)
      │
      ▼
RFQ Input Assembly (epc_to_rfq.py) ──► missing_fields Form (Webflow Step 2)
      │
      ▼
Two Constrained LLM Prompt Moments (Llama 3.3 70B):
  1. Homeowner-facing EPC recommendation summary (Engagement)
  2. Installer-facing RFQ summary (HITL review before installer dispatch)
```

- **Constrained Prompting**: Per the thesis methodology, generation relies on constrained system prompts without fine-tuning or RAG, enforcing zero hallucination of factual property inputs.
- **Privacy & Safety**: Contact details (`email`, `phone`) are automatically redacted via `_redact_contact_details()` before any prompt context is dispatched to the LLM.

---

## 📝 Contact

- **Author**: Vanessa Rebecca Wiyono
- **Student id**: u5729891
- **Email**: Vanessa-Rebecca.Wiyono@warwick.ac.uk

