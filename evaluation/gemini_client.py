"""Thin wrapper around the Google Gemini API used as the LLM judge.

One entry point, `judge_json()`, sends a system + user prompt and returns a
parsed JSON object. Design goals:

  - Offline mock mode (GEMINI_MOCK=1 or mock=True) so the whole scoring loop can
    be exercised with no API key and no cost. Callers pass a `mock_response`
    describing the shape they expect back.
  - Lazy import of the SDK so mock mode works without the package installed.
  - Defensive JSON parsing (the model is asked for JSON, but we still tolerate
    code fences or surrounding prose).

The model is configurable via GEMINI_MODEL (default gemini-3.5-flash-lite). The
API key is read from GEMINI_API_KEY in the environment / .env.

Note on region: gemini-3.5-flash-lite is served only from the Vertex `global`
endpoint, not from regional ones such as us-central1 (a regional call 404s with
"Publisher model ... was not found"). GOOGLE_CLOUD_LOCATION therefore defaults
to "global" here and in .env.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def _mock_enabled(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    return os.getenv("GEMINI_MOCK") in {"1", "true", "yes"}


def _vertex_enabled() -> bool:
    """Vertex AI mode: explicit flag, or inferred when a Cloud project is set
    and no AI Studio API key is present."""
    if os.getenv("GEMINI_USE_VERTEX") in {"1", "true", "yes"}:
        return True
    return bool(os.getenv("GOOGLE_CLOUD_PROJECT")) and not os.getenv("GEMINI_API_KEY")


def _extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of raw model text.

    Tolerates markdown code fences and leading/trailing prose, mirroring the
    brace-depth approach used in generate_rfq.parse_model_output.
    """
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    # Fast path: the whole thing is JSON.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Brace-depth scan for the first balanced object.
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(cleaned[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    start = -1
                    continue
    return {}


class GeminiClient:
    """Minimal Gemini judge client. Instantiate once and reuse."""

    def __init__(self, model: Optional[str] = None, mock: Optional[bool] = None,
                 temperature: float = 0.0, max_retries: int = 3):
        self.model = model or DEFAULT_MODEL
        self.mock = _mock_enabled(mock)
        self.use_vertex = (not self.mock) and _vertex_enabled()
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None  # lazy

    # -- public ----------------------------------------------------------------

    @property
    def backend(self) -> str:
        if self.mock:
            return "mock"
        return f"vertex:{self.model}" if self.use_vertex else self.model

    def judge_json(
        self,
        system_prompt: str,
        user_prompt: str,
        mock_response: Optional[Callable[[], dict] | dict] = None,
    ) -> dict:
        """Return a parsed JSON object from the judge.

        In mock mode, returns `mock_response` (called if it is a callable). In
        live mode, calls Gemini with the system + user prompt, requests a JSON
        response, and parses it.
        """
        if self.mock:
            if callable(mock_response):
                return dict(mock_response())
            return dict(mock_response or {})

        # Fail fast on missing config (a config error, not a transient one).
        if self.use_vertex:
            if not os.getenv("GOOGLE_CLOUD_PROJECT"):
                raise RuntimeError(
                    "Vertex mode is on but GOOGLE_CLOUD_PROJECT is not set. Add it "
                    "to .env (and run `gcloud auth application-default login`)."
                )
        elif not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env, set GEMINI_USE_VERTEX=1 "
                "for Vertex AI, or run in mock mode (GEMINI_MOCK=1 or --mock-judge)."
            )

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                raw = self._call_gemini(system_prompt, user_prompt)
                parsed = _extract_json(raw)
                if parsed:
                    return parsed
                last_err = ValueError("empty or unparseable JSON from Gemini")
            except Exception as e:  # noqa: BLE001 - retry on any transient error
                last_err = e
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Gemini judge failed after {self.max_retries} attempts: {last_err}")

    # -- internal --------------------------------------------------------------

    def _ensure_client(self):
        if self._client is not None:
            return
        # New Google GenAI SDK: `pip install google-genai`.
        from google import genai  # lazy import
        if self.use_vertex:
            # Vertex AI: auth via Application Default Credentials (gcloud auth
            # application-default login). Bills the project's billing account,
            # so it works with the education credit and needs no API key.
            self._client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            )
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Add it to .env, set "
                    "GEMINI_USE_VERTEX=1 for Vertex AI, or run in mock mode."
                )
            self._client = genai.Client(api_key=api_key)

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_client()
        from google.genai import types  # lazy import
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return getattr(resp, "text", "") or ""
