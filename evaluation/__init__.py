"""Evaluation harness for the constrained-LLM RFQ generation thesis.

Produces the quantitative results for Chapter IV:
  - faithfulness (RQ1): deterministic information preservation + Gemini-judged
    fabrication,
  - installer-facing RFQ quality (RQ2): Gemini-as-judge with an installer persona,
  - homeowner-facing recommendation quality (RQ3): Gemini-as-judge with a
    homeowner persona,
  - structural completeness as a supporting metric.

The generator under test is the local Llama model (default 3.2 3B). The judge is
Google Gemini. Both have offline mock modes so the full scoring loop can be
validated without the model weights or an API key.
"""
