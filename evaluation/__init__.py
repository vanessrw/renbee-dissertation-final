"""Evaluation harness for the constrained-LLM RFQ generation thesis.

Produces the quantitative results for Chapter IV:
  - faithfulness (RQ1): deterministic information preservation + Gemini-judged
    fabrication,
  - installer-facing RFQ quality (RQ2): Gemini-as-judge with an installer persona,
  - homeowner-facing recommendation quality (RQ3): Gemini-as-judge with a
    homeowner persona,
  - structural completeness as a supporting metric.

The generator under test is hosted Llama 3.3 70B Instruct. The judge is Google
Gemini. Both have offline mock modes so the full scoring loop can be validated
with no network access and no credentials.
"""
