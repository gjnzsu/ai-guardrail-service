# AI Guardrail Service

CPU-first sensitive-data detection service for AI prompts.

The project is currently in the approved design phase. Implementation will begin
after review of the design specification:

- [AI Guardrail Service design](docs/superpowers/specs/2026-07-24-ai-guardrail-service-design.md)

The first proof of concept will compare:

- deterministic regex detection;
- a fine-tuned English DistilBERT NER model; and
- Qwen3 0.6B zero-shot JSON extraction through `llama.cpp`.

The service will initially integrate with `ai-gateway-service` in shadow mode.
It will not modify or block prompts in that phase.
