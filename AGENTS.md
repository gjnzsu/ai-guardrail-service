# Repository Working Agreement

## Scope

These instructions apply to all work in this repository unless a more specific
nested `AGENTS.md` overrides them.

## Proof-of-concept boundary

- Treat the current work as a learning-oriented proof of concept, not a
  production safety guarantee.
- Keep the first release English-only and limited to `PERSON`, `ADDRESS`,
  `EMAIL`, `API_KEY`, `CUSTOMER_ID`, and `INTERNAL_PROJECT`.
- Use synthetic data only. Do not persist or train on real prompt data.
- Keep inference CPU-first. Do not add GPU infrastructure without explicit user
  approval.
- Keep Qwen observational. Only validated authoritative regex and NER spans may
  enter a future masking set.
- Stop when the approved phase completion gate passes. Put additional
  hardening, optimization, and production readiness work in a backlog unless
  the user explicitly expands the scope.

## Source of truth

- Use the active phase implementation plan as the authoritative progress
  checklist.
- Keep the root README concise and link to the design, active plan, and
  operator workflow instead of duplicating them.
- Record consequential architecture decisions in the design or an ADR when the
  decision is made.

## Delivery sequence

- Establish a runnable vertical slice early: synthetic fixture, shared span
  contract, regex detector, evaluator, and privacy-safe aggregate report.
- Add NER and Qwen through the same detector and evaluation interfaces only
  after that slice runs.
- Prove the service locally with Docker Compose before deploying it to GKE.
- Integrate with the Gateway in non-blocking shadow mode before considering
  masking or enforcement.

## Privacy and artifacts

- Never log raw prompts, entity text, Qwen raw output, generated datasets, or
  model contents.
- Never commit generated datasets, benchmark reports, model weights, or local
  Terraform/state artifacts.
- Public API responses and persisted reports contain validated offsets and
  aggregate metadata, not entity text.

## GKE POC guardrails

- Reuse the explicitly selected existing cluster and the `ai-gateway`
  namespace.
- Do not create, resize, upgrade, or delete a cluster or node pool.
- Do not create a public Service, Ingress, DNS record, TLS endpoint, PVC, or
  database for the POC.
- Deploy Guardrail and Qwen as separate one-replica workloads and expose them
  only through `ClusterIP` Services.
- Use Workload Identity Federation for read-only model access; never create or
  mount service-account key files.
- Pin application images and the upstream `llama.cpp` image to immutable
  versions. Pin model objects by generation and SHA-256.
- The Kustomize resources must not include the existing Namespace object.
  Cleanup may delete only resources declared by the Guardrail POC
  kustomization.
- Cloud deployment must fail rather than mutate cluster capacity when the
  existing nodes cannot satisfy resource requests.

## Verification

- Map each approved acceptance scenario to an automated test or an explicit
  environment check.
- Standard tests must not download model weights or make network calls.
- Before each commit, inspect Git status, staged changes, whitespace errors,
  unexpected binaries, generated directories, and unusually large files.
- Before declaring a phase complete, run its unit, integration, lint, local
  container, privacy, and applicable GKE smoke checks and record exact results.

## Scope checkpoints and retrospective

- Pause for a scope checkpoint when an estimate grows by more than 50%, a new
  infrastructure dependency is proposed, three consecutive work cycles contain
  only fixes or hardening, or the approved POC journey already works.
- Treat privacy leaks, invalid span handling, evaluation contamination, and
  irreproducible results as P0. Treat a broken primary POC journey as P1.
  Defer maintainability automation and production readiness as P2/P3 unless
  explicitly approved.
- At the end of each material POC phase, record what worked, friction and root
  causes, decisions, and action items. Each action item needs a trigger and an
  observable completion condition.
