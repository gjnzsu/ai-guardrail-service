# AI Guardrail Service Design

**Date:** 2026-07-24  
**Updated:** 2026-07-25
**Status:** Approved

## Context

`ai-gateway-service` currently performs prompt-injection keyword checks and
sensitive-data regex matching inside its FastAPI request path. It supports
`log_only`, `audit`, `mask`, and `enforce` modes, but the detection logic cannot
reliably recognize contextual entities such as people, natural-language
addresses, customer identifiers, or internal project names.

The proof of concept will create a separate `ai-guardrail-service` repository.
It will compare deterministic regex detection, a fine-tuned encoder NER model,
and a small generative model. The service will first integrate with the gateway
in shadow mode so that detection failures cannot modify or block production
requests.

The initial assumptions are:

- prompt text is English;
- no production or real sensitive prompt data is available for training;
- training and evaluation data must therefore be synthetic;
- production inference must be CPU-first;
- no standing GPU infrastructure is available;
- Qwen3 0.6B is a comparison detector, not the authoritative masking detector;
- the gateway and guardrail service are separate repositories and deployable
  units.

## Goals

- Detect and locate sensitive entities in English prompt messages.
- Support these entity types:
  - `PERSON`
  - `ADDRESS`
  - `EMAIL`
  - `API_KEY`
  - `CUSTOMER_ID`
  - `INTERNAL_PROJECT`
- Produce validated character spans suitable for deterministic masking.
- Compare regex, a fine-tuned DistilBERT NER model, and Qwen3 0.6B zero-shot
  extraction on the same evaluation data.
- Run on CPU-only infrastructure.
- Keep detector implementations behind one versioned Guardrail API.
- Preserve raw prompts only in process memory and exclude them from logs and
  persistent artifacts.
- Integrate with `ai-gateway-service` in non-blocking shadow mode before any
  masking or enforcement rollout.
- Make partial detector failure observable without failing the entire analysis
  when another detector remains available.
- Prove the complete shadow path on the existing GKE cluster without creating
  public endpoints or new cluster capacity.

## Non-goals

- Prompt-injection, authorization, or data-exfiltration intent classification.
- Chinese or multilingual detection.
- Image, audio, attachment, or multimodal analysis.
- Streaming response guardrails.
- Production enforcement or fail-close policy.
- Storing or training on real prompts.
- Fine-tuning Qwen in the first proof of concept.
- GPU serving infrastructure.
- Committing model weights or generated datasets to Git.
- Creating, resizing, upgrading, or deleting a GKE cluster or node pool.
- Production-grade autoscaling, high availability, ingress, TLS, persistent
  model volumes, or infrastructure-as-code for the GKE proof of concept.

## Repository and Service Boundaries

Two repositories will have distinct responsibilities.

### `ai-gateway-service`

- Call the Guardrail HTTP API.
- Configure shadow-mode enablement and timeout behavior.
- Reuse the gateway request ID for correlation.
- Keep the provider request unchanged during shadow mode.
- Record only safe guardrail call metadata.
- Continue provider calls when the Guardrail Service is unavailable in shadow
  mode.

### `ai-guardrail-service`

- Validate analysis requests and input limits.
- Host deterministic regex detection.
- Load and run the fine-tuned NER model.
- Call the Qwen `llama.cpp` runtime.
- Validate, normalize, merge, and compare detector output.
- Perform deterministic masking when a future mode enables it.
- Generate synthetic data and train the NER model.
- Run offline benchmarks and produce evaluation reports.
- Expose health, readiness, and analysis endpoints.

### Model artifacts

Git stores training code, templates, manifests, checksums, small synthetic test
fixtures, and documentation. A separate artifact store or mounted volume stores:

- the fine-tuned NER model;
- tokenizer files;
- Qwen GGUF weights;
- generated training and evaluation data;
- full benchmark reports.

Local Docker Compose mounts model artifacts from a Git-ignored `models/`
directory. The GKE proof of concept stores model artifacts in a private Cloud
Storage bucket.

## Architecture

The selected architecture is a unified Guardrail Service with a separate
`llama.cpp` Qwen runtime.

```text
ai-gateway-service
        |
        | POST /v1/analyze
        v
ai-guardrail-service
  |-- RegexDetector
  |-- NerDetector
  |-- QwenDetectorClient --------> llama.cpp
  |-- DetectionValidator              |
  |-- DetectionMerger                 `-- Qwen3 0.6B Q4 GGUF
  |-- Masker
  `-- ShadowPolicy
```

The gateway depends only on the Guardrail API contract. It does not know which
model or runtime implements a detector.

## Detector Interface

Every detector implements one responsibility: convert a message into candidate
detections.

Conceptually:

```python
class Detector:
    async def detect(self, message: Message) -> DetectorResult:
        ...
```

`DetectorResult` contains:

- detector name and version;
- status and safe error code;
- elapsed time;
- candidate entity type;
- message index;
- character start and end;
- detector-native confidence when meaningful.

Detector implementations do not decide whether a request should be allowed,
masked, or blocked.

## Regex Detector

Regex remains authoritative for deterministic patterns:

- `EMAIL`
- `API_KEY`
- configured `CUSTOMER_ID` formats when the format is known

Regex configuration is compiled at startup. Invalid production patterns make
readiness fail instead of being silently ignored. Regex matches receive a
deterministic source marker rather than a fabricated probability.

## NER Model

### Base model

The trainable NER baseline uses:

```text
Base checkpoint: distilbert/distilbert-base-cased
Task: token classification
Language: English
Final artifact name: ai-guardrail-ner-en-v1
```

DistilBERT is a pretrained encoder, not an off-the-shelf model for the six
project-specific entities. A token-classification head is added and the model
is fine-tuned on the synthetic span-labelled dataset.

The cased checkpoint is selected because capitalization is a useful English NER
signal, especially for names and project codenames. Training data must still
include casing variations so capitalization does not become a shortcut.

### Labels

BIO labelling produces thirteen output labels:

```text
O
B-PERSON, I-PERSON
B-ADDRESS, I-ADDRESS
B-EMAIL, I-EMAIL
B-API_KEY, I-API_KEY
B-CUSTOMER_ID, I-CUSTOMER_ID
B-INTERNAL_PROJECT, I-INTERNAL_PROJECT
```

Fast-tokenizer offset mappings convert token predictions back to original
character spans. Regex remains authoritative for `EMAIL` and `API_KEY`, even
though these labels are included so the NER model can be compared consistently.

### Confidence

NER confidence is derived from token probabilities using one documented
aggregation rule. The first implementation will report the minimum token
probability within an entity span so that one weak token cannot be hidden by a
high average. Thresholds are selected on the validation set and stored by model
version.

### Future alternative

GLiNER zero-shot extraction is a future experiment, not a first-version runtime
dependency. It can later test whether runtime-provided entity descriptions are
valuable when adding new enterprise entity types.

## Qwen Detector

The generative comparison uses:

```text
Model: Qwen3 0.6B
Runtime: llama.cpp server
Format: GGUF Q4_K_M
Mode: zero-shot, non-thinking
Temperature: 0
Maximum input: 512 tokens
Maximum output: 96 tokens
```

The prompt instructs Qwen to return only allowed entity types and exact
substrings from the input. `llama.cpp` JSON Schema constraints ensure syntactic
output shape but do not establish semantic correctness.

The Qwen adapter must:

1. parse schema-constrained JSON;
2. reject unknown entity types;
3. verify that every proposed entity is an exact source substring;
4. handle repeated substrings without guessing an occurrence;
5. calculate and validate character spans in code;
6. discard invalid candidates and increment a safe hallucination metric;
7. avoid treating model-reported confidence as a calibrated probability.

Qwen is observational during the first release and does not contribute spans to
masking decisions.

## API Contract

### `POST /v1/analyze`

Example request:

```json
{
  "request_id": "req-123",
  "language": "en",
  "mode": "shadow",
  "messages": [
    {
      "index": 0,
      "role": "user",
      "content": "Send John Smith's CUST-93821 record to Project Falcon."
    }
  ],
  "detectors": ["regex", "ner", "qwen"]
}
```

Initial constraints:

- only `language=en`;
- only textual message content;
- unique integer message indexes;
- configured maximum messages, characters, and tokens;
- only known detector names;
- only `mode=shadow` in the first integration.

Example response:

```json
{
  "request_id": "req-123",
  "api_version": "v1",
  "offset_unit": "unicode_codepoint",
  "mode": "shadow",
  "decision": "observe",
  "result_status": "success",
  "entities": [
    {
      "message_index": 0,
      "type": "PERSON",
      "start": 5,
      "end": 15,
      "sources": ["ner", "qwen"],
      "confidence": 0.96
    }
  ],
  "detector_runs": [
    {
      "detector": "regex",
      "status": "success",
      "latency_ms": 2
    },
    {
      "detector": "ner",
      "status": "success",
      "model_version": "ai-guardrail-ner-en-v1",
      "latency_ms": 74
    },
    {
      "detector": "qwen",
      "status": "success",
      "model_version": "qwen3-0.6b-q4",
      "latency_ms": 612
    }
  ],
  "total_latency_ms": 615
}
```

The public response omits entity text. It returns only message identity, entity
type, validated offsets, sources, and meaningful confidence.

`start` is inclusive and `end` is exclusive. Both are Unicode code-point
offsets into the original message content. The explicit `offset_unit` prevents
clients that use UTF-16 indexing from interpreting the span incorrectly.

### HTTP behavior

- `200`: at least one requested detector succeeded; `result_status` may be
  `success` or `partial`.
- `413`: input exceeds a configured size limit.
- `422`: request structure, language, mode, or detector name is unsupported.
- `503`: the service has no usable detector or is not ready.

Errors do not include prompt content, entity content, model raw output, or the
model prompt.

### Health endpoints

`GET /health` indicates process liveness.

`GET /readiness` reports component state:

- regex configuration;
- NER model loading and version;
- Qwen runtime reachability and configured model.

Qwen unavailability produces a degraded readiness state when regex and NER
remain usable.

## Request Data Flow

Within the Guardrail Service:

```text
request validation
  -> bounded message normalization
  -> parallel detector execution
  -> per-detector validation
  -> normalization
  -> overlap and duplicate resolution
  -> policy result
  -> safe metadata response
```

Regex, NER, and Qwen execute concurrently. The overall synchronous analysis
time is therefore dominated by the slowest detector rather than the sum of all
three detector times.

During gateway shadow mode:

```text
gateway request
  |-> original provider call
  `-> guardrail shadow call
```

The two operations start concurrently. The Guardrail result never changes
provider messages or the provider decision. The gateway returns the provider
response without waiting for an unfinished Guardrail task. The task records its
result separately when it completes, using the same request ID for correlation.

When a future mask mode is approved, the ordering will change to:

```text
guardrail analysis
  -> validated authoritative spans
  -> deterministic mask
  -> provider call
```

That behavior requires a separate rollout decision and is outside the first
proof of concept.

## Validation and Merge Rules

All candidates must satisfy:

- the message index exists;
- `0 <= start < end <= len(content)`;
- offsets use Unicode code points and `end` is exclusive;
- the type is in the allowlist;
- the span is within configured length limits;
- Qwen substring and occurrence validation succeeded.

Exact duplicates with the same type are merged and their sources combined.

Overlap rules are deterministic:

1. Identical span and type: merge sources.
2. Identical span with conflicting types: select by detector and type priority,
   while recording a conflict metric.
3. Partial overlap with the same type: retain the longer validated span.
4. Partial overlap with different types: select by deterministic priority and
   record `overlap_conflict`.
5. Unresolvable candidates remain visible in offline comparison output but do
   not enter an authoritative masking set.

Entity risk priority is:

```text
API_KEY
> EMAIL
> CUSTOMER_ID
> INTERNAL_PROJECT
> ADDRESS
> PERSON
```

Regex has highest authority only for entity types assigned to regex. It cannot
override unrelated NER output merely because it is deterministic.

## Masking

The first gateway integration is shadow-only. The service nevertheless defines
a testable deterministic Masker for later use.

Masking:

- accepts only validated and resolved spans;
- creates a new message structure rather than mutating input;
- sorts spans by descending start offset;
- replaces from right to left;
- uses `[REDACTED:<TYPE>]`;
- never calls a model;
- never returns original entity text in metadata;
- never attempts free-form rewriting.

## Failure and Degradation

Per-detector failures produce partial results:

```text
Qwen timeout        -> Regex + NER
NER failure         -> Regex; Qwen remains observational
NER and Qwen fail   -> Regex
All detectors fail  -> HTTP 503
```

In gateway shadow mode:

```text
Guardrail unavailable -> provider call continues
                       -> guardrail_unavailable is recorded
```

Future mask or enforce modes must explicitly define fail-open or fail-close per
consumer. They cannot silently inherit shadow behavior.

## Privacy and Security

- Raw prompts exist only in request memory.
- Request bodies and model prompts are excluded from application and access
  logs.
- Entity text is excluded from public API responses and metrics.
- Qwen raw output is not logged.
- `llama.cpp` prompt logging is disabled.
- Stack traces exposed to clients contain no input.
- The GKE POC API is cluster-internal behind a `ClusterIP` Service. Network
  policy is future hardening rather than a POC dependency.
- Evaluation uses only synthetic values.
- Model and dataset artifacts have version manifests and checksums.
- Service metrics contain only types, counts, statuses, latencies, and versions.

## Synthetic Dataset

Dataset records use JSONL with exact character spans and provenance:

```json
{
  "id": "train-000001",
  "language": "en",
  "text": "Send Jane Cooper's record to Project Orion.",
  "entities": [
    {"type": "PERSON", "start": 5, "end": 16},
    {"type": "INTERNAL_PROJECT", "start": 29, "end": 42}
  ],
  "template_family": "record_transfer",
  "generator_version": "v1",
  "split": "train"
}
```

Generation requirements:

- deterministic random seed;
- fictitious values only;
- positive, negative, hard-negative, multi-entity, casing, punctuation, and
  obfuscation examples;
- no template family shared across training, validation, and challenge splits;
- manually reviewed challenge set;
- source and license information for every reusable word list;
- generator tests that recalculate and verify every span.

Target sizes:

```text
Training:    1,500-3,000
Validation:    300-500
Challenge:     150-300
```

## Training

The NER training pipeline:

```text
JSONL character spans
  -> fast-tokenizer offsets
  -> BIO labels
  -> DistilBERT token-classification fine-tuning
  -> validation checkpoint selection
  -> model and tokenizer export
  -> CPU inference benchmark
  -> version manifest
```

Training configuration records:

- base checkpoint revision;
- dataset and generator versions;
- label mapping;
- random seed;
- hyperparameters;
- library versions;
- selected thresholds;
- evaluation metrics;
- output checksums.

Training may use a temporary external GPU later, but the deployed artifact must
pass CPU inference requirements. Initial implementation does not provision GPU
infrastructure.

## Evaluation

The benchmark compares:

- regex only;
- NER only;
- Qwen only;
- regex plus NER;
- regex plus NER with Qwen observation.

Reported metrics:

- entity precision, recall, and F1;
- strict-span and partial-span F1;
- per-entity metrics;
- sensitive-character miss rate;
- non-sensitive-character mask rate;
- Qwen JSON parse rate;
- Qwen invalid or hallucinated entity rate;
- repeated-run consistency;
- P50, P95, and P99 latency;
- peak memory;
- timeout and error rate.

Initial proof-of-concept acceptance targets:

- Qwen JSON parse rate at least 99%;
- no unvalidated Qwen candidate reaches normalized output;
- `PERSON` and `ADDRESS` strict-span F1 at least 0.85;
- `CUSTOMER_ID` and `INTERNAL_PROJECT` strict-span F1 at least 0.75;
- NER local inference P95 at most 150 ms in a container limited to 2 vCPU and
  2 GiB memory;
- Qwen CPU inference P95 at most 2 seconds in a container limited to 4 vCPU and
  4 GiB memory;
- timeout and service failure degrade to available deterministic results;
- logs contain no prompt or entity text.

These are learning-oriented proof-of-concept targets, not production safety
guarantees.

## Test Strategy

### Unit tests

- regex boundary and invalid configuration behavior;
- token-to-character offset alignment;
- invalid type and out-of-range span rejection;
- duplicate, nested, and conflicting detections;
- repeated substring handling;
- right-to-left deterministic masking;
- Unicode and punctuation boundaries;
- Qwen invalid JSON, invalid entity, ambiguity, and timeout;
- NER load and inference failure degradation;
- captured logs do not contain fixture prompt or entity text.

### API tests

- health and readiness states;
- valid analysis and partial success;
- empty messages and invalid indexes;
- unsupported language, mode, and detector;
- input size limits;
- all-detector failure;
- response schema excludes entity text.

### Integration tests

- fake NER and fake `llama.cpp` in standard CI;
- optional local real-model smoke test;
- Docker Compose Guardrail and `llama.cpp` startup;
- gateway shadow client timeout and unavailable behavior;
- request-ID correlation without raw prompt logging.

Large model downloads and full model benchmarks are separate explicit jobs, not
standard unit-test dependencies.

## Deployment

### Runtime topology

The proof of concept uses separate processes or containers:

```text
guardrail-api
  - FastAPI
  - Regex and DistilBERT NER

qwen-runtime
  - llama.cpp server
  - Qwen3 0.6B Q4_K_M GGUF
```

Initial Qwen benchmark allocation:

```text
CPU request: 2
CPU limit:   4
Memory request: 2 GiB
Memory limit:   4 GiB
```

This is a starting point for measurement, not a guaranteed production sizing
result. Qwen concurrency begins at one and is increased only after load testing.
Every benchmark report records the host CPU model, operating system, container
limits, model checksum, context limit, and runtime version.

The existing gateway pod does not load either model.

### Local Docker Compose

Docker Desktop provides the cloud-independent vertical slice:

```text
docker compose up --build
  |-- ai-guardrail
  |     `-- bind-mounted DistilBERT artifact
  `-- qwen-runtime
        `-- bind-mounted Qwen GGUF
```

The local `models/` directory is ignored by Git. The Qwen container uses an
upstream `llama.cpp` server image pinned by digest. No custom Qwen image is
required.

### GKE proof-of-concept boundary

The cloud proof of concept:

- reuses the explicitly selected existing GKE cluster;
- reuses the existing `ai-gateway` namespace;
- deploys `ai-guardrail` and `qwen-runtime` as separate one-replica
  Deployments;
- exposes both workloads through internal `ClusterIP` Services;
- uses `Recreate` strategy so an update does not temporarily require two copies
  of a model;
- creates no Namespace, LoadBalancer, Ingress, DNS, TLS, PVC, database, cluster,
  or node pool;
- does not automatically resize or otherwise mutate the existing cluster.

Before deployment, the operator confirms the project, cluster, location, node
architecture, and available CPU and memory. Insufficient capacity leaves the
workload unscheduled and is an experiment result, not permission to resize the
cluster.

Guardrail readiness requires a usable API, regex configuration, and NER model.
Qwen unavailability is reported as degraded and does not make the authoritative
detectors unavailable. Gateway shadow calls remain fail-open.

### Model delivery on GKE

One private Cloud Storage bucket stores versioned NER and Qwen artifacts:

```text
gs://<project>-ai-guardrail-poc-models/
|-- ner/ai-guardrail-ner-en-v1/<sha256>/...
`-- qwen/qwen3-0.6b-q4-k-m/<sha256>/model.gguf
```

Each model Pod uses an init container to:

1. download the configured object to an `emptyDir`;
2. require the configured object generation;
3. verify the configured SHA-256;
4. expose the verified files read-only to the runtime container.

A dedicated Kubernetes ServiceAccount receives only
`roles/storage.objectViewer` on the model bucket through Workload Identity
Federation for GKE. No service-account key is created or mounted.

The POC intentionally uses `emptyDir` instead of a PVC. Model download time and
Pod-ready time are measured as learning results.

### Build and deploy

The repository contains:

```text
Dockerfile
compose.yaml
cloudbuild.yaml
deploy/gke-poc/
|-- kustomization.yaml
|-- service-account.yaml
|-- configmap.yaml
|-- guardrail.yaml
`-- qwen.yaml
```

There is no Terraform module and no setup, deploy, verify, or destroy script
suite. A short runbook contains the one-time `gcloud` commands to create the
bucket, upload models, grant bucket read access, and grant the Cloud Build
service account the minimum image-push and GKE-deploy permissions.

One manually invoked Cloud Build:

1. builds the Guardrail image;
2. tags it with the Git commit SHA and pushes it to the existing Artifact
   Registry repository;
3. renders the pinned Guardrail image and pinned upstream `llama.cpp` image;
4. applies the flat Kustomize directory to the selected cluster and namespace;
5. waits for both rollouts;
6. checks both Service endpoints and runs one synthetic smoke request from a
   short-lived in-cluster Pod, because `ClusterIP` is not reachable directly
   from the Cloud Build worker.

Cloud Build is manually invoked for the POC rather than triggered on every
commit. The deployment never uses `latest`.

### GKE POC acceptance and cleanup

The cloud proof of concept is accepted when:

- both Deployments are ready and both Services are `ClusterIP`;
- both model artifacts pass generation and SHA-256 checks;
- one synthetic request produces validated regex and NER spans;
- Qwen runs and remains explicitly observational;
- the Gateway shadow call does not mutate the provider prompt;
- Guardrail timeout or unavailability does not block the provider call;
- logs contain no raw prompt, entity text, or Qwen raw output;
- CPU, memory, detector latency, model download time, and Pod-ready time are
  recorded.

Latency targets remain learning criteria. Missing a target is recorded without
automatically expanding the cluster or POC scope.

Cleanup is one explicit command:

```powershell
kubectl delete -k deploy/gke-poc
```

The kustomization does not contain a Namespace resource, so this removes only
the declared Guardrail POC workloads and leaves the existing Gateway and
`ai-gateway` namespace intact. The model bucket remains until the operator
separately decides to delete it.

## Delivery Phases

### Phase 1: offline experiment

- repository, contracts, and synthetic data generator;
- regex baseline;
- DistilBERT fine-tuning and inference;
- Qwen `llama.cpp` adapter;
- comparable benchmark report.

### Phase 2: local service

- Guardrail API;
- validation, merge, and masking pipeline;
- health and readiness;
- Docker Compose with `llama.cpp`;
- failure and privacy tests.

### Phase 3: gateway shadow integration

- client in `ai-gateway-service`;
- concurrent best-effort shadow analysis;
- safe correlation and detector metadata;
- no prompt mutation or blocking.

### Phase 4: GKE proof of concept

- local Docker Compose vertical slice;
- private GCS model delivery through Workload Identity;
- one-command Cloud Build deployment to the existing cluster and namespace;
- internal Guardrail and Qwen Services;
- synthetic end-to-end shadow verification;
- resource, latency, startup, and privacy evidence.

### Future phases

- conditional Qwen gray-zone routing;
- GLiNER zero-shot comparison;
- model quantization and ONNX optimization;
- controlled mask-mode rollout;
- consumer-specific fail-open or fail-close policy;
- multilingual evaluation.

## Key Decisions

- Use a separate repository and deployable service.
- Use regex as the deterministic baseline.
- Fine-tune `distilbert-base-cased` as `ai-guardrail-ner-en-v1`.
- Use Qwen3 0.6B Q4_K_M through `llama.cpp` as a zero-shot comparison.
- Keep Qwen observational in the first release.
- Return validated spans without entity text.
- Run all detectors concurrently.
- Use synthetic data with split isolation and a hand-reviewed challenge set.
- Integrate with the gateway in shadow mode before masking or enforcement.
- Use Unicode code-point `[start, end)` offsets in the API contract.
- Reuse the existing GKE cluster and `ai-gateway` namespace for the POC.
- Use Docker Compose locally and one Cloud Build configuration for GKE.
- Keep model weights in private GCS and verify object generation and SHA-256.
- Use a pinned upstream `llama.cpp` server image instead of building a custom
  Qwen runtime image.
- Keep GKE manifests flat and cleanup limited to `kubectl delete -k`; do not add
  Terraform or deployment-script frameworks for this POC.

## Local tool-execution experiment (2026-10-05)

See the [execution diagram](../../diagrams/ai-guardrail-service-design-diagram-01.png)
and [editable Draw.io source](../../diagrams/ai-guardrail-service-design-diagram-01.drawio).

This standalone learning experiment in `ai_guardrail.tool_execution` is separate
from detectors, evaluation, and the Guardrail API. The detector non-goals and
observational/shadow semantics above remain unchanged. It introduces no agent
runtime, OPA, network service, banking/payment integration, or dependency.

A trusted local caller injects synchronous mock callables under fixed tool
identifiers and an explicit allowlist into `ToolExecutor`. Every step calls
`execute(tool_id, ...)`, including steps receiving a previous tool's output.
The entry point checks the local allowlist immediately before invoking each
registered callable; missing permission or registration raises a fixed
`PermissionError` without calling the tool. Registration alone grants no access.
An empty allowlist denies all tools. The allowlist is retained so local policy
changes are checked on the next execution; the registry is copied on creation.

Audit records remain in memory and contain only `tool_id`, `decision`, and a
fixed reason (`allowlisted`, `not_allowlisted`, or `not_registered`). Trusted
identifiers must not contain sensitive data. Arguments, outputs, and prompts
are excluded. An allow record describes authorization before the call, not its
successful completion; the executor does not catch or log tool exceptions.

The three synthetic tests demonstrate one allowed call, two individually checked
calls with output chaining, and a registered but unallowlisted callable blocked
before invocation. They also assert the small audit schema excludes synthetic
secrets. This is a single-process mock experiment, not a sandbox or production
security guarantee: callers retaining callable references can bypass the entry
point. Concurrent policy mutation, tool isolation, exception sanitization, and
real integrations are outside this approved local scope.
