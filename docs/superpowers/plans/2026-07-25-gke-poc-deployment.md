# GKE POC Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the implemented Guardrail API and observational Qwen runtime locally with Docker Desktop, then deploy the same two-service topology to the existing GKE cluster through one Cloud Build command.

**Architecture:** Docker Compose bind-mounts local model artifacts for the cloud-independent smoke path. GKE uses two one-replica Deployments in the existing `ai-gateway` namespace; init containers download generation-pinned model artifacts from private GCS and verify SHA-256 before the runtimes start. A manually invoked Cloud Build builds one Guardrail image, renders six flat Kustomize resources, deploys them, and runs an in-cluster synthetic smoke request.

**Tech Stack:** Docker Desktop, Docker Compose, Python 3.11, FastAPI, `llama.cpp`, Kubernetes, Kustomize, Cloud Build, Artifact Registry, Cloud Storage, and Workload Identity Federation for GKE.

## Global Constraints

- Execute this plan only after the local Guardrail API and Gateway shadow-integration plans pass.
- Reuse the explicitly selected existing GKE cluster and the existing `ai-gateway` namespace.
- Do not create, resize, upgrade, or delete a cluster, node pool, or Namespace.
- Do not add Terraform, Helm, deployment-script suites, PVCs, Ingress, LoadBalancer Services, DNS, TLS, databases, GPUs, HPA, or PodDisruptionBudget.
- Deploy `ai-guardrail` and `qwen-runtime` as separate one-replica Deployments with `Recreate` strategy.
- Use only `ClusterIP` Services.
- Use a dedicated Kubernetes ServiceAccount and Workload Identity Federation; never use a service-account key file.
- The model bucket is private. Model downloads must match both object generation and SHA-256.
- Qwen uses `ghcr.io/ggml-org/llama.cpp:server@sha256:4f02c560799a1569be08b0183d52b94b0d4a6e4b88f52f20562d2334c73837d4`.
- Application images use the full Git commit SHA and never use `latest`.
- Cloud Build may deploy workloads but must not alter cluster capacity.
- Raw prompts, entity text, Qwen raw output, model bytes, and generated reports must not enter logs or Git.
- Cleanup is exactly `kubectl delete -k deploy/gke-poc`; the kustomization must not contain a Namespace.
- The checkbox state in this file is the sole Phase 4 progress source.

---

## Dependencies and Consumed Interfaces

This plan consumes these completed interfaces:

- Guardrail container command starts `ai_guardrail.api.app:app` on port `8080`.
- Guardrail exposes `GET /health`, `GET /readiness`, and `POST /v1/analyze`.
- Guardrail reads `NER_MODEL_PATH` and `QWEN_BASE_URL`.
- `POST /v1/analyze` accepts the approved v1 shadow request and returns
  `offset_unit="unicode_codepoint"`.
- Gateway reads `GUARDRAIL_URL=http://ai-guardrail:8080` and keeps shadow calls
  fail-open.
- The NER artifact is a gzip tar archive whose root contains the exported model
  and tokenizer files.
- Qwen is one GGUF file.

Do not compensate for a missing dependency in Kubernetes YAML. Return to the
owning implementation plan and complete the missing interface first.

## File Map

```text
ai-guardrail-service/
|-- .dockerignore
|-- compose.yaml
|-- cloudbuild.yaml
|-- deploy/gke-poc/
|   |-- kustomization.yaml
|   |-- service-account.yaml
|   |-- configmap.yaml
|   |-- guardrail.yaml
|   `-- qwen.yaml
|-- docs/gke-poc.md
`-- tests/deployment/
    |-- test_compose_contract.py
    |-- test_gke_manifests.py
    `-- test_cloudbuild_contract.py
```

### Task 1: Docker Desktop Vertical Slice

**Files:**
- Modify: `.dockerignore`
- Create: `compose.yaml`
- Create: `tests/deployment/test_compose_contract.py`

**Interfaces:**
- Consumes: the completed Guardrail `Dockerfile`, API endpoints, local
  `models/ner/` directory, and `models/qwen/model.gguf`.
- Produces: `docker compose up --build --wait` local startup contract.

- [ ] **Step 1: Write the failing Compose contract test**

Create `tests/deployment/test_compose_contract.py`:

```python
import subprocess

import yaml


def test_compose_has_only_internal_guardrail_to_qwen_dependency() -> None:
    completed = subprocess.run(
        ["docker", "compose", "config"],
        check=True,
        capture_output=True,
        text=True,
    )
    config = yaml.safe_load(completed.stdout)
    services = config["services"]

    assert set(services) == {"ai-guardrail", "qwen-runtime"}
    assert services["ai-guardrail"]["environment"]["QWEN_BASE_URL"] == (
        "http://qwen-runtime:8080"
    )
    assert services["ai-guardrail"]["environment"]["NER_MODEL_PATH"] == (
        "/models/ner"
    )
    assert services["qwen-runtime"]["image"].startswith(
        "ghcr.io/ggml-org/llama.cpp:server@sha256:"
    )
    assert services["ai-guardrail"]["volumes"][0]["read_only"] is True
    assert services["qwen-runtime"]["volumes"][0]["read_only"] is True
```

- [ ] **Step 2: Run the Compose contract test and verify RED**

Run:

```powershell
python -m pytest tests/deployment/test_compose_contract.py -v
```

Expected: FAIL because `compose.yaml` does not exist.

- [ ] **Step 3: Add model exclusions to the Docker build context**

Append to `.dockerignore`:

```text
models/
artifacts/
datasets/generated/
evaluation/reports/
```

- [ ] **Step 4: Create the two-service Compose file**

Create `compose.yaml`:

```yaml
services:
  ai-guardrail:
    build:
      context: .
    ports:
      - "127.0.0.1:8080:8080"
    environment:
      NER_MODEL_PATH: /models/ner
      QWEN_BASE_URL: http://qwen-runtime:8080
      QWEN_TIMEOUT_SECONDS: "2"
      LOG_LEVEL: INFO
    volumes:
      - type: bind
        source: ./models/ner
        target: /models/ner
        read_only: true
    depends_on:
      - qwen-runtime
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - >-
          import urllib.request;
          urllib.request.urlopen('http://127.0.0.1:8080/readiness', timeout=2)
      interval: 5s
      timeout: 3s
      retries: 24
      start_period: 10s

  qwen-runtime:
    image: ghcr.io/ggml-org/llama.cpp:server@sha256:4f02c560799a1569be08b0183d52b94b0d4a6e4b88f52f20562d2334c73837d4
    ports:
      - "127.0.0.1:8081:8080"
    volumes:
      - type: bind
        source: ./models/qwen
        target: /models
        read_only: true
    command:
      - --model
      - /models/model.gguf
      - --ctx-size
      - "1024"
      - --host
      - 0.0.0.0
      - --port
      - "8080"
      - --parallel
      - "1"
      - --log-disable
```

- [ ] **Step 5: Verify the Compose contract**

Run:

```powershell
docker compose config --quiet
python -m pytest tests/deployment/test_compose_contract.py -v
```

Expected: Compose exits `0` and `1 passed`.

- [ ] **Step 6: Run the real local model smoke**

Run:

```powershell
docker compose up --build --detach --wait
$body = @{
  request_id = "local-smoke-001"
  language = "en"
  mode = "shadow"
  messages = @(
    @{
      index = 0
      role = "user"
      content = "Contact Jane Cooper at jane.cooper@example.test."
    }
  )
  detectors = @("regex", "ner", "qwen")
} | ConvertTo-Json -Depth 6
$response = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8080/v1/analyze `
  -ContentType application/json `
  -Body $body
$response.offset_unit
$response.detector_runs | Select-Object detector,status
docker compose down
```

Expected:

- `offset_unit` is `unicode_codepoint`;
- regex and NER report `success`;
- Qwen reports `success` or an explicitly observable `timeout`;
- `docker compose down` exits `0`.

- [ ] **Step 7: Commit the local deployment slice**

```powershell
git add .dockerignore compose.yaml tests/deployment/test_compose_contract.py
git commit -m "feat: add local guardrail container slice"
```

### Task 2: Flat Kustomize Workloads

**Files:**
- Create: `deploy/gke-poc/kustomization.yaml`
- Create: `deploy/gke-poc/service-account.yaml`
- Create: `deploy/gke-poc/configmap.yaml`
- Create: `deploy/gke-poc/guardrail.yaml`
- Create: `deploy/gke-poc/qwen.yaml`
- Create: `tests/deployment/test_gke_manifests.py`

**Interfaces:**
- Consumes: Guardrail image, private model bucket, exact object names,
  generations, and SHA-256 values.
- Produces: two internal Kubernetes workloads selected by
  `app.kubernetes.io/part-of=ai-guardrail-poc`.

- [ ] **Step 1: Write the failing manifest policy test**

Create `tests/deployment/test_gke_manifests.py`:

```python
import subprocess

import yaml


def _objects() -> list[dict]:
    rendered = subprocess.run(
        ["kubectl", "kustomize", "deploy/gke-poc"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [item for item in yaml.safe_load_all(rendered) if item]


def test_gke_poc_is_internal_and_does_not_own_shared_infrastructure() -> None:
    objects = _objects()
    kinds = [item["kind"] for item in objects]
    assert "Namespace" not in kinds
    assert "PersistentVolumeClaim" not in kinds
    assert "Ingress" not in kinds

    services = [item for item in objects if item["kind"] == "Service"]
    assert {item["metadata"]["name"] for item in services} == {
        "ai-guardrail",
        "qwen-runtime",
    }
    assert all(item["spec"]["type"] == "ClusterIP" for item in services)

    deployments = [item for item in objects if item["kind"] == "Deployment"]
    assert len(deployments) == 2
    for deployment in deployments:
        assert deployment["metadata"]["namespace"] == "ai-gateway"
        assert deployment["spec"]["replicas"] == 1
        assert deployment["spec"]["strategy"]["type"] == "Recreate"
        assert (
            deployment["metadata"]["labels"]["app.kubernetes.io/part-of"]
            == "ai-guardrail-poc"
        )
        assert deployment["spec"]["template"]["spec"]["serviceAccountName"] == (
            "ai-guardrail-model-reader"
        )
```

- [ ] **Step 2: Run the manifest test and verify RED**

Run:

```powershell
python -m pytest tests/deployment/test_gke_manifests.py -v
```

Expected: FAIL because `deploy/gke-poc` does not exist.

- [ ] **Step 3: Create the Kustomize ownership boundary**

Create `deploy/gke-poc/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: ai-gateway
resources:
  - service-account.yaml
  - configmap.yaml
  - guardrail.yaml
  - qwen.yaml
```

Create `deploy/gke-poc/service-account.yaml`:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ai-guardrail-model-reader
  namespace: ai-gateway
  labels:
    app.kubernetes.io/part-of: ai-guardrail-poc
```

Create `deploy/gke-poc/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-guardrail-poc-config
  namespace: ai-gateway
  labels:
    app.kubernetes.io/part-of: ai-guardrail-poc
data:
  MODEL_BUCKET: MODEL_BUCKET_VALUE
  NER_OBJECT: NER_OBJECT_VALUE
  NER_GENERATION: NER_GENERATION_VALUE
  NER_SHA256: NER_SHA256_VALUE
  QWEN_OBJECT: QWEN_OBJECT_VALUE
  QWEN_GENERATION: QWEN_GENERATION_VALUE
  QWEN_SHA256: QWEN_SHA256_VALUE
```

The `_VALUE` strings are deliberate fail-safe render tokens consumed by Cloud
Build. Applying the source manifests directly must not start a model.

- [ ] **Step 4: Create the Guardrail Deployment and Service**

Create `deploy/gke-poc/guardrail.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-guardrail
  namespace: ai-gateway
  labels:
    app.kubernetes.io/name: ai-guardrail
    app.kubernetes.io/part-of: ai-guardrail-poc
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: ai-guardrail
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ai-guardrail
        app.kubernetes.io/part-of: ai-guardrail-poc
    spec:
      serviceAccountName: ai-guardrail-model-reader
      volumes:
        - name: ner-model
          emptyDir:
            sizeLimit: 2Gi
      initContainers:
        - name: fetch-ner-model
          image: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
          command:
            - /bin/sh
            - -ceu
            - |
              archive=/models/ner.tar.gz
              gcloud storage cp \
                --if-generation-match="${NER_GENERATION}" \
                "gs://${MODEL_BUCKET}/${NER_OBJECT}" "${archive}"
              echo "${NER_SHA256}  ${archive}" | sha256sum -c -
              mkdir -p /models/export
              tar -xzf "${archive}" -C /models/export
          envFrom:
            - configMapRef:
                name: ai-guardrail-poc-config
          volumeMounts:
            - name: ner-model
              mountPath: /models
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: "1"
              memory: 512Mi
      containers:
        - name: ai-guardrail
          image: AI_GUARDRAIL_IMAGE
          ports:
            - name: http
              containerPort: 8080
          env:
            - name: NER_MODEL_PATH
              value: /models/export
            - name: QWEN_BASE_URL
              value: http://qwen-runtime:8080
            - name: QWEN_TIMEOUT_SECONDS
              value: "2"
          volumeMounts:
            - name: ner-model
              mountPath: /models
              readOnly: true
          resources:
            requests:
              cpu: "1"
              memory: 1Gi
            limits:
              cpu: "2"
              memory: 2Gi
          readinessProbe:
            httpGet:
              path: /readiness
              port: http
            periodSeconds: 5
            failureThreshold: 24
          livenessProbe:
            httpGet:
              path: /health
              port: http
            periodSeconds: 15
            failureThreshold: 4
---
apiVersion: v1
kind: Service
metadata:
  name: ai-guardrail
  namespace: ai-gateway
  labels:
    app.kubernetes.io/part-of: ai-guardrail-poc
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: ai-guardrail
  ports:
    - name: http
      port: 8080
      targetPort: http
```

- [ ] **Step 5: Create the Qwen Deployment and Service**

Create `deploy/gke-poc/qwen.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen-runtime
  namespace: ai-gateway
  labels:
    app.kubernetes.io/name: qwen-runtime
    app.kubernetes.io/part-of: ai-guardrail-poc
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: qwen-runtime
  template:
    metadata:
      labels:
        app.kubernetes.io/name: qwen-runtime
        app.kubernetes.io/part-of: ai-guardrail-poc
    spec:
      serviceAccountName: ai-guardrail-model-reader
      volumes:
        - name: qwen-model
          emptyDir:
            sizeLimit: 2Gi
      initContainers:
        - name: fetch-qwen-model
          image: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
          command:
            - /bin/sh
            - -ceu
            - |
              model=/models/model.gguf
              gcloud storage cp \
                --if-generation-match="${QWEN_GENERATION}" \
                "gs://${MODEL_BUCKET}/${QWEN_OBJECT}" "${model}"
              echo "${QWEN_SHA256}  ${model}" | sha256sum -c -
          envFrom:
            - configMapRef:
                name: ai-guardrail-poc-config
          volumeMounts:
            - name: qwen-model
              mountPath: /models
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: "1"
              memory: 512Mi
      containers:
        - name: qwen-runtime
          image: ghcr.io/ggml-org/llama.cpp:server@sha256:4f02c560799a1569be08b0183d52b94b0d4a6e4b88f52f20562d2334c73837d4
          args:
            - --model
            - /models/model.gguf
            - --ctx-size
            - "1024"
            - --host
            - 0.0.0.0
            - --port
            - "8080"
            - --parallel
            - "1"
            - --log-disable
          ports:
            - name: http
              containerPort: 8080
          volumeMounts:
            - name: qwen-model
              mountPath: /models
              readOnly: true
          resources:
            requests:
              cpu: "2"
              memory: 2Gi
            limits:
              cpu: "4"
              memory: 4Gi
          readinessProbe:
            httpGet:
              path: /health
              port: http
            periodSeconds: 5
            failureThreshold: 36
          livenessProbe:
            httpGet:
              path: /health
              port: http
            periodSeconds: 15
            failureThreshold: 4
---
apiVersion: v1
kind: Service
metadata:
  name: qwen-runtime
  namespace: ai-gateway
  labels:
    app.kubernetes.io/part-of: ai-guardrail-poc
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: qwen-runtime
  ports:
    - name: http
      port: 8080
      targetPort: http
```

- [ ] **Step 6: Verify the rendered ownership and resource contract**

Run:

```powershell
kubectl kustomize deploy/gke-poc | Out-Null
python -m pytest tests/deployment/test_gke_manifests.py -v
python -m ruff check tests/deployment/test_gke_manifests.py
```

Expected: rendering exits `0`, `1 passed`, and Ruff exits `0`.

- [ ] **Step 7: Commit the flat GKE resources**

```powershell
git add deploy/gke-poc tests/deployment/test_gke_manifests.py
git commit -m "feat: add internal GKE guardrail workloads"
```

### Task 3: One-Command Cloud Build Deployment

**Files:**
- Create: `cloudbuild.yaml`
- Create: `tests/deployment/test_cloudbuild_contract.py`

**Interfaces:**
- Consumes: required Cloud Build substitutions and the Task 2 render tokens.
- Produces: one image build/push, one Kustomize apply, rollout checks, endpoint
  checks, and one in-cluster synthetic Guardrail smoke.

- [ ] **Step 1: Write the failing Cloud Build contract test**

Create `tests/deployment/test_cloudbuild_contract.py`:

```python
from pathlib import Path

import yaml


def test_cloudbuild_builds_once_and_deploys_without_latest() -> None:
    text = Path("cloudbuild.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert ":latest" not in text
    assert "${_GIT_SHA}" in text
    assert "kubectl apply -k /workspace/rendered" in text
    assert "kubectl rollout status deployment/ai-guardrail" in text
    assert "kubectl rollout status deployment/qwen-runtime" in text
    assert len(config["steps"]) == 4
```

- [ ] **Step 2: Run the Cloud Build test and verify RED**

Run:

```powershell
python -m pytest tests/deployment/test_cloudbuild_contract.py -v
```

Expected: FAIL because `cloudbuild.yaml` does not exist.

- [ ] **Step 3: Create the Cloud Build pipeline**

Create `cloudbuild.yaml`:

```yaml
steps:
  - id: build
    name: gcr.io/cloud-builders/docker
    args:
      - build
      - --tag
      - ${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/ai-guardrail-service:${_GIT_SHA}
      - .

  - id: push
    name: gcr.io/cloud-builders/docker
    args:
      - push
      - ${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/ai-guardrail-service:${_GIT_SHA}

  - id: render
    name: ubuntu:24.04
    entrypoint: bash
    args:
      - -ceu
      - |
        cp -R deploy/gke-poc /workspace/rendered
        sed -i \
          -e "s|AI_GUARDRAIL_IMAGE|${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/ai-guardrail-service:${_GIT_SHA}|g" \
          -e "s|MODEL_BUCKET_VALUE|${_MODEL_BUCKET}|g" \
          -e "s|NER_OBJECT_VALUE|${_NER_OBJECT}|g" \
          -e "s|NER_GENERATION_VALUE|${_NER_GENERATION}|g" \
          -e "s|NER_SHA256_VALUE|${_NER_SHA256}|g" \
          -e "s|QWEN_OBJECT_VALUE|${_QWEN_OBJECT}|g" \
          -e "s|QWEN_GENERATION_VALUE|${_QWEN_GENERATION}|g" \
          -e "s|QWEN_SHA256_VALUE|${_QWEN_SHA256}|g" \
          /workspace/rendered/*.yaml
        if grep -R -E '(_VALUE|AI_GUARDRAIL_IMAGE)' /workspace/rendered; then
          echo "unrendered deployment token" >&2
          exit 1
        fi

  - id: deploy-and-smoke
    name: gcr.io/cloud-builders/kubectl
    entrypoint: sh
    env:
      - CLOUDSDK_COMPUTE_REGION=${_CLUSTER_LOCATION}
      - CLOUDSDK_CONTAINER_CLUSTER=${_CLUSTER}
    args:
      - -ceu
      - |
        kubectl apply -k /workspace/rendered
        kubectl -n ${_NAMESPACE} rollout status deployment/ai-guardrail --timeout=10m
        kubectl -n ${_NAMESPACE} rollout status deployment/qwen-runtime --timeout=10m
        kubectl -n ${_NAMESPACE} get endpoints ai-guardrail qwen-runtime
        kubectl -n ${_NAMESPACE} exec deployment/ai-guardrail -- python -c '
        import json, urllib.request
        body = json.dumps({
            "request_id": "gke-smoke-001",
            "language": "en",
            "mode": "shadow",
            "messages": [{
                "index": 0,
                "role": "user",
                "content": "Contact Jane Cooper at jane.cooper@example.test."
            }],
            "detectors": ["regex", "ner", "qwen"]
        }).encode()
        request = urllib.request.Request(
            "http://ai-guardrail:8080/v1/analyze",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        response = json.load(urllib.request.urlopen(request, timeout=15))
        assert response["offset_unit"] == "unicode_codepoint"
        assert response["mode"] == "shadow"
        print(json.dumps({
            "result_status": response["result_status"],
            "detectors": [
                {"detector": run["detector"], "status": run["status"]}
                for run in response["detector_runs"]
            ],
        }, sort_keys=True))
        '

images:
  - ${_REGION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/ai-guardrail-service:${_GIT_SHA}

options:
  logging: CLOUD_LOGGING_ONLY

timeout: 1800s
```

The smoke output contains only detector names and statuses. It does not print
the request or returned entity spans.

- [ ] **Step 4: Verify the Cloud Build contract**

Run:

```powershell
python -m pytest tests/deployment/test_cloudbuild_contract.py -v
python -m ruff check tests/deployment/test_cloudbuild_contract.py
```

Expected: `1 passed`; Ruff exits `0`.

- [ ] **Step 5: Commit Cloud Build deployment**

```powershell
git add cloudbuild.yaml tests/deployment/test_cloudbuild_contract.py
git commit -m "feat: deploy guardrail POC with Cloud Build"
```

### Task 4: Minimal GKE Runbook and Acceptance Evidence

**Files:**
- Create: `docs/gke-poc.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: exact GCP target, model files, Cloud Build substitutions, and
  completed Gateway shadow integration.
- Produces: one-time bootstrap, one deploy command, one acceptance checklist,
  and one cleanup command.

- [ ] **Step 1: Write the minimal operator runbook**

Create `docs/gke-poc.md` with:

````markdown
# GKE POC

This runbook reuses an existing GKE cluster and the existing `ai-gateway`
namespace. It does not create or alter cluster capacity.

## 1. Set the explicit target

```powershell
$ProjectId = 'gen-lang-client-0896070179'
$Region = 'us-central1'
$Cluster = 'helloworld-cluster'
$ClusterLocation = 'us-central1'
$Namespace = 'ai-gateway'
$Repository = 'ai-gateway-repo'
$Bucket = "$ProjectId-ai-guardrail-poc-models"

gcloud config set project $ProjectId
gcloud container clusters describe $Cluster `
  --location $ClusterLocation `
  --project $ProjectId
gcloud container clusters get-credentials $Cluster `
  --location $ClusterLocation `
  --project $ProjectId
kubectl get nodes
kubectl top nodes
```

Stop when the selected target differs from the intended existing cluster or
when current allocatable resources cannot satisfy the committed requests. Do
not resize the cluster from this workflow.

## 2. One-time model bucket and access

```powershell
gcloud storage buckets create "gs://$Bucket" `
  --project $ProjectId `
  --location $Region `
  --uniform-bucket-level-access `
  --public-access-prevention

tar -czf models/ner/ai-guardrail-ner-en-v1.tar.gz `
  -C artifacts/ai-guardrail-ner-en-v1 .
gcloud storage cp models/ner/ai-guardrail-ner-en-v1.tar.gz `
  "gs://$Bucket/ner/ai-guardrail-ner-en-v1/model.tar.gz"
gcloud storage cp models/qwen/model.gguf `
  "gs://$Bucket/qwen/qwen3-0.6b-q4-k-m/model.gguf"

$ProjectNumber = (
  gcloud projects describe $ProjectId --format='value(projectNumber)'
).Trim()
$Principal = "principal://iam.googleapis.com/projects/$ProjectNumber/locations/global/workloadIdentityPools/$ProjectId.svc.id.goog/subject/ns/$Namespace/sa/ai-guardrail-model-reader"
gcloud storage buckets add-iam-policy-binding "gs://$Bucket" `
  --role roles/storage.objectViewer `
  --member $Principal

$BuildServiceAccount = (
  gcloud builds get-default-service-account `
    --project $ProjectId `
    --format='value(serviceAccountEmail)'
).Trim()
gcloud artifacts repositories add-iam-policy-binding $Repository `
  --project $ProjectId `
  --location $Region `
  --member "serviceAccount:$BuildServiceAccount" `
  --role roles/artifactregistry.writer
gcloud projects add-iam-policy-binding $ProjectId `
  --member "serviceAccount:$BuildServiceAccount" `
  --role roles/container.developer
```

If the bucket already exists, inspect it and reuse it; do not recreate or
change ownership blindly.

## 3. Capture immutable model values

```powershell
$NerObject = 'ner/ai-guardrail-ner-en-v1/model.tar.gz'
$QwenObject = 'qwen/qwen3-0.6b-q4-k-m/model.gguf'
$NerGeneration = (
  gcloud storage objects describe "gs://$Bucket/$NerObject" `
    --format='value(generation)'
).Trim()
$QwenGeneration = (
  gcloud storage objects describe "gs://$Bucket/$QwenObject" `
    --format='value(generation)'
).Trim()
$NerSha = (
  Get-FileHash models/ner/ai-guardrail-ner-en-v1.tar.gz -Algorithm SHA256
).Hash.ToLowerInvariant()
$QwenSha = (
  Get-FileHash models/qwen/model.gguf -Algorithm SHA256
).Hash.ToLowerInvariant()
$GitSha = (git rev-parse HEAD).Trim()
```

## 4. Prove the local slice

```powershell
docker compose up --build --detach --wait
docker compose ps
docker compose down
```

## 5. Deploy with one Cloud Build command

```powershell
gcloud builds submit `
  --project $ProjectId `
  --region $Region `
  --config cloudbuild.yaml `
  --substitutions "_REGION=$Region,_REPOSITORY=$Repository,_GIT_SHA=$GitSha,_CLUSTER=$Cluster,_CLUSTER_LOCATION=$ClusterLocation,_NAMESPACE=$Namespace,_MODEL_BUCKET=$Bucket,_NER_OBJECT=$NerObject,_NER_GENERATION=$NerGeneration,_NER_SHA256=$NerSha,_QWEN_OBJECT=$QwenObject,_QWEN_GENERATION=$QwenGeneration,_QWEN_SHA256=$QwenSha"
```

## 6. Acceptance

- Both Deployments are ready.
- Both Services are `ClusterIP`.
- The Cloud Build synthetic Guardrail smoke passed.
- One synthetic request through the existing Gateway proves shadow analysis
  does not mutate the provider prompt.
- Temporarily scaling `ai-guardrail` to zero proves Gateway fail-open behavior;
  restore it immediately to one replica.
- Logs contain no raw prompt, entity text, or Qwen raw output.
- Record CPU, memory, detector latency, model download time, and Pod-ready time.

Missing a latency target is a recorded POC result, not permission to resize the
cluster or expand scope.

## 7. Remove only the POC workloads

```powershell
kubectl delete -k deploy/gke-poc
kubectl -n ai-gateway get deployment ai-gateway
```

The private model bucket remains until an operator separately decides to remove
it.
````

- [ ] **Step 2: Link the runbook from the README**

Add under the README documentation list:

```markdown
- [GKE POC runbook](docs/gke-poc.md)
```

- [ ] **Step 3: Run the complete static deployment gate**

Run:

```powershell
docker compose config --quiet
kubectl kustomize deploy/gke-poc | Out-Null
python -m pytest tests/deployment -v
python -m ruff check tests/deployment
git diff --check
```

Expected: all commands exit `0`.

- [ ] **Step 4: Commit the runbook**

```powershell
git add README.md docs/gke-poc.md
git commit -m "docs: add minimal GKE POC runbook"
```

- [ ] **Step 5: Execute and record the environment acceptance**

Follow `docs/gke-poc.md` once against the explicitly confirmed existing
cluster. Record only:

- project, cluster, location, namespace, and Git SHA;
- application and `llama.cpp` image digests;
- NER and Qwen object generations and SHA-256;
- rollout status;
- aggregate detector status;
- CPU, memory, latency, model-download, and Pod-ready measurements;
- Gateway shadow success and fail-open outcomes;
- confirmation that log inspection found no fixture text.

Do not record prompt text, entity values, Qwen raw output, access tokens, or
credentials.

## Phase 4 Completion Gate

Phase 4 is complete only when:

- the Docker Desktop vertical slice passes;
- Cloud Build builds exactly one project image and deploys both workloads;
- source manifests contain no Namespace, public Service, Ingress, PVC, or
  mutable image tag;
- model downloads require the configured generation and SHA-256;
- both GKE workloads are ready on existing capacity;
- the synthetic Guardrail and Gateway shadow journeys pass;
- Gateway fail-open behavior is demonstrated;
- privacy-safe resource and latency evidence is recorded;
- cleanup removes only the resources declared in `deploy/gke-poc`.

Stop after this gate. Production hardening, autoscaling, networking policy,
remote state, promotion environments, and automated infrastructure deletion
remain outside the POC.
