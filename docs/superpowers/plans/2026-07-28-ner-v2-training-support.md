# NER v2 Training Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trusted v2 NER release profile, train `ai-guardrail-ner-en-v2` with the generated v2 dataset, select its validation threshold, and compare it with v1 on the frozen challenge seed.

**Architecture:** Introduce a small allowlisted release-profile abstraction in the artifact manifest module and reuse it from training, verification, detector loading, and threshold artifacts. Preserve the existing v1 constants and default CLI behavior. Run a controlled A/B experiment with the same base checkpoint, hyperparameters, and training seed.

**Tech Stack:** Python 3.12, Pydantic, PyTorch, Hugging Face Transformers/Trainer, pytest, Ruff, JSONL synthetic datasets.

## Global Constraints

- Keep `distilbert/distilbert-base-cased`, three epochs, batch size 16, learning rate `2e-5`, weight decay `0.01`, and `eval_loss` checkpoint selection unchanged.
- Use training seed `20260725` for both v1 and v2.
- Keep v2 dataset generation seed `20260727`.
- Preserve the existing v1 artifact without migration or overwrite.
- Allow only the complete trusted v1 and v2 release profiles.
- Never use challenge data for training, checkpoint selection, or threshold selection.
- Never log or persist prompt text or entity text in reports.
- Keep model, threshold, and evaluation artifacts ignored by Git.
- Do not use subagents.

---

### Task 1: Trusted NER Release Profiles

**Files:**
- Modify: `src/ai_guardrail/ner/manifest.py`
- Modify: `tests/ner/test_manifest.py`

**Interfaces:**
- Produces: `NerReleaseProfile(version: str, artifact_name: str, generator_version: str, dataset_version: str)`
- Produces: `get_release_profile(version: str) -> NerReleaseProfile`
- Produces: `get_release_profile_for_artifact(artifact_name: str) -> NerReleaseProfile`
- Preserves: `ARTIFACT_NAME`, `GENERATOR_VERSION`, and `BASE_CHECKPOINT` as v1-compatible constants.

- [ ] **Step 1: Write failing profile and v2 artifact tests**

Add tests that assert:

```python
profile = get_release_profile("v2")
assert profile.artifact_name == "ai-guardrail-ner-en-v2"
assert profile.generator_version == "v2"
assert profile.dataset_version == "v2"
```

Create a valid synthetic v2 artifact by setting its directory, manifest
`artifact_name`, `generator_version`, and `dataset_version` to the complete v2
profile. Assert `verify_model_artifact` accepts it. Parameterize mixed-profile
cases such as v2 directory plus `generator_version="v1"` and assert they fail
with `invalid NER model artifact`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ner/test_manifest.py -q
```

Expected: FAIL because the release-profile API and v2 verification do not
exist.

- [ ] **Step 3: Implement the allowlisted profile abstraction**

Add:

```python
@dataclass(frozen=True)
class NerReleaseProfile:
    version: str
    artifact_name: str
    generator_version: str
    dataset_version: str


_RELEASE_PROFILES = {
    "v1": NerReleaseProfile("v1", "ai-guardrail-ner-en-v1", "v1", "v1"),
    "v2": NerReleaseProfile("v2", "ai-guardrail-ner-en-v2", "v2", "v2"),
}
```

Make both lookup functions reject unknown values without echoing the supplied
value. Update `verify_model_artifact` to select the profile from the allowlisted
directory name and require all manifest identity fields to match it. Update
`verified_model_snapshot` to copy into a directory named with the verified
profile artifact name.

Extend `build_manifest` with `release_version: str = "v1"` and require
`dataset_version` plus `generator_version` to match the selected profile.

- [ ] **Step 4: Verify GREEN and v1 compatibility**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ner/test_manifest.py -q
```

Expected: all manifest tests PASS, including existing v1 tamper and checksum
tests.

- [ ] **Step 5: Commit**

```powershell
git add src/ai_guardrail/ner/manifest.py tests/ner/test_manifest.py
git commit -m "feat: add trusted NER release profiles"
```

### Task 2: Version-Aware Training

**Files:**
- Modify: `src/ai_guardrail/ner/train.py`
- Modify: `tests/ner/test_train.py`

**Interfaces:**
- Consumes: `get_release_profile(version: str) -> NerReleaseProfile`
- Changes: `validate_training_datasets(train_path, validation_path, expected_generator_version)`
- Adds CLI: `--release-version {v1,v2}`, default `v1`.

- [ ] **Step 1: Write failing v2 training tests**

Add a parser/default test and a mocked training test that supplies:

```python
Namespace(
    train=train_path,
    validation=validation_path,
    output=tmp_path / "ai-guardrail-ner-en-v2",
    seed=20260725,
    base_checkpoint=BASE_CHECKPOINT,
    release_version="v2",
)
```

Use v2 `LabeledExample` records. Assert the generated manifest contains:

```python
assert manifest["artifact_name"] == "ai-guardrail-ner-en-v2"
assert manifest["generator_version"] == "v2"
assert manifest["dataset_version"] == "v2"
assert manifest["seed"] == 20260725
```

Add tests rejecting a v2 profile with v1 records and a v1 profile with the v2
output name before ML dependency loading.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ner/test_train.py -q
```

Expected: FAIL because `--release-version` and version-aware dataset validation
are absent.

- [ ] **Step 3: Implement version-aware training**

Add the CLI argument:

```python
parser.add_argument(
    "--release-version",
    choices=("v1", "v2"),
    default="v1",
)
```

Resolve the trusted profile before ML dependencies load. Validate the output
directory name and both dataset generator versions against the profile. Pass
`release_version`, profile dataset version, and profile generator version into
manifest creation. Keep all hyperparameters unchanged.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ner/test_train.py -q
```

Expected: all training tests PASS, including existing v1 default behavior.

- [ ] **Step 5: Commit**

```powershell
git add src/ai_guardrail/ner/train.py tests/ner/test_train.py
git commit -m "feat: support v2 NER training"
```

### Task 3: Version-Aware Loading and Threshold Artifacts

**Files:**
- Modify: `src/ai_guardrail/detectors/ner.py`
- Modify: `src/ai_guardrail/evaluation/threshold_artifact.py`
- Modify: `tests/detectors/test_ner.py`
- Modify: `tests/evaluation/test_threshold_cli.py`
- Modify: `tests/evaluation/test_cli.py`

**Interfaces:**
- Consumes verified manifest field `artifact_name`.
- Produces threshold payload `model_version` matching the verified artifact.
- Preserves the v1 threshold artifact format and validation checks.

- [ ] **Step 1: Write failing v2 loading and threshold tests**

Create a valid v2 test artifact and assert:

```python
detector = NerDetector.load(v2_model_path, threshold=0.5)
assert detector.model_version == "ai-guardrail-ner-en-v2"
```

Build a v2 threshold artifact and assert:

```python
assert payload["model_version"] == "ai-guardrail-ner-en-v2"
```

Assert loading rejects a threshold whose `model_version` does not match the
verified model profile.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/detectors/test_ner.py tests/evaluation/test_threshold_cli.py tests/evaluation/test_cli.py -q
```

Expected: FAIL because detector and threshold identity are hardcoded to v1.

- [ ] **Step 3: Implement dynamic verified identity**

In `NerDetector.load`, set:

```python
model_version=snapshot.verified.manifest["artifact_name"]
```

In threshold artifact creation and loading, derive the expected model version
from `verify_model_artifact(model_path).manifest["artifact_name"]` instead of
`ARTIFACT_NAME`. Keep checksum, validation provenance, and candidate-threshold
checks unchanged.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/detectors/test_ner.py tests/evaluation/test_threshold_cli.py tests/evaluation/test_cli.py -q
```

Expected: all focused tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/ai_guardrail/detectors/ner.py src/ai_guardrail/evaluation/threshold_artifact.py tests/detectors/test_ner.py tests/evaluation/test_threshold_cli.py tests/evaluation/test_cli.py
git commit -m "feat: load and calibrate versioned NER artifacts"
```

### Task 4: Documentation and Static Verification

**Files:**
- Modify: `docs/offline-evaluation.md`
- Modify: `datasets/README.md`

**Interfaces:**
- Documents the exact v2 training and threshold commands.
- Documents that v1 and v2 use training seed `20260725`.

- [ ] **Step 1: Update operator commands**

Add the v2 commands from the approved design and state that
`datasets/challenge/en-v1.seed.jsonl` is the frozen A/B challenge. Do not
describe `challenge.candidates.jsonl` as reviewed.

- [ ] **Step 2: Run full verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
uv tool run ruff check .
git diff --check
git status --short
```

Expected: 0 test failures, no Ruff findings, no whitespace errors, and only the
intended documentation changes remain.

- [ ] **Step 3: Commit**

```powershell
git add docs/offline-evaluation.md datasets/README.md
git commit -m "docs: add NER v2 training workflow"
```

### Task 5: Train and Calibrate v2

**Files:**
- Generate, ignored: `artifacts/ai-guardrail-ner-en-v2/`
- Generate, ignored: `artifacts/thresholds/ai-guardrail-ner-en-v2.selected-threshold.json`

**Interfaces:**
- Consumes: `datasets/generated/v2/train.jsonl`
- Consumes: `datasets/generated/v2/validation.jsonl`
- Produces: verified v2 model and threshold artifacts.

- [ ] **Step 1: Record unchanged v1 identity**

Run a privacy-safe command that records the v1 artifact and manifest SHA-256
values from `verify_model_artifact`. Do not print manifest content.

- [ ] **Step 2: Train v2**

Run:

```powershell
.\.venv\Scripts\python.exe -m ai_guardrail.ner.train `
  --release-version v2 `
  --train datasets/generated/v2/train.jsonl `
  --validation datasets/generated/v2/validation.jsonl `
  --output artifacts/ai-guardrail-ner-en-v2 `
  --seed 20260725
```

Expected: three epochs complete and a verified
`artifacts/ai-guardrail-ner-en-v2/training-manifest.json` is written.

- [ ] **Step 3: Verify v2 artifact**

Run `verify_model_artifact` and print only artifact name, artifact SHA-256,
manifest SHA-256, seed, dataset version, and aggregate evaluation metrics.

- [ ] **Step 4: Select v2 threshold**

Run:

```powershell
.\.venv\Scripts\python.exe -m ai_guardrail.evaluation.threshold_cli `
  --validation datasets/generated/v2/validation.jsonl `
  --ner-model artifacts/ai-guardrail-ner-en-v2 `
  --output artifacts/thresholds/ai-guardrail-ner-en-v2.selected-threshold.json
```

Expected: a verified threshold from the fixed candidate grid `0.50` through
`0.95`.

### Task 6: Frozen Challenge A/B Comparison

**Files:**
- Generate, ignored: `evaluation/reports/ner-v1-v2-challenge.json`

**Interfaces:**
- Consumes v1 artifact with threshold `0.55`.
- Consumes verified v2 artifact and selected v2 threshold.
- Consumes `datasets/challenge/en-v1.seed.jsonl`.
- Produces aggregate metrics only.

- [ ] **Step 1: Run both detectors**

Use `BenchmarkRunner` or the existing metric functions to calculate strict,
partial, per-entity, character, latency, and exact-record metrics for both
models. Write JSON that contains no record text, entity text, offsets, or raw
predictions.

- [ ] **Step 2: Check privacy and result validity**

Assert the report contains exactly nine evaluated examples, both detector
names, finite metrics, no prompt fragments, and no entity values from the
challenge seed.

- [ ] **Step 3: Recheck v1 identity**

Verify the v1 artifact and manifest SHA-256 values match those recorded before
training.

- [ ] **Step 4: Final verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
uv tool run ruff check .
git status --short
```

Expected: all tests and lint pass; model, threshold, generated dataset, and
evaluation outputs remain ignored; tracked worktree is clean.
