# NER v2 Training Support Design

## Goal

Train and evaluate `ai-guardrail-ner-en-v2` with the generated v2 dataset while
preserving the existing v1 artifact, validation behavior, and reproducibility.
The first comparison changes only the dataset release and artifact identity.
The base checkpoint and training hyperparameters remain unchanged.

## Release Profiles

The NER artifact layer will define two trusted release profiles:

| Release | Artifact name | Generator version | Dataset version |
|---|---|---|---|
| v1 | `ai-guardrail-ner-en-v1` | `v1` | `v1` |
| v2 | `ai-guardrail-ner-en-v2` | `v2` | `v2` |

Profiles are internal allowlisted values, not arbitrary user-provided metadata.
The existing v1 constants remain backward-compatible aliases so current tests,
artifacts, and callers keep their behavior.

## Training Interface

The training CLI adds `--release-version` with allowlisted values `v1` and
`v2`. It defaults to `v1`.

Training validates that:

- the output directory name matches the selected profile;
- every training and validation record uses the selected generator version;
- train and validation remain disjoint by identifier, template family, and
  content hash;
- the base checkpoint remains the pinned
  `distilbert/distilbert-base-cased` revision;
- the output directory is local and empty.

The v2 command uses:

```powershell
python -m ai_guardrail.ner.train `
  --release-version v2 `
  --train datasets/generated/v2/train.jsonl `
  --validation datasets/generated/v2/validation.jsonl `
  --output artifacts/ai-guardrail-ner-en-v2 `
  --seed 20260725
```

For a controlled A/B comparison, both releases retain the current three
epochs, batch sizes, learning rate, weight decay, BIO label mapping, and
`eval_loss` checkpoint selection. V1 and v2 also use the same training seed,
`20260725`, so classification-head initialization, data shuffling, dropout, and
other seeded training operations are controlled. The v2 dataset generator
continues to use its release seed, `20260727`; changing the generated dataset
is the experimental treatment.

## Artifact Verification and Loading

Manifest creation records the selected release's artifact name, generator
version, and dataset version. Verification reads the artifact identity and
requires it to match one complete trusted profile. Mixed combinations fail
closed.

Verified temporary snapshots use the verified artifact name instead of the
hardcoded v1 name. `NerDetector.load` reports the verified artifact identity as
its model version. Existing v1 artifacts remain valid without migration.

## Threshold Selection and Comparison

Threshold selection remains validation-only. Once v2 training succeeds:

1. reproduce or load the v1 threshold from v1 validation;
2. select the v2 threshold from v2 validation;
3. evaluate both artifacts against the same frozen
   `datasets/challenge/en-v1.seed.jsonl`;
4. report aggregate strict-span, partial-span, character, per-entity, latency,
   and exact-record metrics without writing prompt or entity text.

Challenge records never enter training, checkpoint selection, or threshold
selection.

## Failure Handling

- Unknown releases, mismatched output names, mixed-version datasets, and
  inconsistent manifests fail before ML dependencies are loaded.
- Training writes only to a new empty v2 artifact directory and never
  overwrites v1.
- A failed training run is not accepted as an artifact.
- Threshold or challenge evaluation failure leaves both model artifacts
  unchanged.
- Generated model, threshold, and evaluation artifacts remain ignored by Git.

## Testing and Verification

Implementation follows TDD:

1. add failing tests for selecting v2, rejecting mixed profiles, verifying v2
   manifests, preserving v1 defaults, and loading a v2 detector identity;
2. implement the minimal trusted profile abstraction;
3. run focused and full tests plus Ruff;
4. run one v2 CPU training with seed `20260725`;
5. select the v2 validation threshold;
6. run the frozen challenge comparison;
7. verify aggregate results, artifact checksums, privacy constraints, Git
   status, and the unchanged v1 artifact.

Multi-seed variance measurement, per-entity thresholds, alternative checkpoint
metrics, and model architecture changes remain follow-up experiments.
