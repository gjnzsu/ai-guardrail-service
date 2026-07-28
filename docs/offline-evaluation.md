# Offline evaluation workflow

## 1. Create the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
```

## 2. Generate synthetic data

Generate v1:

```powershell
python -c "from pathlib import Path; from ai_guardrail.synthetic.generator import generate_dataset; print(generate_dataset(Path('config/synthetic-v1.yaml'), Path('datasets/generated/v1')))"
```

Generate v2:

```powershell
python -c "from pathlib import Path; from ai_guardrail.synthetic.generator import generate_dataset; print(generate_dataset(Path('config/synthetic-v2.yaml'), Path('datasets/generated/v2')))"
```

Generated data remains under `datasets/generated/` and is not committed.

## 3. Review and promote the challenge release

The generator writes `challenge.candidates.jsonl`, not an evaluation release.
Manually inspect every challenge record and its entity spans, then promote it
only after that review. Record the release checksum beside the reviewed file:

```powershell
# After completing the human review of every candidate record:
Copy-Item datasets/generated/v1/challenge.candidates.jsonl datasets/generated/v1/challenge.reviewed.jsonl
$challengePath = 'datasets/generated/v1/challenge.reviewed.jsonl'
$challengeSha256 = (Get-FileHash -Algorithm SHA256 $challengePath).Hash.ToLowerInvariant()
Set-Content -NoNewline "$challengePath.sha256" $challengeSha256
```

The reviewed release must remain distinct from training and validation template
families. Do not run the benchmark with an unreviewed candidates file.

## 4. Train DistilBERT

Train v1:

```powershell
python -m ai_guardrail.ner.train `
  --release-version v1 `
  --train datasets/generated/v1/train.jsonl `
  --validation datasets/generated/v1/validation.jsonl `
  --output artifacts/ai-guardrail-ner-en-v1 `
  --seed 20260725
```

Train v2 with the same training seed for a controlled dataset comparison:

```powershell
python -m ai_guardrail.ner.train `
  --release-version v2 `
  --train datasets/generated/v2/train.jsonl `
  --validation datasets/generated/v2/validation.jsonl `
  --output artifacts/ai-guardrail-ner-en-v2 `
  --seed 20260725
```

Training downloads `distilbert/distilbert-base-cased`; standard tests do not.

## 5. Start Qwen separately

Provide `Qwen3-0.6B-Q4_K_M.gguf` through the approved artifact channel, then
obtain an approved `llama-server.exe` binary. In a second terminal, set its
local path, then run it in the foreground with a 1,024-token context, a
512-token detector input limit, and prompt logging disabled. Keep one
`llama-server` instance running:

```powershell
$llamaServer = 'C:\approved-tools\llama-server.exe'
& $llamaServer `
  -m .\models\Qwen3-0.6B-Q4_K_M.gguf `
  -c 1024 `
  --host 127.0.0.1 `
  --port 8080 `
  --parallel 1 `
  --log-disable
```

## 6. Select the NER threshold

Select the v1 threshold from v1 validation:

```powershell
python -m ai_guardrail.evaluation.threshold_cli `
  --validation datasets/generated/v1/validation.jsonl `
  --ner-model artifacts/ai-guardrail-ner-en-v1 `
  --output artifacts/thresholds/ai-guardrail-ner-en-v1.selected-threshold.json
```

Select the v2 threshold independently from v2 validation:

```powershell
python -m ai_guardrail.evaluation.threshold_cli `
  --validation datasets/generated/v2/validation.jsonl `
  --ner-model artifacts/ai-guardrail-ner-en-v2 `
  --output artifacts/thresholds/ai-guardrail-ner-en-v2.selected-threshold.json
```

The threshold artifact must remain outside the immutable model artifact
directory. The challenge set is not used for threshold selection.

For the initial NER-only A/B comparison, evaluate both releases against the
same frozen `datasets/challenge/en-v1.seed.jsonl`. The v2 generated
`challenge.candidates.jsonl` remains unreviewed and is not an evaluation
release.

## 7. Evaluate

```powershell
$thresholdArtifact = 'artifacts/thresholds/ai-guardrail-ner-en-v1.selected-threshold.json'
$challengePath = 'datasets/generated/v1/challenge.reviewed.jsonl'
$recordedChallengeSha256 = (Get-Content "$challengePath.sha256").Trim().ToLowerInvariant()
$actualChallengeSha256 = (Get-FileHash -Algorithm SHA256 $challengePath).Hash.ToLowerInvariant()
if ($actualChallengeSha256 -ne $recordedChallengeSha256) { throw 'reviewed challenge checksum mismatch' }
$qwenPids = @(Get-NetTCPConnection -State Listen -LocalPort 8080 | Select-Object -ExpandProperty OwningProcess -Unique)
if ($qwenPids.Count -ne 1) { throw 'expected exactly one llama-server listener on port 8080' }
$qwenPid = [int]$qwenPids[0]
$qwenHash = (Get-FileHash -Algorithm SHA256 models/Qwen3-0.6B-Q4_K_M.gguf).Hash.ToLowerInvariant()
$llamaServer = 'C:\approved-tools\llama-server.exe'
$llamaVersion = (& $llamaServer --version | Select-Object -First 1)

python -m ai_guardrail.evaluation.cli `
  --challenge datasets/generated/v1/challenge.reviewed.jsonl `
  --regex-config config/regex-patterns.yaml `
  --ner-model artifacts/ai-guardrail-ner-en-v1 `
  --ner-threshold-artifact $thresholdArtifact `
  --qwen-url http://127.0.0.1:8080 `
  --qwen-pid $qwenPid `
  --qwen-sha256 $qwenHash `
  --llama-version $llamaVersion `
  --guardrail-cpu-limit 2 `
  --guardrail-memory-limit-mib 2048 `
  --qwen-cpu-limit 4 `
  --qwen-memory-limit-mib 4096 `
  --repetitions 3 `
  --output-dir evaluation/reports/v1
```

Store JSON and Markdown reports under `evaluation/reports/`; the directory is
ignored by Git. Reports contain aggregate metrics only and no example text.
The CPU and RAM arguments are operator declarations recorded as provenance;
they do not limit either process. Treat results as acceptance evidence only
when those limits are independently enforced in an external controlled
environment, such as configured Docker Desktop/container limits or an
equivalent runtime mechanism.

## 8. Acceptance review

- Qwen JSON parse rate is at least 99%.
- No invalid Qwen candidate enters normalized output.
- `PERSON` and `ADDRESS` strict-span F1 are at least 0.85.
- `CUSTOMER_ID` and `INTERNAL_PROJECT` strict-span F1 are at least 0.75.
- NER P95 is at most 150 ms with 2 vCPU and 2 GiB.
- Qwen P95 is at most 2 seconds with 4 vCPU and 4 GiB.
- Captured logs contain no prompt or entity text.

Failure of a target is an experiment result, not permission to alter the target
or enable masking.
