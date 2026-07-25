# Offline evaluation workflow

## 1. Create the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
```

## 2. Generate synthetic data

```powershell
python -c "from pathlib import Path; from ai_guardrail.synthetic.generator import generate_dataset; print(generate_dataset(Path('config/synthetic-v1.yaml'), Path('datasets/generated/v1')))"
```

Generated data remains under `datasets/generated/` and is not committed.

## 3. Train DistilBERT

```powershell
python -m ai_guardrail.ner.train `
  --train datasets/generated/v1/train.jsonl `
  --validation datasets/generated/v1/validation.jsonl `
  --output artifacts/ai-guardrail-ner-en-v1 `
  --seed 20260725
```

Training downloads `distilbert/distilbert-base-cased`; standard tests do not.

## 4. Start Qwen separately

Provide `Qwen3-0.6B-Q4_K_M.gguf` through the approved artifact channel, then
start `llama-server` with a 1,024-token context, a 512-token detector input
limit, and prompt logging disabled:

```powershell
.\tools\llama-server.exe `
  -m .\models\Qwen3-0.6B-Q4_K_M.gguf `
  -c 1024 `
  --host 127.0.0.1 `
  --port 8080 `
  --parallel 1 `
  --log-disable
```

## 5. Select the NER threshold

```powershell
python -m ai_guardrail.evaluation.threshold_cli `
  --validation datasets/generated/v1/validation.jsonl `
  --ner-model artifacts/ai-guardrail-ner-en-v1 `
  --output artifacts/ai-guardrail-ner-en-v1/selected-threshold.json
```

The challenge set is not used for threshold selection.

## 6. Evaluate

```powershell
$thresholdArtifact = 'artifacts/ai-guardrail-ner-en-v1/selected-threshold.json'
$qwenProcess = Get-Process llama-server
$qwenHash = (Get-FileHash -Algorithm SHA256 models/Qwen3-0.6B-Q4_K_M.gguf).Hash.ToLowerInvariant()
$llamaVersion = (& .\tools\llama-server.exe --version | Select-Object -First 1)

python -m ai_guardrail.evaluation.cli `
  --challenge datasets/generated/v1/challenge.reviewed.jsonl `
  --regex-config config/regex-patterns.yaml `
  --ner-model artifacts/ai-guardrail-ner-en-v1 `
  --ner-threshold-artifact $thresholdArtifact `
  --qwen-url http://127.0.0.1:8080 `
  --qwen-pid $qwenProcess.Id `
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

## 7. Acceptance review

- Qwen JSON parse rate is at least 99%.
- No invalid Qwen candidate enters normalized output.
- `PERSON` and `ADDRESS` strict-span F1 are at least 0.85.
- `CUSTOMER_ID` and `INTERNAL_PROJECT` strict-span F1 are at least 0.75.
- NER P95 is at most 150 ms with 2 vCPU and 2 GiB.
- Qwen P95 is at most 2 seconds with 4 vCPU and 4 GiB.
- Captured logs contain no prompt or entity text.

Failure of a target is an experiment result, not permission to alter the target
or enable masking.
