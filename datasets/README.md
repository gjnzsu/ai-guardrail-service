# Dataset policy

Only fictitious synthetic values may be used in this repository.

- `datasets/generated/` is reproducible output and is ignored by Git.
- `datasets/challenge/en-v1.seed.jsonl` is a small reviewed smoke fixture.
- No production prompt, customer identifier, credential, or copied internal
  project name may be committed.
- Every generated span must be verified by slicing the source text.
- A challenge release is accepted only after every line is manually reviewed
  for entity type, exact offsets, ambiguity, and absence of real information.

Generate the original v1 release with:

```powershell
python -c "from pathlib import Path; from ai_guardrail.synthetic.generator import generate_dataset; print(generate_dataset(Path('config/synthetic-v1.yaml'), Path('datasets/generated/v1')))"
```

Generate the higher-diversity v2 release with:

```powershell
python -c "from pathlib import Path; from ai_guardrail.synthetic.generator import generate_dataset; print(generate_dataset(Path('config/synthetic-v2.yaml'), Path('datasets/generated/v2')))"
```

V2 uses split-isolated compositional templates and catalogs. Its generated
challenge candidates remain unreviewed and must not replace the frozen
`datasets/challenge/en-v1.seed.jsonl` benchmark.
