# Dataset policy

Only fictitious synthetic values may be used in this repository.

- `datasets/generated/` is reproducible output and is ignored by Git.
- `datasets/challenge/en-v1.seed.jsonl` is a small reviewed smoke fixture.
- No production prompt, customer identifier, credential, or copied internal
  project name may be committed.
- Every generated span must be verified by slicing the source text.
- A challenge release is accepted only after every line is manually reviewed
  for entity type, exact offsets, ambiguity, and absence of real information.
