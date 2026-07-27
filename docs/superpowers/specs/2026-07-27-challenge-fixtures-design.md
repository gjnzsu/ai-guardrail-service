# Natural-Language Challenge Fixtures

## Goal

Add two reviewed English-only synthetic records to the committed challenge
seed. They preserve detector failures observed during local NER and Qwen smoke
testing without expanding the POC into retraining or model tuning.

## Records

1. A lowercase person name followed by a numeric account identifier.
   - Label `tommy` as `PERSON` at `[0, 5)`.
   - Label `123456` as `CUSTOMER_ID` at `[16, 22)`.
   - Use the family `manual-lowercase-account-challenge`.
2. A natural request for the status of project Apex.
   - Label the complete phrase `project Apex` as `INTERNAL_PROJECT` at
     `[23, 35)`, following the existing convention that includes the
     `Project` prefix.
   - Do not label the status color or backlog numbers.
   - Use the family `manual-project-status-challenge`.

## Implementation

- Append `challenge-seed-000005` and `challenge-seed-000006` to
  `datasets/challenge/en-v1.seed.jsonl`.
- Extend the existing committed-seed test with the exact entity text and
  types for both records.
- Do not change the synthetic generator, generated datasets, model artifacts,
  inference behavior, or deployment configuration.

## Verification

- First update the committed-seed expectation and observe it fail because the
  two records are absent.
- Add the two records and rerun the focused test.
- Run Ruff and the full pytest suite.
