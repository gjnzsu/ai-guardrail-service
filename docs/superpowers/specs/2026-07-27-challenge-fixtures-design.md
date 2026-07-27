# Natural-Language Challenge Fixtures

## Goal

Add five reviewed English-only synthetic records to the committed challenge
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
3. A request for Project Apex followed by a synthetic numeric-local-part
   email address.
   - Label `Project Apex` as `INTERNAL_PROJECT` at `[23, 35)`.
   - Label `30156758@example.test` as `EMAIL` at `[53, 74)`.
   - Use the family `manual-project-numeric-email-challenge`.
4. The same request followed by a synthetic alphabetic-local-part email
   address.
   - Label `Project Apex` as `INTERNAL_PROJECT` at `[23, 35)`.
   - Label `project.owner@example.test` as `EMAIL` at `[53, 79)`.
   - Use the family `manual-project-email-challenge`.
5. The same synthetic email request with the project codename in uppercase.
   - Label `Project APEX` as `INTERNAL_PROJECT` at `[23, 35)`.
   - Label `project.owner@example.test` as `EMAIL` at `[53, 79)`.
   - Use the family `manual-uppercase-project-email-challenge`.

The committed records must use only the synthetic `example.test` addresses.
The possibly real address used during interactive smoke testing must not be
persisted.

## Implementation

- Append `challenge-seed-000005` through `challenge-seed-000009` to
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
