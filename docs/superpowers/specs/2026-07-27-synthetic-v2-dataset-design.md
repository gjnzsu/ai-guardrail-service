# Synthetic v2 Dataset Design

## Goal

Create a deterministic English-only synthetic v2 dataset that improves NER
generalization without changing or contaminating the existing v1 release. The
v2 data must address the failure families captured in the frozen challenge v1
seed while avoiding exact copies of those challenge records.

The POC remains limited to `PERSON`, `ADDRESS`, `EMAIL`, `API_KEY`,
`CUSTOMER_ID`, and `INTERNAL_PROJECT`. All values and records remain synthetic.

## Delivery Boundary

- Keep the v1 configuration, generator behavior, templates, catalogs, and
  generated paths unchanged.
- Add `config/synthetic-v2.yaml` as an independent generation entry point.
- Generate local artifacts under `datasets/generated/v2/`.
- Do not commit generated JSONL files.
- Keep `datasets/challenge/en-v1.seed.jsonl` frozen and excluded from training,
  validation, and threshold selection.
- Do not retrain the NER model or change inference behavior in this change.

The default v2 release contains:

- 4,000 training records;
- 800 validation records;
- 200 unreviewed challenge candidates.

## Generation Architecture

The v2 generator uses deterministic compositional generation. A record is
assembled from a split-specific template family, synthetic catalog values, and
optional controlled variations such as casing, punctuation, line breaks,
possessives, abbreviations, and light informal phrasing.

V2 generation remains behind the existing `generate_dataset` entry point. The
configuration version selects the v1 or v2 strategy, so existing callers do not
need a second public API. Version-specific templates and catalogs remain
separate modules to prevent accidental v1 drift.

For each split, generation:

1. selects a split-specific template family;
2. selects only values assigned to that split;
3. applies deterministic variations using the configured seed;
4. derives entity spans from inserted values rather than post-hoc matching;
5. rejects duplicate text;
6. validates the completed record;
7. continues until the configured count is reached or a bounded attempt limit
   is exceeded.

Exhausting the attempt limit is a generation error rather than permission to
emit duplicate or undersized data.

## Coverage Model

### Positive examples

The dataset includes single-entity and multi-entity records in varied entity
orders and contexts. Coverage specifically includes:

- lowercase, title-case, and uppercase person forms;
- possessive person names and informal requests;
- project references using `Project`, `project`, and `PROJECT`;
- full `Project <codename>` span boundaries across punctuation and line breaks;
- alphabetic and numeric email local parts;
- formatted and numeric customer identifiers in explicit customer or account
  contexts;
- API keys with hyphenated, underscored, and grouped forms;
- one-line and multi-line addresses.

These examples represent the same failure families as challenge v1 without
copying its complete sentences or reserved challenge values.

### Hard negatives

Approximately 20 percent of records contain no entities or contain nearby
non-entity text. Hard-negative families include:

- ordinary numbers, percentages, backlog counts, and dates that are not
  customer identifiers;
- uses of `project`, `apex`, and other codenames as ordinary words;
- usernames, filenames, and incomplete domains that are not email addresses;
- version strings and build identifiers that are not API keys;
- lowercase ordinary words in grammatical positions similar to names;
- status colors, street-like phrases, and other context distractors.

Hard negatives use split-specific families and values just like positive
examples.

## Split Isolation

Training and validation must be disjoint in all of the following dimensions:

- record identifiers;
- template families;
- complete text hashes;
- catalog entity values.

Challenge candidates use a third set of template families and values. The
frozen challenge v1 seed remains an external benchmark and is not copied into
any v2 split.

The generator must fail if isolation checks do not hold.

## Data Quality Gates

Automated tests enforce:

- deterministic byte-identical output for the same configuration and seed;
- no change to deterministic v1 output;
- at least 95 percent unique text in train and validation;
- the exact configured record count for every split;
- valid entity types, offsets, and non-empty entity text;
- representation of all six entity types in train and validation;
- split-specific template families and catalog values;
- at least 15 percent and at most 25 percent entity-free hard negatives in
  train and validation;
- explicit v2 coverage for lowercase people, case-varied project names,
  numeric-local-part emails, numeric customer identifiers, and multiline
  inputs;
- no exact text overlap with the frozen challenge v1 seed.

The implementation may exceed these minimum coverage requirements but must not
weaken them without a new design decision.

## Error Handling

Configuration validation rejects unknown fields, unsupported versions,
non-positive counts, and invalid seeds. Generation rejects malformed templates,
unknown entity placeholders, invalid span boundaries, duplicate text, split
leakage, and insufficient combinatorial capacity.

Errors must identify the failed split and quality rule without including raw
generated text or entity values.

## Verification

Implementation follows test-driven development:

1. add focused tests for v2 determinism, coverage, uniqueness, and isolation;
2. observe the tests fail because v2 is not implemented;
3. implement the smallest compositional generator that satisfies them;
4. generate `datasets/generated/v2/` locally;
5. run focused generator tests, the full pytest suite, and Ruff;
6. inspect generated aggregate counts and hashes without logging record text;
7. inspect Git status to confirm generated datasets remain untracked and
   uncommitted.

The v2 dataset is ready for later retraining only when all quality gates pass.
Model training and benchmark comparison are separate follow-up work.
