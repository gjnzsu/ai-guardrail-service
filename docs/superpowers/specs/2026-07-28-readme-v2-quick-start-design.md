# README v2 Quick Start Design

## Goal

Update the project README so a new contributor can understand the current
v1/v2 NER proof of concept and run the v2 workflow without duplicating the
full offline-evaluation guide.

## Scope

The README will:

- describe the current v1/v2 DistilBERT NER comparison;
- retain the development environment and test commands;
- add concise commands for generating v2 data, training v2, and selecting its
  validation threshold;
- summarize the privacy-safe v1/v2 challenge result, including the eight-record
  clean-subset limitation;
- state that model weights, generated datasets, and reports remain local and
  are ignored by Git; and
- link to `docs/offline-evaluation.md` for the complete Qwen and evaluation
  workflow.

The README will not duplicate the full benchmark command, document production
deployment, or treat the small challenge result as a statistically conclusive
model-quality claim.

## Structure

1. Project overview and current POC scope.
2. Documentation links.
3. Development setup and verification.
4. v2 quick start:
   - generate the deterministic v2 dataset;
   - train the trusted v2 release with seed `20260725`;
   - select the threshold using v2 validation.
5. Current aggregate POC result and sample-size caveat.
6. Local-artifact and privacy boundary.

## Validation

- Run Markdown whitespace validation with `git diff --check`.
- Verify every referenced command and path against the existing CLI and
  `docs/offline-evaluation.md`.
- Run Ruff and the full pytest suite because the README is shipped alongside
  the implementation it describes.
