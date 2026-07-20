# Phase 1 Audit

This directory contains the static repository, duplication, provenance, and dependency audit required before scientific verification or code extraction.

## Documents

- [`migration-map.md`](migration-map.md) — complete classification and proposed destination for every currently tracked path.
- [`duplicate-component-comparison.md`](duplicate-component-comparison.md) — blob-level and functional comparison of repeated architecture code.
- [`provenance-audit.md`](provenance-audit.md) — preliminary origin, attribution, and licence-risk review.
- [`dependency-matrix.md`](dependency-matrix.md) — historical environment evidence and preliminary dependency grouping.
- [`unresolved-questions.md`](unresolved-questions.md) — questions that must be resolved in later phases.

## Scope

This is an audit-only change. It does not:

- move files to the new repositories;
- alter research formulas or behavior;
- add a licence;
- claim that the historical code currently runs;
- select a canonical ARViT implementation.

## Baseline

- Source repository: `mormille/self-attention-regularization`
- Frozen source branch: `archive/phase-0-source-baseline-2026-07-20`
- Baseline commit: `9f294c2a19fb855dd6230d6fcb38607e36fcac7e`
- Current tracked paths audited: 81
- History-only paths noted: 16

## Exit assessment

The Phase 1 exit criterion is met at the documentation level: every important current path has a proposed destination and provenance classification. Scientific correctness is intentionally deferred to Phase 2.
