# Phase 2 — Paper-to-Code Verification

## Scope

This directory compares the frozen historical implementation at commit
`9f294c2a19fb855dd6230d6fcb38607e36fcac7e` with the three research papers:

1. *Regularizing self-attention on vision transformers with 2D spatial distance loss*.
2. *Introducing inductive bias on Vision Transformers through Gram matrix similarity based regularization*.
3. *Adversarial Self-attention Misdirection: Improving vision transformers performance with adversarial pre-training*.

The migration plan states that the papers are the scientific sources of truth. This phase therefore documents where the historical implementation agrees with the papers and where it represents an earlier, partial, or different experimental formulation.

## Deliverables

- [`paper-code-comparison.md`](paper-code-comparison.md)
- [`formula-to-function-mapping.md`](formula-to-function-mapping.md)
- [`confirmed-discrepancies-and-likely-bugs.md`](confirmed-discrepancies-and-likely-bugs.md)
- [`scientifically-safe-change-list.md`](scientifically-safe-change-list.md)
- [`questions-for-luiz.md`](questions-for-luiz.md)

## Overall conclusion

The repository contains historically valuable implementations of all three research directions, but the current code should not be presented as a faithful executable reproduction of all final paper formulations.

- The distance implementation is the closest to its paper, but its loss reduction, logarithm, attention normalization, and available training scripts differ from the published method.
- The region-similarity implementation contains the intended high-level pipeline, but its pairwise distance, bias, loss reduction, and 32×32-region scaling differ materially from the final paper.
- The SAM implementation is an earlier adversarial attention experiment rather than a direct implementation of the two-page paper's similarity-matrix objective.
- ARViT-Base can be separated as a reusable architecture, but the published distance configuration and the later region/SAM configuration should be represented as named variants rather than silently treated as identical.

## Phase 3 implication

Phase 3 may now preserve and extract the three historical directories as archival implementations. The extracted repositories must carry a clear historical-code notice. The standalone `arvit` repository should use a provenance-documented baseline derived primarily from the distance-paper implementation, while preserving the later 512-dimensional, eight-head configuration as a separate variant.

No scientific code is modified by this phase.