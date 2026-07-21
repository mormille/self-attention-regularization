# Scientifically Safe Change List

This document distinguishes changes that are safe during historical extraction from changes that must wait for reproduction tests and explicit scientific decisions.

---

## 1. Safe during Phase 3 — historical extraction

The following changes do not alter model behavior and are safe when populating the target repositories.

### Preserve method directories as historical snapshots

- Extract `distance_based/` into `arvit-distance-regularization`.
- Extract `region_similarity_based/` into `arvit-region-similarity-regularization`.
- Extract `SAM - Adversarial Self-Attention Misdirection/` into `sam-self-attention-misdirection`.
- Preserve relevant commit history where technically practical.
- Move each extracted subdirectory's contents to the target repository root without modifying formulas or tensor operations.
- Record the exact source commit and source path.
- Tag the imported state as a historical research archive.

Suggested tag:

```text
v0.1.0-research-archive
```

### Add archival metadata after import

It is safe to add documentation that states:

- the code is a preserved historical implementation;
- it has not yet been reproduced under a modern environment;
- differences from the final papers are recorded in the Phase 2 verification;
- the repository must not yet be treated as an exact reproduction package;
- licensing is under provenance review.

### Keep checkpoints out of ordinary source history

- Preserve Git LFS pointers only as historical metadata where extraction naturally retains them.
- Do not re-upload large `.pth`, `.pt`, or `.pkl` artifacts into standard Git history.
- Do not claim that a checkpoint corresponds to a published result until metadata is verified.

### Preserve names and experimental scripts

- Retain historical filenames, launch scripts, notebooks, and obsolete variants in the archive import.
- Do not silently delete multi-layer losses, old loss versions, or incomplete SAM utilities.
- Mark uncertain files in provenance notes rather than deciding their scientific status during extraction.

---

## 2. Safe approach for the standalone `arvit` repository

There is no independent historical ARViT directory, so its Phase 3 baseline must be constructed with explicit provenance.

### Recommended baseline

Use the shared architecture from the distance-paper implementation as the primary historical source because that paper introduced and most precisely specified ARViT.

Record each file's origin, for example:

```text
source repository: mormille/self-attention-regularization
source commit: 9f294c2a19fb855dd6230d6fcb38607e36fcac7e
source area: distance_based/ARViT2D/
```

### Safe contents

The standalone historical baseline may include:

- patch embedding;
- encoder and transformer layers;
- fixed positional encoding;
- base model forward path;
- attention extraction;
- classification/pretext head;
- configuration notes for the published variants.

It should not include:

- the Manhattan penalty matrix;
- Distance Loss;
- Gram-matrix region computation;
- Attention Loss;
- SAM generator or adversarial losses.

### Named configurations

Document, but do not silently merge, at least two historical configurations:

- **distance-paper variant:** six layers, 12 heads, hidden dimension 516;
- **region/SAM variant:** six layers, eight heads, hidden dimension 512.

The initial extraction should preserve architecture behavior. A clean configurable package belongs to Phase 5.

---

## 3. Safe documentation changes before reproduction

The following claims can be made because they are directly supported by the papers and static code review:

- what each method was intended to do;
- which paper corresponds to each repository;
- the published architecture and training configurations;
- the historical file structure;
- the existence of documented paper-code discrepancies;
- that SAM was preliminary and lacked reported final validation;
- that the current code has not yet been executed in a recovered environment.

The following claims are not yet safe:

- “reproduces the paper”;
- “runs on modern PyTorch”;
- “the included checkpoint achieves X%”;
- “all equations are implemented exactly”;
- “SAM improves generalization” as an experimentally verified result;
- “supports arbitrary image and patch sizes.”

---

## 4. Changes that must wait until after historical preservation

These changes are scientifically justified but must not be mixed into the archival import.

### ARViT modernization

- remove batch-size-bound position/mask state;
- use registered buffers where appropriate;
- generalize token and spatial dimensions;
- expose raw and normalized attention through explicit interfaces;
- decide whether to retain the historical `0.3` position-encoding scale;
- implement named published configurations;
- verify parameter counts and checkpoint compatibility.

### Distance-paper-faithful implementation

- calculate the penalty scalar independently for each example;
- apply `max(l_x, 0)` per example;
- calculate `log(l_x* + 1)` per example;
- sum or reduce across examples explicitly;
- decide, based on evidence, whether paper-faithful mode uses raw or normalized attention;
- remove hard-coded 256-token assumptions;
- vectorize Manhattan-distance generation;
- add the published ImageNet rotation configuration separately from lightweight examples.

### Region-paper-faithful implementation

- calculate pairwise Gram-matrix MSE;
- retain explicit, stable normalization with zero-range handling;
- implement general double pooling from patch to region resolution;
- support both 16×16 and 32×32 region configurations;
- enforce `rho >= 0` and add the published `rho=0.3` configuration;
- calculate per-example `log(l_x + 1)`;
- add the published `lambda=0.005` configuration;
- document whether the historical Gram scale factor is retained or removed.

### SAM paper-faithful implementation

- decide whether to reproduce the cited lens architecture or adopt a clearly labelled substitute;
- calculate similarity matrices for both original and generated images;
- preserve gradients through the generated-image similarity path;
- implement `MSE(S, S*)`;
- implement `Lrec - lambda Lm - alpha LF` with explicit reductions;
- document and test the alternating optimization schedule;
- keep the repository labelled experimental until new results are obtained.

---

## 5. Safe engineering corrections after an archival tag exists

Once the historical snapshot is tagged and regression behavior can be checked, the following corrections are safe as separate, documented commits:

- replace working-directory-sensitive imports with package-relative imports;
- replace mutable list defaults;
- add validation for image/patch/region divisibility;
- add epsilon or conditional handling to normalization;
- move machine-specific paths into command-line/configuration parameters;
- support CPU and single-GPU smoke tests;
- replace deprecated distributed launch syntax;
- make output directories configurable;
- add tests without altering formulas;
- add formatters, linters, and CI;
- distinguish historical and paper-faithful modes where both need preservation.

Each behavior-changing correction must state whether it:

1. fixes a software defect while preserving the intended formula;
2. changes historical behavior to match a final paper;
3. introduces a new modern implementation.

---

## 6. Phase 3 authorization boundary

Phase 2 supports proceeding to Phase 3 under this boundary:

> Preserve first. Extract the historical methods without correction, attach provenance and status notices, and create a provenance-documented ARViT baseline. Do not modify scientific formulas, training behavior, or model dimensions during the extraction phase.

This boundary ensures that later corrections can always be compared with the original historical implementation.