# Phase 1 Duplicate-Component Comparison

## Purpose

Identify which implementations are truly shared, which are method-specific wrappers, and which must not be selected as canonical ARViT without Phase 2 verification.

## Blob-level comparison

| Component | Distance | Region similarity | SAM | Finding |
|---|---|---|---|---|
| `layers/encoder.py` | SHA `159ba34...` | SHA `159ba34...` | SHA `159ba34...` | Exact duplicate in all three areas. |
| `layers/layers.py` | SHA `812f9ff...` | SHA `812f9ff...` | SHA `812f9ff...` | Exact duplicate in all three areas. |
| `layers/drop.py` | SHA `c42a387...` | SHA `c42a387...` | SHA `c42a387...` | Exact duplicate in all three areas. |
| `utils/positional_encoding.py` | SHA `a43536d...` | SHA `a43536d...` | SHA `a43536d...` | Exact duplicate in all three areas. |
| `GM_Mask.py` | not present | SHA `42e8794...` | SHA `42e8794...` | Exact duplicate between region and SAM. |
| `attention_loss.py` | different distance loss | SHA `4d0548c...` | SHA `4d0548c...` | Exact duplicate between region and SAM. |
| `metrics.py` | distance-specific SHA `5c33b15...` | SHA `1f9e3f1...` | SHA `1f9e3f1...` | Region and SAM are exact duplicates. |
| main model | `ARViT2D.py`, SHA `f184975...` | `ARViT.py`, SHA `630285a...` | `ARViT.py`, SHA `09553c1...` | Region and SAM are functionally identical except import paths; distance is a structural sibling with a different penalty output. |
| visualization | absent | SHA `3664bbb...` | SHA `b005195...` | Near-duplicate family, but imports and method context differ. |

## Shared architecture pattern

All three model families:

- create patch embeddings using a convolution with kernel and stride equal to patch size;
- reshape embeddings into a two-dimensional feature map;
- add fixed two-dimensional sinusoidal positional encodings;
- use the same custom transformer encoder;
- capture attention maps from every encoder layer;
- normalize the final feature map;
- flatten the complete final encoder output into a linear task head;
- omit a classification token;
- store masks and positional tensors with a fixed historical batch dimension.

## Distance model

`distance_based/ARViT2D/ARViT2D.py` additionally:

- constructs a precomputed spatial penalty matrix;
- freezes it as a non-trainable parameter;
- returns it as output item four;
- applies simple min-max rescaling to each attention map;
- hard-codes `R = 256` in attention rescaling.

The constructor defaults do not equal the paper-oriented training configuration. It defaults to 8 heads and hidden dimension 768, while the launch scripts instantiate 12 heads and hidden dimension 516.

## Region-similarity model

`region_similarity_based/ARViT/ARViT.py` additionally:

- computes an input-dependent Gram-distance representation through `GM_Mask`;
- returns it as output item four;
- uses an attention-rescaling path intended to align attention and region representations;
- hard-codes `R = 256`.

The constructor defaults to 8 encoder layers, while launch scripts instantiate 6. The launch scripts use 8 heads and hidden dimension 512.

## SAM model

`SAM.../models/ARViT.py` is functionally the same as the region-similarity ARViT apart from package import paths.

`SAM.../models/SAM.py` is not the complete adversarial system. It wraps the ARViT critic only; its generator member is commented out. The generator and critic are assembled externally in `launch-SAM.py` through FastAI GAN machinery.

## Canonical ARViT decision

No existing model file should be copied directly into `mormille/arvit` and labelled canonical.

A defensible canonical architecture must be reconstructed from the shared intersection after Phase 2 verifies:

- six versus eight encoder-layer defaults;
- 12 versus 8 attention heads;
- hidden dimension 516 versus 512;
- feed-forward structure;
- attention-map representation;
- fixed positional encoding;
- flattening head and parameter counts;
- which differences are architecture choices versus experiment configurations.

The canonical package should expose configuration where scientifically appropriate, while reproduction configurations preserve exact published variants.

## Deduplication strategy

### During Phase 3

Keep all copies unchanged in their extracted historical repositories. This preserves behavior and history.

### During Phases 5–8

1. Implement the verified shared architecture in `arvit`.
2. Add regression tests against each archived model.
3. Make method packages depend on `arvit`.
4. Retain only method-specific penalty, loss, preprocessing, and experiment code.
5. Remove duplicated adapted files only after tests demonstrate equivalent behavior.

## Additional cross-cutting issues

- mutable list defaults appear in multilayer loss constructors;
- wildcard and working-directory-relative imports recur throughout;
- masks and positional tensors are tied to configured batch size;
- local save and data paths recur in launch scripts;
- the region Attention Loss and metrics were copied into SAM, potentially carrying earlier method behavior into the adversarial experiments.

These are modernization targets, not Phase 1 changes.
