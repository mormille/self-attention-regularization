# Phase 1 Migration Map

**Audit date:** 2026-07-20  
**Historical source:** `mormille/self-attention-regularization`  
**Frozen source ref:** `archive/phase-0-source-baseline-2026-07-20`  
**Baseline commit:** `9f294c2a19fb855dd6230d6fcb38607e36fcac7e`

This is a static inventory and destination map. It does not move or modify research code.

## Inventory summary

The frozen source tree contains **81 currently tracked paths**:

- 27 under `SAM - Adversarial Self-Attention Misdirection/`;
- 19 under `distance_based/`;
- 33 under `region_similarity_based/`;
- 2 root-level files.

The history additionally contains a removed Python `.gitignore` and 15 removed SAM checkpoint pointers.

## Repository-level destination map

| Historical scope | Historical extraction target | Maintained scope |
|---|---|---|
| Shared model code repeated under all three method directories | `mormille/arvit` after verification | Unregularized ARViT architecture, patch embedding, encoder, attention extraction and positional encoding |
| `distance_based/` | `mormille/arvit-distance-regularization` | Manhattan penalty matrix, Distance Loss, experiments and reproductions |
| `region_similarity_based/` | `mormille/arvit-region-similarity-regularization` | Gram-region distances, attention scaling, Attention Loss, experiments and reproductions |
| `SAM - Adversarial Self-Attention Misdirection/` | `mormille/sam-self-attention-misdirection` | Experimental generator, critic integration, SAM losses and adversarial training |
| root files and original history | `mormille/self-attention-regularization` | Historical source and eventual umbrella/index |

## Classification rules

These rules classify every path in the exhaustive tree below.

| Path pattern | Classification | Treatment |
|---|---|---|
| root `README.md` | documentation | Keep in umbrella; replace only in Phase 12. |
| root `__init__.py` | obsolete aggregation glue | Preserve historically; do not migrate into maintained packages. |
| `*/layers/encoder.py`, `*/layers/layers.py`, `*/layers/drop.py`, `*/utils/positional_encoding.py` | shared architecture / adapted infrastructure | Preserve in each historical extraction; later replace or consolidate in `arvit` after provenance and regression verification. |
| primary `ARViT*.py` model files | shared architecture wrapper plus method integration | Preserve historically; extract common base architecture only after Phase 2. |
| `distance_based/.../penalty_matrix.py`, `distance_based/losses/distance_loss.py` | distance-specific scientific code | Migrate to distance repository and verify against the distance paper. |
| region `GM_Mask.py` and `attention_loss.py` | region-similarity scientific code | Migrate to region repository and verify against the final paper. |
| SAM copies of `GM_Mask.py` and `attention_loss.py` | duplicated region-similarity scientific code | Preserve in SAM archive; later reuse a verified implementation rather than maintain another copy. |
| `sam_loss.py`, `launch-SAM.py`, `models/SAM.py` | SAM-specific experimental code | Migrate to SAM repository and verify against the SAM paper. |
| `models/unet.py` | adversarial generator / adapted infrastructure | Preserve historically; major provenance and licence decision required before maintained release. |
| `models/utils/fastai_gan.py` | copied/adapted training infrastructure | Preserve historically; identify exact FastAI source/version or replace with public APIs. |
| `launch-*.py` | training or evaluation scripts | Migrate historically; modernize only after historical reproduction. |
| `visualization/*` | visualization tooling | Migrate with method; later remove hard-coded data and file assumptions. |
| `*.ipynb` | notebook / evaluation / visualization artifact | Preserve historically; later replace with reproducible examples. |
| `old_attention_loss.py` | obsolete historical scientific code | Preserve in archive but exclude from maintained public API. |
| `metrics.py` | evaluation glue | Preserve historically; later replace with focused metrics and tests. |
| `__init__.py` below method directories | packaging glue | Preserve during extraction; replace in modern package layouts. |
| `pretrained_models/*.pth` | Git LFS checkpoint pointer / generated artifact | Preserve in archive only; do not keep in maintained normal Git history. |

## Exhaustive tracked tree

```text
README.md
__init__.py
SAM - Adversarial Self-Attention Misdirection/ARViT.ipynb
SAM - Adversarial Self-Attention Misdirection/__init__.py
SAM - Adversarial Self-Attention Misdirection/launch-ARViT-MultiLayer.py
SAM - Adversarial Self-Attention Misdirection/launch-ARViT.py
SAM - Adversarial Self-Attention Misdirection/launch-Finetune.py
SAM - Adversarial Self-Attention Misdirection/launch-SAM.py
SAM - Adversarial Self-Attention Misdirection/losses/__init__.py
SAM - Adversarial Self-Attention Misdirection/losses/attention_loss.py
SAM - Adversarial Self-Attention Misdirection/losses/metrics.py
SAM - Adversarial Self-Attention Misdirection/losses/old_attention_loss.py
SAM - Adversarial Self-Attention Misdirection/losses/sam_loss.py
SAM - Adversarial Self-Attention Misdirection/models/ARViT.py
SAM - Adversarial Self-Attention Misdirection/models/SAM.py
SAM - Adversarial Self-Attention Misdirection/models/__init__.py
SAM - Adversarial Self-Attention Misdirection/models/layers/__init__.py
SAM - Adversarial Self-Attention Misdirection/models/layers/drop.py
SAM - Adversarial Self-Attention Misdirection/models/layers/encoder.py
SAM - Adversarial Self-Attention Misdirection/models/layers/layers.py
SAM - Adversarial Self-Attention Misdirection/models/unet.py
SAM - Adversarial Self-Attention Misdirection/models/utils/GM_Mask.py
SAM - Adversarial Self-Attention Misdirection/models/utils/__init__.py
SAM - Adversarial Self-Attention Misdirection/models/utils/fastai_gan.py
SAM - Adversarial Self-Attention Misdirection/models/utils/positional_encoding.py
SAM - Adversarial Self-Attention Misdirection/visualization/__init__.py
SAM - Adversarial Self-Attention Misdirection/visualization/view_functions.py
distance_based/ARViT-2D.ipynb
distance_based/ARViT2D/ARViT2D.py
distance_based/ARViT2D/__init__.py
distance_based/ARViT2D/layers/__init__.py
distance_based/ARViT2D/layers/drop.py
distance_based/ARViT2D/layers/encoder.py
distance_based/ARViT2D/layers/layers.py
distance_based/ARViT2D/utils/__init__.py
distance_based/ARViT2D/utils/penalty_matrix.py
distance_based/ARViT2D/utils/positional_encoding.py
distance_based/__init__.py
distance_based/launch-ARViT2D-MultiLayer.py
distance_based/launch-ARViT2D.py
distance_based/losses/__init__.py
distance_based/losses/distance_loss.py
distance_based/losses/metrics.py
distance_based/pretrained_models/ARViT2D-Base.pth
distance_based/pretrained_models/ARViT2D-L2.pth
distance_based/pretrained_models/ARViT2D-L3.pth
region_similarity_based/ARViT.ipynb
region_similarity_based/ARViT/ARViT.py
region_similarity_based/ARViT/__init__.py
region_similarity_based/ARViT/layers/__init__.py
region_similarity_based/ARViT/layers/drop.py
region_similarity_based/ARViT/layers/encoder.py
region_similarity_based/ARViT/layers/layers.py
region_similarity_based/ARViT/utils/GM_Mask.py
region_similarity_based/ARViT/utils/__init__.py
region_similarity_based/ARViT/utils/positional_encoding.py
region_similarity_based/__init__.py
region_similarity_based/launch-ARViT-MultiLayer.py
region_similarity_based/launch-ARViT.py
region_similarity_based/launch-Finetune.py
region_similarity_based/losses/__init__.py
region_similarity_based/losses/attention_loss.py
region_similarity_based/losses/metrics.py
region_similarity_based/losses/old_attention_loss.py
region_similarity_based/pretrained_models/ARViT-Base.pth
region_similarity_based/pretrained_models/ARViT-FULL.pth
region_similarity_based/pretrained_models/ARViT-L-2-3-4-5.pth
region_similarity_based/pretrained_models/ARViT-L1-G32.pth
region_similarity_based/pretrained_models/ARViT-L1.pth
region_similarity_based/pretrained_models/ARViT-L2-G32.pth
region_similarity_based/pretrained_models/ARViT-L2.pth
region_similarity_based/pretrained_models/ARViT-L3-G32.pth
region_similarity_based/pretrained_models/ARViT-L3.pth
region_similarity_based/pretrained_models/ARViT-L4-G32.pth
region_similarity_based/pretrained_models/ARViT-L4.pth
region_similarity_based/pretrained_models/ARViT-L5-G32.pth
region_similarity_based/pretrained_models/ARViT-L5.pth
region_similarity_based/pretrained_models/ARViT-L6-G32.pth
region_similarity_based/pretrained_models/ARViT-L6.pth
region_similarity_based/visualization/__init__.py
region_similarity_based/visualization/view_functions.py
```

## History-only paths

The initial commit contained a standard Python `.gitignore`, which was later removed. A new project-specific `.gitignore` should be authored after extraction instead of restoring it unchanged.

The final historical commits also removed 15 Git LFS pointer files under the SAM `pretrained_models/` directory. Their names cover base, full, multilayer, L1–L6 and G32 variants. Their underlying LFS-object availability is unknown and belongs to Phase 11.

## Extraction recommendation

1. Extract each method directory with relevant Git history.
2. Tag each unmodified imported state as `v0.1.0-research-archive`.
3. Preserve duplicated architecture files in those archive states.
4. Build and verify the independent `arvit` package.
5. Replace duplicated architecture only in later modernization pull requests.
6. Keep checkpoints out of maintained normal Git history.

## Exit assessment

Every currently tracked path has a proposed destination and classification through the repository map, path rules, and exhaustive tree. Obsolete and generated files remain recoverable in the historical archive rather than being silently discarded.
