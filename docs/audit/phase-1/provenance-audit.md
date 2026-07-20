# Phase 1 Provenance Audit

**Status:** Preliminary static audit  
**Source ref:** `archive/phase-0-source-baseline-2026-07-20`

This document identifies likely or explicit origins. It is not a legal opinion and does not authorize relicensing.

## Summary

The repository contains a mixture of:

1. original research code for the distance, region-similarity, and SAM methods;
2. adapted or copied transformer infrastructure;
3. adapted FastAI training internals;
4. a U-Net implementation closely matching a third-party project;
5. generated notebooks and model-checkpoint pointers.

The repository currently has no licence. An MIT, Apache 2.0, or other licence must not be added until the items below are resolved.

## Findings

| Component | Historical paths | Evidence in source | Preliminary origin | Risk / required action |
|---|---|---|---|---|
| Transformer encoder | all three `*/layers/encoder.py` copies | Header states “Copy-paste from torch.nn.Transformer with modifications”; identical blob SHA across all copies | PyTorch transformer structure with wording and modifications associated with Facebook DETR | High attribution risk. Locate the exact upstream revision, retain required notices, or replace it with a clean implementation using public PyTorch APIs. |
| Patch embedding and utility layers | all three `*/layers/layers.py` copies | Header credits Ross Wightman, copyright 2020, and Google Vision Transformer | historical `timm` utility and patch-embedding code | Identify exact upstream files and preserve attribution, or depend on/reimplement supported primitives. |
| DropBlock / DropPath | all three `*/layers/drop.py` copies | Header credits Ross Wightman, copyright 2020 | historical `timm` implementation | Same action as above. |
| 2D sinusoidal positional encoding | all three `*/utils/positional_encoding.py` copies | Identical code and public snippet pattern | likely adapted from `wzlxjtu/PositionalEncoding2D` | Upstream licence is unclear in the inspected repository. Treat as unresolved copyright; replace with a clean implementation or obtain permission. |
| U-Net generator | `SAM.../models/unet.py` | Structure, class names, comments, and linked bug references closely match Milesial/Pytorch-UNet; residual output is locally added | Milesial/Pytorch-UNet derivative | Major compatibility issue because the upstream project is GPL-3.0. Do not publish this file under MIT/Apache without resolving it. Replace it, use GPL-compatible licensing, or obtain legal guidance. |
| FastAI GAN/distributed overrides | `SAM.../models/utils/fastai_gan.py` | Copies and modifies private/internal FastAI GAN and distributed training methods | FastAI v2 internals | Locate the exact FastAI release and methods; retain attribution or eliminate copied internals by using public APIs. |
| ARViT model wrappers | the three main model implementations | Research-specific assembly and outputs built around the shared infrastructure | Luiz Mormille and research collaborators, subject to embedded third-party components | Scientific assembly appears project-specific, but file-level licensing is constrained by included/adapted infrastructure. |
| Distance penalty and loss | `penalty_matrix.py`, `distance_loss.py` | Method-specific equations and naming | original research implementation | Verify against the final paper before identifying it as canonical. |
| Region Gram mask and Attention Loss | `GM_Mask.py`, `attention_loss.py` | Method-specific Gram computations, normalization and losses | original research implementation | Verify final-paper correspondence; exact copies also appear in SAM. |
| SAM losses and workflow | `sam_loss.py`, `launch-SAM.py`, `models/SAM.py` | Research-specific adversarial formulation | original experimental research implementation | Verify signs, component roles, and optimization procedure against the paper. |
| Visualization code | both `visualization/view_functions.py` files | Custom plotting around FastAI/PyTorch | mostly project-specific, with duplicated method code | Compare copies; later remove unused imports and hard-coded files. |
| Notebooks | three `.ipynb` files | Executed research exploration with local paths and outputs | project-generated artifacts | Preserve historically; clean local paths, external URLs and outputs in maintained examples. |
| Checkpoints | 18 current `.pth` pointers plus 15 removed SAM pointers | Git LFS pointer records | generated training artifacts | Ownership, training-data terms, identity and availability must be checked separately. |

## Exact duplicate groups

The following components have identical Git blob SHAs across multiple research areas:

- `encoder.py`: `159ba34dde94cdb4cfe527c66cf7ae04abbd6147`
- `layers.py`: `812f9ff9185dd4c1380899bf8e7c20d4117ff5ca`
- `drop.py`: `c42a38778cf12bc251c2a1626c97984686e449aa`
- `positional_encoding.py`: `a43536dc1c2af9d6ebd0f6f084609ca76c4bc426`
- region/SAM `GM_Mask.py`: `42e879480a1f264e4ee6c71fdd33cb1c42a8b87f`
- region/SAM `attention_loss.py`: `4d0548c0e8f8618e1e21f0f12c38876fba34072c`
- region/SAM `metrics.py`: `1f9e3f1021e5bc3d6c840513f9a42e8938466a80`

Exact duplication supports later consolidation, but does not establish ownership or licence compatibility.

## Upstream references requiring exact-version verification

- PyTorch: https://github.com/pytorch/pytorch
- Facebook DETR: https://github.com/facebookresearch/detr
- timm: https://github.com/huggingface/pytorch-image-models
- 2D positional encoding candidate: https://github.com/wzlxjtu/PositionalEncoding2D
- Milesial/Pytorch-UNet: https://github.com/milesial/Pytorch-UNet
- FastAI: https://github.com/fastai/fastai

Exact commits and copied ranges remain to be established. This audit intentionally does not add notices to the historical source without exact attribution, because an incorrect notice could be misleading.

## Licence decision gate

Before selecting a repository licence:

- [ ] identify exact upstream commits for adapted files;
- [ ] determine whether the encoder is primarily PyTorch- or DETR-derived;
- [ ] replace or resolve the positional-encoding implementation;
- [ ] decide whether to replace the GPL-derived U-Net;
- [ ] review possible Soka University and co-author ownership;
- [ ] add a `NOTICE` or third-party attribution file where required;
- [ ] keep historical archive notices distinct from newly written clean implementations.

## Preliminary recommendation

Use the historical code as an archived research record until provenance is resolved. For maintained packages, prefer clean implementations using current PyTorch public APIs, while preserving original scientific behavior through regression tests. This reduces licensing ambiguity and dependency on copied legacy infrastructure.
