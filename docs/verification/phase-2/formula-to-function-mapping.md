# Formula-to-Function Mapping

## Status key

- **Match** — the historical function implements the paper concept with no material static discrepancy identified.
- **Partial** — the intended concept is present, but reduction, normalization, dimensions, configuration, or another important detail differs.
- **Different** — the historical function implements a materially different formulation.
- **Absent** — no corresponding implementation was found in the frozen repository.
- **Unverified** — static correspondence exists, but execution is required to confirm behavior.

---

## 1. ARViT architecture

| Paper concept | Historical location | Function or class | Status | Notes |
|---|---|---|---|---|
| Divide image into non-overlapping patches and linearly embed them | `distance_based/ARViT2D/layers/layers.py`; identical copies in region and SAM | `PatchEmbed` | Match | Uses one `Conv2d` whose kernel and stride equal patch size. |
| Fixed 2D sinusoidal positional encoding | `*/utils/positional_encoding.py` | `PositionalEncodingSin.positionalencoding2d` | Partial | Correct fixed sinusoidal structure; model multiplies it by `0.3`, which is not stated in the papers. |
| Six encoder blocks | `*/layers/encoder.py`; launch scripts | `EncoderModule`, `TransformerEncoder` | Match for selected launch configurations | Constructors have other defaults, but published scripts pass six layers. |
| Multi-head self-attention and `N × N` attention map | `*/layers/encoder.py` | `TransformerEncoderLayer.forward_post` | Partial | Returns PyTorch attention weights. Historical API defaults average attention across heads. |
| Reduced feed-forward block | `*/layers/encoder.py` | `TransformerEncoderLayer` | Different from literal paper wording | Code uses two linear projections with an activation, not one linear layer. |
| No classification token | model files | `ARViT.forward`, `ARViT2D.forward` | Match | No CLS token is constructed or appended. |
| Complete final encoder output used by head | model files | `forward` | Match | Output is normalized, flattened, and passed to one linear task head. |
| ARViT-Base without regularization | model plus cross-entropy-only use | `ARViT_CrossEntropy` or loss wrapper with no selected layer | Partial | Architecture supports it, but no clean standalone base package exists. |
| Distance-paper configuration | `distance_based/launch-ARViT2D*.py` | model construction | Match | Six layers, 12 heads, hidden dimension 516, 256 input, 16 patch. |
| Region-paper configuration | `region_similarity_based/launch-*.py` | model construction | Partial | Six layers, eight heads, hidden dimension 512; final paper training pipeline and hyperparameters are not fully represented. |

---

## 2. Distance-based regularization

| Paper equation or step | Historical location | Function or class | Status | Notes |
|---|---|---|---|---|
| Pairwise Manhattan distance matrix `D` | `distance_based/ARViT2D/utils/penalty_matrix.py` | `Penalty_Matrix.distance_matrix` | Match for the 16×16 patch grid | Uses SciPy `cityblock`; implementation assumes a square-style layout and later hard-codes 256 rows. |
| Distance-to-penalty transformation `P` with `alpha` and `beta` | same file | `Penalty_Matrix.penalty_weights`, branch `penalty_factor == "2"` | Match in algebraic intent | Uses row maxima through a symmetric-matrix shortcut; hard-coded reshape to 256. |
| Precompute penalty matrix before training | same file and distance model | `penalty_matrix`; `ARViT2D.__init__` | Match | Matrix is repeated by configured batch size and stored as a non-trainable parameter. |
| Per-image pointwise product `l_X = sum(A * P)` | `distance_based/losses/distance_loss.py` | `Distance_loss.forward` | Partial | Computes the product, but sums across the entire batch before applying the nonlinear transform. |
| Clamp negative `l_X` to zero | same file | `Distance_loss.forward` | Different | Code clamps the aggregate to one whenever it is at or below one. |
| Per-image `log(l_X* + 1)` | same file | `Distance_loss.forward` | Different | Code calculates one `log(value)` after batch aggregation and no explicit `+1`. |
| Sum Distance Loss across examples | same file | `Distance_loss.forward` | Different | Batch examples are combined before clamping/logging rather than after. |
| `L_total = L_task + lambda L_D` | same file | `ARViT2D_Loss.forward` | Match at wrapper level | The internal `L_D` differs from the paper. |
| Single-layer variants L1–L6 | distance launch and loss files | `ARViT2D_Loss(layer=...)` | Match structurally | Repository has one configurable layer index. |
| Multi-layer regularization | distance multi-layer launch/loss | `ARViT2D_MultiLayer_Loss` | Historical extension | Not part of the reported distance-paper experiment set. |
| ImageNet rotation-estimation pretraining | no matching current script | — | Absent | Current launch scripts use supervised Imagenette or historical local paths rather than a visible rotation-label pipeline. |

---

## 3. Region-similarity regularization

| Paper equation or step | Historical location | Function or class | Status | Notes |
|---|---|---|---|---|
| Divide image into regions of size `G × G` | `region_similarity_based/ARViT/utils/GM_Mask.py` | `GM_Mask.img_patches` | Partial | Non-overlapping extraction is present; uses `.data`, assumes RGB, and is tied to historical dimensions. |
| Flatten each RGB region and compute `3 × 3` Gram matrix | same file | `grid_gram_matrix` | Partial | Correct correlation structure; includes an additional division by `3G²`. |
| Pairwise MSE between Gram matrices | same file | `gram_dist_matrix` | Different | Uses `nn.PairwiseDistance(p=0.1)`, not mean squared error. |
| Normalize region-distance matrix to `[0,1]` | same file | `gram_dist_matrix` | Partial | Min-max normalization exists, but has no zero-range guard. |
| Symmetric, non-negative, hollow distance matrix | same file | `gram_dist_matrix` | Unverified/partial | Pairwise construction should be symmetric with zero diagonal in exact arithmetic, but the non-MSE metric and normalization require runtime tests. |
| Nine-step double pooling when `G>P` | `region_similarity_based/ARViT/ARViT.py` | `ARViT.rescale` | Different for `G=32` | Pooling kernel remains one and output stays 256×256; paper requires 64×64 for 32-pixel regions. |
| Normalize attention directly when `G=P` | same file | `ARViT.rescale` | Match for the 16-pixel path | Min-max normalization yields a 256×256 representation. |
| `l_AX = sum(Msa * (D + rho))` with `rho >= 0` | `region_similarity_based/losses/attention_loss.py` | `Attention_loss.penalty_factor`, `forward` | Different | Uses a default/launch bias of `-0.17` and then aggregates the complete batch. |
| `L_A = sum_i log(l_AX_i + 1)` | same file | `Attention_loss.forward` | Different | One aggregate is clamped to one and logged without explicit `+1`. |
| `L_total = L_task + lambda L_A` | same file | `ARViT_Loss`, `ARViT_MultiLayer_Loss` | Match at wrapper level | Internal attention loss and reported hyperparameters differ. |
| Published `lambda=0.005`, `rho=0.3` | no matching current configuration | — | Absent | Current principal scripts use `lambda=0.01` and bias `-0.17`. |
| Region sizes 16 and 32 | model/launch arguments and checkpoint names | `gm_patch` | Partial | Both values are represented historically, but current scaling appears incompatible with 32. |
| BYOL augmentations, multi-crop, ImageNet rotation pretraining | no complete matching current script | — | Absent or not located | Requires any later/private training code or experiment notes. |

---

## 4. Adversarial Self-Attention Misdirection

| Paper equation or step | Historical location | Function or class | Status | Notes |
|---|---|---|---|---|
| U-Net-like lens with four residual units | `SAM - Adversarial Self-Attention Misdirection/models/unet.py` | `UNet` | Different | Conventional DoubleConv/max-pool/skip U-Net; no residual units. |
| Add generated noise to input | same file | `UNet.forward` | Match | Returns `noise + x0`. |
| Similarity matrix `S` from original image | no explicit paired computation | — | Absent | ARViT computes one Gram-distance representation for its current input. |
| Similarity matrix `S*` from noised image | critic path via ARViT | `GM_Mask.forward` | Partial | A matrix can be computed for the noised image, but it is not paired with an original-image matrix in the loss. |
| `L_m = MSE(S, S*)` | `losses/sam_loss.py` | `Misdirection_loss` | Different | Historical objective inverts one distance matrix and combines it with attention. |
| Pixel-wise reconstruction `L_rec` | same file | `GeneratorLoss.forward` | Partial | Uses PyTorch mean MSE rather than summed squared L2 norm. |
| Negative task-loss component `-alpha L_F` | same file | `GeneratorLoss.forward` | Match in sign/intention | Implemented as `-beta * classificationLoss`; symbol and reduction differ. |
| Negative misdirection component `-lambda L_m` | same file | `GeneratorLoss.forward` | Different | Historical `Lm` is added, but it is also a different function from the paper's `Lm`. |
| Alternating adversarial optimization | `models/utils/fastai_gan.py`; `launch-SAM.py` | GAN learner/trainer customizations | Partial | Alternation exists, but exact schedule and behavior require runtime verification. |
| ImageNet rotation-estimation adversarial pretraining | `launch-SAM.py` | training setup | Absent as documented pipeline | Uses a local custom dataset and ten classes; no visible rotation transformation/labels. |
| Final validated downstream results | repository and paper | — | Absent | Paper explicitly states that results were expected later. |

---

## Mapping decision for Phase 3

The mapping supports history-preserving extraction, but it does not support claiming that the archived source exactly reproduces each final paper. Each extracted repository should distinguish:

1. **historical implementation**, preserved without correction;
2. **published formulation**, documented from the paper;
3. **future verified implementation**, introduced only after tests and explicit behavior decisions.