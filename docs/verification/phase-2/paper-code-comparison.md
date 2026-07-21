# Paper-to-Code Comparison

## Verification basis

Historical source:

- repository: `mormille/self-attention-regularization`
- frozen branch: `archive/phase-0-source-baseline-2026-07-20`
- commit: `9f294c2a19fb855dd6230d6fcb38607e36fcac7e`

This is a static comparison. The historical code has not yet been executed in a recovered environment.

---

## 1. ARViT architecture

### Published architecture

The distance paper defines ARViT as a reduced ViT-style architecture with:

- six transformer encoder blocks;
- 12 attention heads per block;
- hidden dimension 516;
- approximately 10 million trainable parameters;
- direct patch embedding using a convolution whose kernel and stride equal the patch size;
- fixed two-dimensional sinusoidal positional encoding;
- no classification token;
- the complete final encoder representation flattened and passed to the task head;
- a reduced feed-forward component described as a single fully connected layer.

The region-similarity paper describes the ARViT used in that work as:

- six encoder blocks;
- eight attention heads per block;
- the same general reduced-ARViT design.

The historical region and SAM launch scripts instantiate hidden dimension 512, while the distance scripts instantiate hidden dimension 516.

### Historical implementation

The implementation confirms the following:

- patch embedding is a single `Conv2d` with kernel size and stride equal to the patch size;
- fixed 2D sinusoidal positional encodings are created as non-trainable parameters;
- no classification token is used;
- the final encoder output is normalized, flattened, and passed to a linear head;
- attention maps are returned from every encoder block;
- the published distance launch configuration uses six layers, 12 heads, and hidden dimension 516;
- the region and SAM launch configurations use six layers, eight heads, and hidden dimension 512.

### Confirmed differences

1. **Constructor defaults are not paper configurations.**
   - Distance model defaults: six layers, eight heads, hidden dimension 768.
   - Region/SAM model defaults: eight layers, eight heads, hidden dimension 768.
   - Published configurations are supplied only by selected launch scripts.

2. **Feed-forward structure differs from the distance paper's wording.**
   - The code uses two linear projections, `d_model -> 256 -> d_model`, with an activation between them.
   - This is a two-linear-layer feed-forward network, not literally one fully connected layer.

3. **Position encoding is multiplied by `0.3`.**
   - The papers describe fixed 2D sinusoidal positional encoding but do not document this scaling factor.

4. **Attention weights are those returned by PyTorch `MultiheadAttention`.**
   - With the historical API defaults, weights are averaged across heads.
   - The papers describe an `N × N` attention map but do not explicitly discuss this implementation detail.

5. **The model regularizes normalized attention representations.**
   - The model min-max normalizes returned attention matrices before passing them to the historical losses.
   - The distance paper's equations use the attention map directly and do not specify this normalization.
   - The region paper does require normalization of the derived self-attention matrix.

6. **Static parameter count is below the rounded paper claim.**
   - For the six-layer, 12-head, 516-dimensional model with a four-class rotation head, the historical code contains approximately 8,930,920 trainable parameters.
   - The paper's “roughly 10M” description is reasonable rounding, especially when compared at architecture scale.
   - For the six-layer, 512-dimensional, four-class region/SAM configuration, the static count is approximately 8,812,548 trainable parameters.

### Canonical architecture decision

ARViT should be represented as a reusable base architecture independent of a particular regularization loss because:

- the papers explicitly report `ARViT-Base` without regularization;
- the historical code returns attention maps and method-specific auxiliary matrices separately;
- distance, region similarity, and SAM use the same architectural core with different configurations and objectives.

The standalone `arvit` repository should therefore provide:

- `ARViTDistancePaperConfig`: six layers, 12 heads, hidden dimension 516;
- `ARViTRegionConfig`: six layers, eight heads, hidden dimension 512;
- configurable architecture parameters;
- no embedded distance or Gram-matrix regularizer in the base model.

---

## 2. Distance-based self-attention regularization

### Published method

The paper defines:

1. an `N × N` Manhattan-distance matrix between patch coordinates;
2. a bounded penalty matrix derived from distance using `alpha` and `beta`;
3. a per-image scalar penalty from the pointwise product of attention and penalty matrices;
4. clamping each negative per-image penalty to zero;
5. `log(clamped_penalty + 1)` per image;
6. summation across the training examples;
7. total objective `task_loss + lambda * distance_loss`;
8. single-layer variants `ARViT-L1` through `ARViT-L6`;
9. ImageNet self-supervised rotation pretraining with 256×256 inputs and 16×16 patches.

### Historical implementation that agrees

- Patch coordinates are generated on a 2D grid.
- Manhattan distance is calculated with SciPy `cityblock`.
- The distance matrix is symmetric and has a zero diagonal for valid square grids.
- The penalty transformation uses the intended `alpha` and `beta`-controlled bounded form.
- The penalty matrix is precomputed and stored as a non-trainable model parameter.
- Single-layer and multi-layer loss wrappers exist.
- The published model configuration appears in the launch scripts: six layers, 12 heads, hidden dimension 516, image size 256, patch size 16, `alpha=4`, `beta=0.5`.

### Historical implementation that differs

1. **Loss reduction is across the complete batch before the logarithm.**
   - Paper: compute and transform one scalar per image, then sum.
   - Code: `torch.sum(pm * sattn)` over batch and matrix dimensions, then one logarithm.

2. **The clamp and logarithm are not equivalent to the paper.**
   - Paper: `log(max(l_x, 0) + 1)`.
   - Code: replace the aggregate value with 1 when it is at or below 1, then calculate `log(value)`.
   - The code produces zero loss for positive aggregate penalties between zero and one, whereas the paper produces a positive value.
   - Batch aggregation can also allow positive and negative examples to cancel before clamping.

3. **The loss consumes min-max-normalized attention.**
   - Paper: no normalization is specified before the pointwise product.
   - Code: the attention matrix is min-max normalized by `ARViT2D.rescale`.

4. **The repository launch scripts are not the published pretraining pipeline.**
   - The principal scripts train on Imagenette as a supervised ten-class task.
   - They do not implement ImageNet rotation prediction.
   - They use ten epochs and batch size 50 rather than a documented paper reproduction configuration.

5. **Multi-layer distance regularization is historical experimental code, not a result reported in the distance paper.**

6. **Several operations are hard-coded to 256 patches.**
   - The penalty matrix reshapes a row-specific maximum to `256 × 1`.
   - Attention normalization hard-codes `R=256`.

### Phase 2 status

The historical code captures the core distance-regularization idea, but the current loss and launch scripts are not a faithful executable reproduction of the final published formulation.

---

## 3. Region-similarity regularization

### Published method

The final paper defines:

1. image patches of size `P × P` for ARViT input;
2. image regions of size `G × G`, where `G` is an integer multiple of `P`;
3. a `3 × 3` Gram matrix for each RGB region;
4. pairwise mean squared errors between Gram matrices;
5. normalization of the region-distance matrix to `[0, 1]`;
6. a symmetric, non-negative, hollow distance matrix;
7. a nine-step double-average-pooling procedure to map patch attention to region resolution when `G > P`;
8. simple attention normalization when `G = P`;
9. per-image penalty `sum(Msa * (D + rho))` with `rho >= 0`;
10. `log(penalty + 1)` per image;
11. total objective `task_loss + lambda * attention_loss`;
12. published best values `lambda=0.005` and `rho=0.3`;
13. experiments with region sizes 16 and 32, batch size 80, Adam at learning rate 0.0001, ImageNet rotation pretraining, BYOL augmentations, and multi-crop.

### Historical implementation that agrees

- Images are divided into non-overlapping regions.
- Each region is flattened per RGB channel.
- A `3 × 3` Gram matrix is calculated with batch matrix multiplication.
- The output distance representation is normalized per image to `[0, 1]` when the maximum is nonzero.
- The model returns a region-distance matrix alongside attention maps.
- For `G=P=16`, the model produces a normalized `256 × 256` attention matrix compatible with a `256 × 256` region-distance matrix.
- Six-layer, eight-head, hidden-dimension-512 launch configurations exist.

### Historical implementation that differs

1. **Pairwise region distance is not MSE.**
   - Paper: pairwise mean squared error between the nine Gram entries.
   - Code: `nn.PairwiseDistance(p=0.1)`.
   - A `p=0.1` quasi-norm is materially different from MSE and strongly changes relative distances.

2. **The Gram matrix includes an undocumented scale factor.**
   - Code divides by `3 * G * G`.
   - The paper's displayed equation does not include this factor.
   - Because distances are later normalized, a uniform positive scale factor should cancel mathematically, but it remains a paper-code difference.

3. **The 32×32-region attention scaling is not implemented correctly in the current model.**
   - Paper: average pooling with kernel and stride `G/P`, yielding `64 × 64` when `G=32` and `P=16`.
   - Code: creates `AvgPool2d(1, stride=1)` regardless of region size and hard-codes `R=256`.
   - With `gm_patch=32`, the Gram-distance matrix is `64 × 64` while the returned attention matrix remains `256 × 256`, causing an incompatible loss shape.
   - Therefore the current historical code appears usable only for the `G=P=16` path.

4. **Distance bias has the opposite sign and range.**
   - Paper: `rho >= 0`, best reported value `0.3`.
   - Code default and launch value: `-0.17`.
   - Negative entries may be introduced into `D + bias`, contradicting the paper's non-negative penalty interpretation.

5. **Loss reduction and logarithm differ.**
   - Paper: per-image `log(l_x + 1)`, then sum.
   - Code: sum over batch and matrix dimensions, clamp the aggregate to one, then take one logarithm.

6. **Published hyperparameters and training pipeline are absent from the current scripts.**
   - Paper: `lambda=0.005`, `rho=0.3`, batch 80, ImageNet rotation, BYOL augmentation, multi-crop.
   - Principal historical script: `lambda=0.01`, bias `-0.17`, batch 50, supervised Imagenette.
   - Another script uses custom ImageNet-like data and four classes but still uses historical bias/lambda values and does not visibly implement the final augmentation pipeline.

7. **Normalization can divide by zero.**
   - A uniform image or batch item with identical Gram matrices produces maximum distance zero.
   - The code divides by that maximum without an epsilon or conditional path.

### Phase 2 status

The repository contains an earlier or partial implementation of the region-similarity concept. It should be preserved historically, but a future paper-faithful implementation must replace the pairwise metric, loss, bias, and region-attention scaling.

---

## 4. Adversarial Self-Attention Misdirection

### Published method

The two-page paper defines:

- a lens network based on a U-Net-like encoder/decoder with four residual units;
- residual addition of generated noise to the input image;
- a similarity matrix `S` for the original image;
- a similarity matrix `S*` for the noised image;
- misdirection loss `MSE(S, S*)`;
- pixel-wise L2 reconstruction loss;
- adversarial objective `Lrec - lambda * Lm - alpha * LF`;
- adversarial pretraining of SAM and ARViT on ImageNet rotation estimation;
- no final validated performance results.

### Historical implementation that agrees

- A convolutional encoder/decoder produces a three-channel noise image.
- The generated output is added residually to the original image.
- Reconstruction loss is implemented with MSE.
- The generator objective includes a negative classification/task-loss term.
- Generator and critic are trained through an alternating FastAI GAN-style procedure.
- The implementation is clearly experimental and no final result tables or tracked final SAM checkpoint are present.

### Historical implementation that differs

1. **The network is not the paper's four-residual-unit lens architecture.**
   - The code is a conventional U-Net with `DoubleConv`, max-pooling, skip connections, and upsampling.
   - No residual units are implemented.

2. **The paper's two similarity matrices are not calculated.**
   - The code computes a Gram-distance matrix only for the image forwarded through the ARViT critic.
   - It does not explicitly calculate and compare `S(original)` and `S(noised)`.

3. **Historical `Misdirection_loss` is a different objective.**
   - It inverts one distance matrix, combines it with attention maps, sums the result, clamps it, and takes a logarithm.
   - It is not `MSE(S, S*)`.

4. **The sign of the misdirection component differs.**
   - Paper: subtract `lambda * Lm` so the generator maximizes the difference between original and noised similarity matrices.
   - Code: adds its historical `Lm` term to the minimized generator loss.
   - Because the code's `Lm` is itself a different inverted-attention objective, this cannot be treated as a simple sign typo; it is a different formulation.

5. **Reconstruction reduction differs.**
   - Paper: sum of pixel-wise squared L2 norms over examples.
   - Code: PyTorch MSE mean reduction.
   - This is primarily a scaling difference but affects hyperparameter interpretation.

6. **The launch script does not reproduce the paper's stated training setup.**
   - It points to a machine-specific custom ImageNet path.
   - It does not visibly create rotation labels.
   - The model is instantiated for ten classes.

### Phase 2 status

The historical SAM directory should be presented as an earlier adversarial attention-misdirection prototype. A paper-faithful SAM implementation will require a new similarity-matrix objective and a decision on whether to reproduce the cited lens architecture or document a deliberate U-Net substitution.

---

## 5. Phase 3 readiness decision

Phase 3 can proceed under the following constraints:

- extract the three historical directories without scientific correction;
- tag them as historical research snapshots;
- include provenance notes stating that final-paper discrepancies are documented in Phase 2;
- do not call the extracted code a reproducibility package;
- construct the standalone ARViT baseline from shared architecture code and named published configurations;
- do not merge historical regularizer logic into the standalone `arvit` base package;
- defer corrected, paper-faithful implementations to the modernization phases.