# Confirmed Discrepancies and Likely Bugs

This register separates facts established by static paper-to-code comparison from issues that still require execution to confirm.

---

## 1. Confirmed paper-to-code discrepancies

### ARViT

1. **Published architecture configurations are not constructor defaults.**
   - Distance publication: six layers, 12 heads, hidden dimension 516.
   - Region publication/current historical variant: six layers, eight heads, hidden dimension 512.
   - Historical constructors default to other layer counts or hidden dimensions.

2. **Feed-forward implementation does not literally match the distance paper's “single fully connected layer” wording.**
   - Code uses two linear projections with a nonlinearity.

3. **Position encoding is scaled by `0.3` in the model.**
   - This factor is not documented in the supplied papers.

4. **The distance regularizer consumes min-max-normalized attention.**
   - The distance paper's formula does not state this preprocessing step.

### Distance regularization

5. **The Distance Loss aggregates the whole batch before clamping and taking the logarithm.**
   - The paper defines a transformed value per image and then sums over examples.

6. **The clamp/logarithm differs mathematically.**
   - Paper: `log(max(l_x, 0) + 1)`.
   - Code: set aggregate values at or below one to one, then calculate `log(value)`.

7. **The current launch scripts do not reproduce ImageNet rotation pretraining.**
   - The principal visible script trains a ten-class supervised Imagenette task.

8. **The multi-layer distance loss is an additional historical experiment, not a reported distance-paper variant.**

### Region-similarity regularization

9. **Gram-matrix distance uses `PairwiseDistance(p=0.1)`, not MSE.**

10. **The code adds an undocumented Gram scaling factor `1/(3G²)`.**
    - This may cancel after normalization, but it remains a formulation difference.

11. **The current model does not implement the paper's 32×32-region attention scaling.**
    - Its attention output remains 256×256 while the corresponding region-distance matrix is 64×64.

12. **The historical bias has the opposite sign from the final paper's constraint.**
    - Paper: `rho >= 0`, best value `0.3`.
    - Code/scripts: `-0.17`.

13. **The historical Attention Loss aggregates the batch before logging and omits the paper's explicit `+1`.**

14. **Published region hyperparameters and training pipeline are not represented by the principal current scripts.**
    - Paper: `lambda=0.005`, `rho=0.3`, batch 80, ImageNet rotation, BYOL augmentation, multi-crop.
    - Code: different lambda/bias/batch and no complete visible final pipeline.

### SAM

15. **The historical U-Net is not the paper's four-residual-unit lens architecture.**

16. **The paper's paired similarity matrices `S` and `S*` are not explicitly computed and compared.**

17. **Historical `Misdirection_loss` is not `MSE(S, S*)`.**

18. **The combined historical generator objective differs from the paper.**
    - Paper: `Lrec - lambda Lm - alpha LF`.
    - Code: `Lrec + historical_Lm - beta * classification_loss`.

19. **The SAM launch script does not visibly construct rotation labels or reproduce the stated ImageNet rotation task.**

20. **No final validated SAM results are available in the supplied paper or current tracked repository.**

---

## 2. Confirmed implementation defects or incompatibilities

### Region 32×32 path

The current region model produces incompatible tensor dimensions when `gm_patch=32` and transformer patch size is 16:

- region-distance matrix: 64×64;
- normalized attention matrix: 256×256.

The historical Attention Loss multiplies these tensors pointwise. This path cannot work as written without broadcasting failure or a different untracked implementation.

### Hard-coded token count

Several functions expose configurable image and patch sizes but hard-code 256 tokens:

- distance penalty reshaping;
- distance attention normalization;
- region attention normalization/scaling.

Configurations that do not produce exactly 256 patches are therefore unsupported despite the apparent constructor parameters.

### Zero-range normalization

Both attention and Gram-distance min-max normalization divide by `high-low` or by a maximum without an epsilon/conditional guard. Constant inputs or degenerate attention maps can produce division by zero and NaNs.

### Gradient detachment in `GM_Mask`

`GM_Mask.img_patches` accesses `batch.data` before unfolding. This bypasses autograd tracking.

- In region regularization, the region-distance matrix may intentionally be treated as a non-trainable target, although `.detach()` would be safer and explicit.
- In the SAM formulation, the similarity representation of the generated image must influence the generator. Detaching the generated image prevents a similarity-based loss from backpropagating through that path.

This makes the current `GM_Mask` unsuitable for a paper-faithful SAM misdirection loss.

### Batch-size-bound model state

Position encodings, masks, and distance penalty matrices are created for a constructor-specified batch size. The forward functions slice them for smaller batches but do not support larger batches than configured. This unnecessarily couples architecture state to training batch size.

---

## 3. Likely bugs requiring runtime confirmation

1. **In-place scalar clamp during autograd.**
   - Statements such as `att_loss[att_loss <= 1] = 1` modify a computed scalar tensor in place.
   - Depending on the historical PyTorch version and graph, this may trigger an autograd versioning error or create undesirable gradients.

2. **`Optional[Tensor]` used as a default argument value.**
   - Historical forward signatures use expressions such as `mask=Optional[Tensor]` rather than `mask: Optional[Tensor] = None`.
   - Calls normally overwrite these values internally, but the signatures are semantically incorrect.

3. **Rectangular-image grid indexing.**
   - `Penalty_Matrix.distance_matrix` increments/reset logic uses horizontal and vertical grid counts in a way that is reliable for the historical square case but may order coordinates incorrectly for rectangular grids.

4. **Region count based on width only.**
   - `GM_Mask.qt_grids = (width // patch_size) ** 2` assumes square images and ignores height.

5. **Mutable list defaults.**
   - Multi-layer and SAM losses use lists as constructor/function defaults. Mutation could leak state between instances or calls.

6. **FastAI internal monkey-patching.**
   - SAM replaces internal learner methods and uses private classes/functions. Behavior is tightly version-dependent and may not match the intended optimization schedule under another FastAI release.

7. **GAN tuple/list handling.**
   - Several paths branch only when `type(inputs) is tuple` and may behave differently when FastAI supplies lists or other sequence types.

8. **Checkpoint loading semantics.**
   - Some code calls `load_learner` for paths named `.pth`, then treats the result as a state dictionary. Whether this worked depends on how those historical artifacts were actually serialized.

9. **Attention-map meaning across PyTorch versions.**
   - `nn.MultiheadAttention` return shapes and head averaging options have evolved. Runtime recovery must verify that saved checkpoints and expected `N × N` tensors align with the historical version.

10. **Penalty matrix stored as `nn.Parameter`.**
    - It is non-trainable but should conceptually be a registered buffer. The historical approach may affect state dictionaries, device transfer, and checkpoint compatibility.

---

## 4. Classification for later work

| Finding type | Phase 3 treatment | Modernization treatment |
|---|---|---|
| Paper-code discrepancy | Preserve and document | Implement paper-faithful version behind tests |
| Confirmed shape incompatibility | Preserve in historical snapshot | Correct with general dimension logic |
| Likely runtime bug | Preserve and flag | Reproduce/test before fixing |
| Configuration mismatch | Preserve scripts as historical | Add explicit published configuration files |
| Missing training pipeline | Do not invent | Recover from other records or rebuild transparently |
| Missing SAM evidence | Label experimental | Do not claim performance without new verification |