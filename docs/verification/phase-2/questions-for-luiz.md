# Questions for Luiz

These questions do not block preservation of the historical directories, but their answers will affect the paper-faithful implementations, repository descriptions, checkpoint mapping, and licensing.

---

## 1. ARViT architecture

1. **Should the distance-paper configuration be treated as the primary published ARViT definition?**
   - six encoder blocks;
   - 12 attention heads;
   - hidden dimension 516.

2. **Should the six-layer, eight-head, hidden-dimension-512 model be presented as a later ARViT variant used for region similarity and SAM?**

3. The distance paper says the two-layer MLP was replaced with a single fully connected layer, while the code uses two linear projections with an activation. Which is intended to be authoritative?

4. Was multiplying the 2D positional encoding by `0.3` an intentional experimental choice that should be preserved in the modern architecture?

5. Were the regularization losses intended to use raw multi-head-attention weights or the historical min-max-normalized matrices returned by `rescale`?

6. Is the paper's “roughly 10M parameters” description intended as rounded architecture scale, or do you recall another configuration that produced a count closer to exactly 10M?

---

## 2. Distance regularization

7. Do you have another version of the Distance Loss code that calculates the loss independently per image as described in the final paper?

8. Was the historical aggregate-batch implementation used to generate the published results, or was it an earlier/later experiment?

9. Do you have the ImageNet rotation-estimation training script, configuration, notebook, or logs used for the paper?

10. Do you recall the published value of `lambda` and the complete pretraining schedule beyond the model values already visible in the paper and scripts?

11. Were the multi-layer distance experiments completed, and should they remain as an explicitly unpublished extension?

---

## 3. Region-similarity regularization

12. Was `PairwiseDistance(p=0.1)` an early experimental distance that was later replaced with MSE for the final paper?

13. Do you have the final implementation that produced the published region-similarity results?

14. Where was the nine-step 32×32-region attention scaling implemented? It is not present in the current tracked model.

15. Do you have final training code/configuration using:
    - `lambda=0.005`;
    - `rho=0.3`;
    - batch size 80;
    - BYOL augmentations;
    - multi-crop;
    - ImageNet rotation estimation?

16. Were the current `-0.17` bias and `0.01` lambda values from an earlier version of the method?

17. Are any of the surviving or removed checkpoint names known to correspond to the final 16×16 or 32×32 paper models?

18. Should the additional Gram-matrix division by `3G²` be retained as intentional normalization, even though the displayed paper equation omits it?

---

## 4. SAM

19. Was the conventional U-Net in the repository an early prototype created before the paper adopted the four-residual-unit lens architecture?

20. Does another implementation exist for the lens network described in the paper?

21. Does another implementation exist for calculating both `S(original)` and `S(noised)` and applying `MSE(S, S*)`?

22. Was the plus sign on the historical attention-based `Lm` intentional for that prototype, or was it intended to maximize the misdirection term?

23. Was SAM ever evaluated beyond the preliminary paper?

24. Do you have any later SAM checkpoints, logs, tables, figures, or unpublished results?

25. Should the new SAM repository focus on:
    - faithfully preserving the historical prototype;
    - faithfully implementing the paper;
    - or clearly including both as separate `legacy` and `paper` implementations?

The recommended answer is to preserve both when possible, but not to conflate them.

---

## 5. Training artifacts and checkpoints

26. Are the original datasets, rotation-label pipelines, experiment logs, or configuration files stored elsewhere, such as an old workstation, Google Drive, university storage, or another private repository?

27. Do you recognize the checkpoint naming conventions well enough to map files such as `ARViT-L3`, `ARViT-L5-G32`, and `ARViT-FULL` to paper tables?

28. Were `.pkl` files exported FastAI learners while `.pth` files were state dictionaries, or were some `.pth` paths also FastAI exports?

29. Should unidentifiable checkpoints be retained privately until they can be tested, rather than published with the new repositories?

---

## 6. Ownership and release

30. Did Soka University, the Atsumi Laboratory, or any research grant impose code-ownership or release conditions?

31. Should Clifford Broni-Bediako and Masayasu Atsumi be consulted before assigning an open-source licence or publishing modernized paper implementations?

32. Are there original source notices or copied-code attributions in local files that are absent from GitHub?

---

## Recommended default decisions if records cannot be recovered

If the original final code or logs cannot be found, the project can still proceed transparently:

- preserve the GitHub code as `legacy`;
- implement final paper equations independently in new tested modules;
- describe those modules as reconstructions from the papers rather than the exact original training code;
- publish only results directly taken from the papers, clearly labelled as published rather than reproduced;
- avoid distributing unidentified checkpoints;
- keep SAM experimental and report no performance claim;
- use the distance-paper configuration as the primary published ARViT configuration and the 512/8-head version as a named later variant.