# Phase 1 Unresolved Questions

## Priority A — canonical architecture or licensing blockers

1. **Which ARViT configuration is canonical?**  
   Distance launch scripts use 6 layers, 12 heads, and hidden dimension 516. Region and SAM launch scripts use 6 layers, 8 heads, and hidden dimension 512, while constructor defaults differ again.

2. **Is ARViT the unregularized base architecture, or does the name inherently include attention regulation?**  
   This determines the correct scope of the standalone `arvit` repository.

3. **What exact upstream revision produced `encoder.py`?**  
   Its header says it was copied from `torch.nn.Transformer`, while the modifications and wording resemble DETR. Exact attribution is required.

4. **What exact `timm` revision produced `layers.py` and `drop.py`?**  
   Copyright appears in comments, but repository-level notices are absent.

5. **What is the licence status of the copied two-dimensional positional encoding?**  
   The likely upstream repository does not expose a clear licence. Should it be replaced with an independently implemented equivalent?

6. **How should the SAM U-Net be handled?**  
   It closely follows a GPL-3.0 implementation. Options include replacing it with a clean implementation of the paper’s lens architecture, using GPL-compatible licensing, or obtaining legal guidance.

7. **Do Soka University, Clifford Broni-Bediako, Masayasu Atsumi, or other collaborators hold rights that affect publication or relicensing?**

## Priority B — scientific-verification blockers

8. Does the historical Distance Loss implement the final paper’s clamp-and-log formula exactly?

9. Does `GM_Mask` implement the final paper’s Gram-matrix mean squared error, or an earlier `PairwiseDistance(p=0.1)` experiment?

10. Why does the current historical region loss use a negative bias (`-0.17`) while an older file uses a positive bias and the final paper defines a non-negative distance bias?

11. Is the attention-rescaling code in the region implementation equivalent to the nine-step pooling procedure in the final paper?

12. Why does the historical SAM generator objective add its `Lm` term while the paper writes the misdirection term with a negative sign?

13. Does `models/SAM.py` intentionally wrap only the critic, or is a generator member missing from the committed version?

14. Were SAM results completed beyond the preliminary two-page paper? Where are any logs, tables, notebooks, or checkpoints?

## Priority C — reproducibility blockers

15. What exact FastAI version was used?

16. What exact PyTorch, torchvision, CUDA, cuDNN, and NCCL versions were used?

17. Were experiments run from each method directory, the repository root, or with a modified `PYTHONPATH`?

18. Are the `.pth` files state dictionaries, exported FastAI learners, or differently serialized objects with misleading extensions?

19. Which historical Git LFS objects are still downloadable?

20. Which checkpoint maps to each published table row and random seed?

21. Where is `data/flowers.csv`, referenced by the fine-tuning script?

22. What was the exact structure of `~/Luiz/gan_attention/data/Custom_ImageNet`?

23. Were uncommitted augmentation or self-supervised rotation scripts used for published ImageNet pretraining? The visible simple launch scripts do not by themselves demonstrate the complete published protocol.

## Priority D — migration and maintenance decisions

24. Should history-preserving extraction include checkpoint-pointer commits, or should source and artifact history be split?

25. Should archive code remain at the repository root or under `legacy/` after extraction?

26. Should method repositories pin `arvit` by release, Git commit, or workspace during early modernization?

27. Should all maintained repositories share one licence after provenance is resolved, or use different licences based on inherited components?

28. Should visualizations be recreated instead of carrying hard-coded notebook-era utilities?

29. Should the historical umbrella repository be archived after migration, or remain writable for index and documentation updates?

## Information requested from Luiz

The following materials would significantly reduce uncertainty:

- old `requirements.txt`, Conda environment, `pip freeze`, or workstation notes;
- thesis appendices or implementation chapter;
- experiment spreadsheets and training logs;
- original supplementary files from paper submissions;
- surviving model checkpoints outside GitHub;
- notes identifying third-party repositories used during implementation;
- later SAM results, if any;
- confirmation of code ownership and collaborator expectations.

## Phase boundary

These questions are documented rather than answered by assumption. Formula questions belong to Phase 2, environment execution to Phase 4, and licence resolution must precede public licensing in Phase 9.
