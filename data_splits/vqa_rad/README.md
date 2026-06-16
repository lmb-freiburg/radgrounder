# VQA-RAD split

The VQA-RAD train/test split used for the reported numbers, provided for exact
reproducibility. We use the **train/test split from Wu et al. (2025)** rather than the
original VQA-RAD splits, because the original splits have **overlap between train and
test images**.

- `train_fixed_split.json` (785), `validation_fixed_split.json`, `test_fixed_split.json` (735)
- Each entry: `{"question", "answer", "image_name", "q_type"}` where `q_type ∈ {open, closed}`.
- `image_name` references the official VQA-RAD images (`synpic*.jpg`) — **download the
  images** from the official VQA-RAD release and point `VQA_RAD_ROOT` at the dir
  containing `images/`. Only the split/QA JSONs are redistributed here, not the images.

**Split source:** Wu, C., Zhang, X., Zhang, Y., Hui, H., Wang, Y., Xie, W.
*Towards generalist foundation model for radiology by leveraging web-scale 2D&3D
medical data.* Nature Communications 16(1), 7866 (Aug 2025).
https://doi.org/10.1038/s41467-025-62385-7

**Underlying dataset:** VQA-RAD — Lau, J.J., Gayen, S., Ben Abacha, A.,
Demner-Fushman, D. *A dataset of clinically generated visual questions and answers
about radiology images.* Sci Data 5, 180251 (2018). https://osf.io/89kps/

Verify the licenses (VQA-RAD QA text and the Wu et al. split) permit redistribution
before publishing (the images are not included here).
