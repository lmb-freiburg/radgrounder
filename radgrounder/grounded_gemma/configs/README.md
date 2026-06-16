# Training configs

JSON configs passed to the trainers via `--config`. Each is a flat dict of hyperparameters
and dataset switches (see [docs/TRAINING.md](../../../docs/TRAINING.md) for the full key
reference). The two `radgrounder_*` configs are the exact recipes behind the released
checkpoints.

| Config | Trains | Recipe | Launched by |
|---|---|---|---|
| `radgrounder_detection.json` | **RadGrounder (detection)** — token-based bounding boxes | RefRad2D detect+VQA+SLAKE+VQA-RAD mix, frozen fine-tuned SigLIP, 6 epochs, batch 24×12, no augment | `run_train_detect.sh` |
| `radgrounder_segmentation.json` | **RadGrounder (segmentation)** — mask decoder + `<seg>` spans | same data mix + segmentation head (`train_seg_head: true`), frozen FT SigLIP, 6 epochs, augment on | `run_train_segment.sh` |
| `train_public_vqa.json` | Public-only baseline (no RefRad2D) | external SLAKE + VQA-RAD only, **stock** SigLIP (`load_siglip_weights: false`), 1 epoch, batch 8×4 | `run_train_public_external.sh` |

Key switches:
- `dataset_name` / `selected_dataset` — route to the loaders in
  [`../../dataset/dataset_manager.py`](../../dataset/dataset_manager.py).
- `add_other_vqa_datasets` — mix SLAKE + VQA-RAD into the RefRad2D training set.
- `load_siglip_weights` — load the fine-tuned SigLIP into the (frozen) vision tower; `false`
  uses PaliGemma-2's stock encoder so no checkpoint is needed.
- `train_seg_head` — segmentation only; trains the auxiliary mask decoder.

To train on your own data, copy a config and point the `REFRAD2D_*` env vars at your dataset
(see [docs/DATASET_FORMAT.md](../../../docs/DATASET_FORMAT.md)).
