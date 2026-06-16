# Training

RadGrounder fine-tunes **PaliGemma-2 (`google/paligemma2-3b-pt-224`)** with an
optional frozen fine-tuned SigLIP vision encoder. There are two entry points:

| Task | Script | Config (released) |
|---|---|---|
| Detection (boxes inline in the report) | `radgrounder/grounded_gemma/train_detectgemma.py` | `configs/radgrounder_detection.json` |
| Segmentation (mask decoder + `<seg>` spans) | `radgrounder/grounded_gemma/train_groundedgemma.py` | `configs/radgrounder_segmentation.json` |

Launch with the wrappers (they activate `.venv`, then call the trainer):

```bash
bash radgrounder/grounded_gemma/run_train_detect.sh     # detection
bash radgrounder/grounded_gemma/run_train_segment.sh    # segmentation
# or directly:
python radgrounder/grounded_gemma/train_detectgemma.py --config <path/to/config.json>
```

## Prerequisites

- The uv environment (see [README](../README.md#environment-setup)).
- Your data laid out per [DATASET_FORMAT.md](DATASET_FORMAT.md) and the
  `REFRAD2D_*` env vars pointing at it.
- The fine-tuned SigLIP encoder. The released configs (`load_siglip_weights: true`,
  `siglip_model_path: ""`) default to the staged `models/siglip/siglip_refrad2d_v18.ckpt`.
  To use your own, set `SIGLIP_CKPT_PATH` (or the `siglip_model_path` config key); set
  `load_siglip_weights: false` to train with PaliGemma-2's stock vision tower instead.

## Config key reference

```jsonc
{
  "model_id": "google/paligemma2-3b-pt-224",  // base model
  "cache_dir": null,                          // HF cache (null = default)
  "output_dir": "./runs",                     // run dirs are written here
  "notes": "my_experiment",                   // free-text; part of the run name

  "dataset_name": "refrad2d_detect_merged",    // detection: *_detect_merged, segmentation: *_segment_merged
  "selected_dataset": "refrad2d_detect",       // sub-task filter
  "add_other_vqa_datasets": true,             // also mix in SLAKE + VQA-RAD
  "image_size": 224,
  "language": "all",                          // "en" | "de" | "all"
  "modality": "all",                          // "ct" | "mr" | "all"
  "body_part": "ALL",
  "question_types": "all",
  "normalization": "medgemma",                // "medgemma" | "dataset_stats" | "min_max"
  "augment": false,

  "load_siglip_weights": true,                // load a custom SigLIP into the vision tower
  "siglip_model_path": "",                    // empty -> use $SIGLIP_CKPT_PATH
  "train_encoder": false,                     // freeze vision tower
  "train_projector": true,                    // train the multimodal projector
  // segmentation only:
  "train_lang_model": true,                   // (groundedgemma) train the LM
  "train_seg_head": true,                     // (groundedgemma) train the mask decoder

  "num_epochs": 6,
  "batch_size": 24,
  "gradient_accumulation_steps": 12,          // effective batch = 24 * 12 = 288
  "eval_batch_size": 16,
  "learning_rate": 5e-5,
  "lr_scheduler_type": "cosine",
  "optim": "adafactor",
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "max_length": 200,
  "gradient_checkpointing": true,

  "eval_strategy": "steps", "eval_steps": 100, "save_steps": 100,
  "save_total_limit": 3, "val_dataset_size": 512,
  "use_wandb": true, "wandb_project": "radgrounder"
}
```

The released checkpoints used exactly the two configs above (frozen FT SigLIP,
medgemma normalization, all datasets without G-VQA; detection used `augment:false`).
Outputs are written to `runs/<run_name>/` with `final_model/` (HF format) and
intermediate `checkpoints/`.

## Quick try without the private data

You don't need the RefRad2D data to exercise the pipeline — the public benchmarks
work out of the box once downloaded (see README). The most reliable end-to-end
check is **evaluation against the released weights** on SLAKE / VQA-RAD:

```bash
bash radgrounder/grounded_gemma/run_eval_detect.sh     # uses models/detection
```

## Train on public data (no RefRad2D, no SigLIP)

A self-contained training example that needs only a GPU and the public **VQA-RAD +
SLAKE** downloads — no RefRad2D data and no SigLIP checkpoint (it uses PaliGemma-2's
stock vision tower). It trains on both external VQA datasets combined, the MICCAI
external-dataset setup. Config: [configs/train_public_vqa.json](../radgrounder/grounded_gemma/configs/train_public_vqa.json)
(`dataset_name: refrad2d_detect_merged`, `selected_dataset: external_dataset`, `load_siglip_weights: false`).

```bash
# 1. download SLAKE 1.0 and VQA-RAD and point at them
export SLAKE_ROOT=/path/to/Slake1.0
export VQA_RAD_ROOT=/path/to/VQA-RAD
# 2. train (1 epoch on the combined VQA-RAD + SLAKE English VQA train splits)
bash radgrounder/grounded_gemma/run_train_public_external.sh
#    or directly:
python radgrounder/grounded_gemma/train_detectgemma.py \
    --config radgrounder/grounded_gemma/configs/train_public_vqa.json
```

Outputs land in `runs/`. For an even quicker check add `"max_steps": 3` to the config.
To train on a single dataset instead, set `"dataset_name": "slake_vqa"` (or `"vqa_rad"`)
and `"selected_dataset": null`; to use the released fine-tuned SigLIP encoder, set
`"load_siglip_weights": true` (defaults to the staged `models/siglip/...`). Grounding
(detection/segmentation) training needs data in
the [RefRad2D format](DATASET_FORMAT.md).
