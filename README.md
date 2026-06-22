# RadGrounder

A medical vision-language model for radiology, built on **PaliGemma-2 (3B)**. It
generates radiology **reports**, answers **visual questions** (open/closed), and
produces **grounded** outputs — bounding-box **detection** and **segmentation**
masks tied to the referenced findings — for CT and MR, in English and German.

This repository contains the latest training + evaluation code and two released
checkpoints. The clinical RefRad2D dataset is **private and not distributed**; the
[dataset format](docs/DATASET_FORMAT.md) is documented so you can train on your own
data, and the public **SLAKE** / **VQA-RAD** benchmarks work out of the box.

This is the official code for *Scalable Training of Spatially Grounded 2D
Vision–Language Models for Radiology* — the model is **RadGrounder** and the dataset is
**RefRad2D** (subsets **RefRad2D-Grounded** and **RefRad2D-VQA**).

## How do I…?

| Goal | Command / entry point |
|---|---|
| Evaluate the released models | [`run_eval_detect.sh`](radgrounder/grounded_gemma/run_eval_detect.sh) / [`run_eval_segment.sh`](radgrounder/grounded_gemma/run_eval_segment.sh) |
| Train with only public data (no RefRad2D) | [`run_train_public_external.sh`](radgrounder/grounded_gemma/run_train_public_external.sh) |
| Train on my own clinical data | [docs/TRAINING.md](docs/TRAINING.md) + [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md) |
| Generate grounding annotations from scans+reports | [grounding pipeline](radgrounder/dataset/segmentation/grounding_pipeline/README.md) |
| Score reports/VQA with LLMScore (LLM judge) | [radgrounder/llm_score/](radgrounder/llm_score/README.md) |

## Released models

Both are PaliGemma-2 (3B) with a **frozen fine-tuned SigLIP** encoder, trained on
report + VQA + SLAKE + VQA-RAD plus one grounding task:

| Model | Grounding | Local path |
|---|---|---|
| **Detection** | bounding boxes | `models/detection/` |
| **Segmentation** | masks | `models/segmentation/` |
| **Fine-tuned SigLIP** | vision encoder (for training) | `models/siglip/siglip_refrad2d_v18.ckpt` |

The two checkpoints already contain their vision encoder — the standalone SigLIP is only
needed to **train** new models. Each model folder also ships its `training_config.json` (the
exact recipe; the same configs live in [`grounded_gemma/configs/`](radgrounder/grounded_gemma/configs/)).

### Download the weights

Weights live on the Hugging Face Hub at **[`lmb-freiburg/radgrounder`](https://huggingface.co/lmb-freiburg/radgrounder)**
(private until the paper is accepted — `hf auth login` with an account that has access).
Download everything into `models/` so the eval/train scripts find it with no extra config:

```bash
pip install -U "huggingface_hub[cli]"          # provides the `hf` CLI
hf download lmb-freiburg/radgrounder --local-dir models
# -> models/{detection,segmentation}/  + models/siglip/siglip_refrad2d_v18.ckpt
```

To grab just one checkpoint, add e.g. `--include "detection/*"`. The **detection** model is a
stock PaliGemma‑2 and loads with vanilla `transformers`; the **segmentation** model uses the
custom `GroundedGemmaForConditionalGeneration` class, so load it with this repo's code (run
from `radgrounder/grounded_gemma/`, as `run_eval_segment.sh` does):

```python
# detection — stock transformers
from transformers import PaliGemmaForConditionalGeneration
m = PaliGemmaForConditionalGeneration.from_pretrained("models/detection")

# segmentation — needs the radgrounder code on the path
from modeling_groundedgemma import GroundedGemmaForConditionalGeneration
m = GroundedGemmaForConditionalGeneration.from_pretrained("models/segmentation")
```

### Results on the open benchmarks

Test-set means (the paper reports 95% bootstrap CIs over the same samples). Regenerate the
per-sample CSVs with `run_eval_*.sh` (see below) and compare column means with
[`tools/compare_eval.py`](tools/compare_eval.py). **LLMScore** is the LLM-as-judge metric
(Gemma-3-27B; see [llm_score/](radgrounder/llm_score/README.md)):

| Model | Split | CIDEr | F1 | Acc | LLMScore |
|---|---|---|---|---|---|
| Detection | SLAKE open | 3.00 | 0.87 | 0.83 | 4.52 |
| Detection | SLAKE closed | 2.33 | 0.91 | 0.91 | 4.65 |
| Detection | VQA-RAD | 0.99 | 0.50 | 0.44 | 3.22 |
| Segmentation | SLAKE open | 3.02 | 0.86 | 0.82 | 4.49 |
| Segmentation | SLAKE closed | 2.29 | 0.90 | 0.90 | 4.59 |
| Segmentation | VQA-RAD | 1.00 | 0.50 | 0.45 | 3.30 |

## Repository layout

```
radgrounder/
  paths.py                 # central, env-overridable path config (single source of truth)
  grounded_gemma/          # training + evaluation entry points and the RadGrounder model
    train_detectgemma.py   eval_detectgemma.py     run_train_detect.sh  run_eval_detect.sh
    train_groundedgemma.py eval_groundedgemma.py   run_train_segment.sh run_eval_segment.sh
    modeling_groundedgemma.py  segmentation_heads/  g_iou/   # model code (checkpoint-bound)
    configs/               # training configs (radgrounder_detection/segmentation, train_public_vqa) + README
  dataset/                 # dataset loaders (RefRad2D, SLAKE, VQA-RAD) + preprocessing
    segmentation/          # grounding loaders, label_map/ (README), grounding_pipeline/ (data generation)
  llm_score/               # LLMScore: LLM-as-judge served via vLLM (README)
docs/                      # DATASET_FORMAT.md, TRAINING.md
data_splits/vqa_rad/       # bundled VQA-RAD train/val/test split (Wu et al.)
models/                    # staged checkpoints: detection/ segmentation/ siglip/ (git-ignored)
```

## Environment setup

The environment is managed with [uv](https://docs.astral.sh/uv/) and pinned via
`uv.lock` (Python 3.10, torch 2.7.0 + cu126; a single env covers training,
evaluation, segmentation, and the LLM judge):

```bash
# from the repo root
git submodule update --init                       # fetch the OVQA metrics library

uv sync                       # creates .venv and installs the locked dependencies
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`uv sync` installs [OVQA](https://github.com/lmb-freiburg/ovqa) (the CIDEr / F1 /
accuracy metrics, vendored as the `ovqa/` submodule and installed editable) and the
`radgrounder` package itself.

All machine-specific paths live in [`radgrounder/paths.py`](radgrounder/paths.py) and are
env-overridable. Data locations use the `REFRAD2D_*` / `SLAKE_ROOT` / `VQA_RAD_ROOT` vars
below; output locations default under `results/` and can be redirected with
`RADGROUNDER_OUTPUT_DIR` (and `RADGROUNDER_VALIDATION_RESULTS_DIR`).

## Datasets

### SLAKE / VQA-RAD (public — used for the released numbers)

**SLAKE** — download [SLAKE 1.0](https://www.med-vqa.com/slake/) (CC BY-SA 4.0); its
official `train/validate/test.json` splits are used as-is. Point `SLAKE_ROOT` at it:

```bash
export SLAKE_ROOT=/path/to/Slake1.0           # contains imgs/ and {train,validate,test}.json
```

**VQA-RAD** — the original VQA-RAD splits have train/test image overlap, so we use the
train/test split from **Wu et al. (2025)** ([Nature Communications](https://doi.org/10.1038/s41467-025-62385-7)).
That split (+ open/closed labels) **ships with this repo** under
[data_splits/vqa_rad/](data_splits/vqa_rad/) — see its README for attribution. You only
need the **images** from the official [VQA-RAD release](https://osf.io/89kps/)
(Lau et al., 2018); point `VQA_RAD_ROOT` at the dir containing `images/`:

```bash
export VQA_RAD_ROOT=/path/to/vqa-rad      # contains images/ (synpic*.jpg)
# split JSONs default to data_splits/vqa_rad/; override with VQA_RAD_SPLIT_DIR if needed
```

### RefRad2D (private)

Not distributed. To train on your own clinical data, lay it out as described in
[docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md) and set `REFRAD2D_DICOM_DIR`,
`REFRAD2D_VQA_PARQUET`, `REFRAD2D_SEGMENT_DIR`, `REFRAD2D_SPLIT_DIR`.

## Evaluate the released models

The LLM metric (`--eval_llm_score`) needs an LLM-as-judge served via vLLM. vLLM cannot
share the main environment (it conflicts with transformers ≥4.54, which the segmentation
model needs), so the judge runs in a **separate** env and the eval reaches it over HTTP:

```bash
# one-time: create the judge env
uv venv --python 3.10 .venv-judge
uv pip install --python .venv-judge -r requirements-judge.txt

# start the judge (separate shell, on a GPU)
export LLM_JUDGE_MODEL=google/gemma-3-27b-it   # or a local path
bash radgrounder/llm_score/start_gemma3_server.sh
```

The eval (run from the main `.venv`) auto-detects the running server. If you don't pass
`--eval_llm_score`, no judge is needed.

Then run evaluation on SLAKE + VQA-RAD (uses the staged weights by default):

```bash
bash radgrounder/grounded_gemma/run_eval_detect.sh     # detection model
bash radgrounder/grounded_gemma/run_eval_segment.sh    # segmentation model
# override the checkpoint with:  MODEL_PATH=/path/to/run bash run_eval_detect.sh
```

Per-sample CSVs are written under `results/refrad2d_validation_results/`; column
means reproduce the table above. On SLURM, submit with `sbatch --gres=gpu:1 …`.

## Train

See [docs/TRAINING.md](docs/TRAINING.md).

**No RefRad2D data?** Train end-to-end on the public external VQA datasets (VQA-RAD +
SLAKE combined, the MICCAI external-dataset setup) with only a GPU + the two downloads
(no SigLIP needed — stock vision tower):

```bash
export SLAKE_ROOT=/path/to/Slake1.0
export VQA_RAD_ROOT=/path/to/VQA-RAD
bash radgrounder/grounded_gemma/run_train_public_external.sh
```

**Full RefRad2D training** (grounding tasks need data in the [RefRad2D format](docs/DATASET_FORMAT.md)):

```bash
bash radgrounder/grounded_gemma/run_train_detect.sh      # detection
bash radgrounder/grounded_gemma/run_train_segment.sh     # segmentation
```

## License

This repository is released under multiple licenses depending on the component:

- **Original RadGrounder code — MIT** (see [`LICENSE`](LICENSE)). All source
  code in this repo is MIT unless a file header says otherwise.
- **Gemma / PaliGemma-derived code — Apache 2.0** (see
  [`LICENSE_APACHE`](LICENSE_APACHE)). Files derived from Google's Gemma /
  PaliGemma reference code or from Hugging Face Transformers carry an Apache 2.0
  header — notably
  [`radgrounder/grounded_gemma/modeling_groundedgemma.py`](radgrounder/grounded_gemma/modeling_groundedgemma.py).
  The vendored [OVQA](https://github.com/lmb-freiburg/ovqa) metrics submodule is
  also Apache 2.0 and ships its own [`ovqa/LICENSE`](ovqa/LICENSE).
- **Released model weights — CC BY-NC-SA 4.0 + Gemma Terms of Use** (see
  [`LICENSE_MODELS`](LICENSE_MODELS)). The checkpoints are
  Attribution–NonCommercial–ShareAlike, and because they are derivatives of
  PaliGemma-2 / Gemma they additionally remain subject to the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and
  [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

The weights are also subject to the licenses of the data used for training. The
model is a research artifact and is **not** a medical device. Anyone intending
to use RadGrounder in a commercial or clinical setting should independently
verify all dataset, model, and regulatory requirements and obtain any required
permissions.
