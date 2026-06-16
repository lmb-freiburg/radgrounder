"""Centralized, environment-overridable path configuration for RadGrounder.

All machine-specific locations live here so the rest of the code contains no
hardcoded absolute paths. Every entry can be overridden with an environment
variable; the defaults are sensible relative locations under the repo so the
public benchmarks (SLAKE / VQA-RAD) work out of the box once downloaded into
``data/``.

See ``docs/DATASET_FORMAT.md`` for the layout the private-data variables expect.
"""

import os
from pathlib import Path


def _env_path(var: str, default) -> Path:
    val = os.environ.get(var)
    return Path(val).expanduser() if val else Path(default)


# Repo root = parent of the ``radgrounder`` package directory.
REPO_ROOT = Path(
    os.environ.get("RADGROUNDER_REPO_ROOT", Path(__file__).resolve().parents[1])
)

# ---- Output locations -------------------------------------------------------
OUTPUT_DIR = _env_path("RADGROUNDER_OUTPUT_DIR", REPO_ROOT / "results")
VALIDATION_RESULTS_DIR = _env_path(
    "RADGROUNDER_VALIDATION_RESULTS_DIR", OUTPUT_DIR / "refrad2d_validation_results"
)
EXPERIMENT_RESULTS_JSON = _env_path(
    "RADGROUNDER_RESULTS_JSON", OUTPUT_DIR / "results_summary" / "refrad2d_results.json"
)

# HuggingFace cache; ``None`` lets transformers use its own default (~/.cache).
HF_CACHE_DIR = os.environ.get("HF_HOME") or None

# ---- Open benchmark datasets (downloadable) ---------------------------------
DATA_ROOT = _env_path("RADGROUNDER_DATA_ROOT", REPO_ROOT / "data")
SLAKE_ROOT = _env_path("SLAKE_ROOT", DATA_ROOT / "Slake1.0")
# VQA_RAD_ROOT holds the images (images/). The custom split JSONs ship with the repo
# under data_splits/vqa_rad/ (VQA-RAD has no standard split); override if needed.
VQA_RAD_ROOT = _env_path("VQA_RAD_ROOT", DATA_ROOT / "vqa-rad" / "fixed_split")
VQA_RAD_SPLIT_DIR = _env_path("VQA_RAD_SPLIT_DIR", REPO_ROOT / "data_splits" / "vqa_rad")

# ---- Private RefRad2D dataset (not shipped; see docs/DATASET_FORMAT.md) -------
REFRAD2D_DICOM_DIR = _env_path("REFRAD2D_DICOM_DIR", DATA_ROOT / "refrad2d" / "dicoms_anon")
REFRAD2D_VQA_PARQUET = _env_path(
    "REFRAD2D_VQA_PARQUET", DATA_ROOT / "refrad2d" / "refrad2d_vqa_dataset.parquet"
)
REFRAD2D_SPLIT_DIR = _env_path(
    "REFRAD2D_SPLIT_DIR", DATA_ROOT / "refrad2d" / "split_v18" / "generated"
)
REFRAD2D_SEGMENT_DIR = _env_path(
    "REFRAD2D_SEGMENT_DIR", DATA_ROOT / "refrad2d" / "refrad2d_segment"
)
REFRAD2D_LABEL_MAP = _env_path(
    "REFRAD2D_LABEL_MAP",
    REPO_ROOT / "radgrounder" / "dataset" / "segmentation" / "label_map" / "merged_label_map.json",
)

# ---- Vision encoder (SigLIP) checkpoint used for training -------------------
# The released fine-tuned SigLIP is staged under models/siglip/ (published with the
# checkpoints; git-ignored). Override with SIGLIP_CKPT_PATH to train with your own.
SIGLIP_CKPT_PATH = str(
    _env_path("SIGLIP_CKPT_PATH", REPO_ROOT / "models" / "siglip" / "siglip_refrad2d_v18.ckpt")
)

# ---- LLM-as-judge model (served via vLLM); HF id or local path --------------
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "google/gemma-3-27b-it")
