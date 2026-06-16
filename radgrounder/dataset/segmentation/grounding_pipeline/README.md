# Grounding dataset generation pipeline

This pipeline turns **3D CT/MR scans + their radiology reports** into the per-slice
**grounding annotations** that RadGrounder trains on — bounding boxes and segmentation
masks tied to the anatomical structures mentioned in each report sentence. We do **not**
publish the clinical RefRad2D dataset, but this code lets you build the same kind of
grounding data on **your own** scans and reports.

It has no manual annotation: organs are localized automatically with
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator), and report keywords are
extracted with an LLM (we used `gpt-oss-120b`). The two are intersected per slice — a
keyword is grounded only if its structure is both mentioned in the report and segmented in
that slice.

## Install

These dependencies are heavy and only needed for generation, so they live in an optional
extra (the core train/eval install does **not** pull them in):

```bash
uv pip install -e ".[grounding]"      # or: uv sync --extra grounding
```

This adds `opencv-python-headless` (bounding boxes) and `totalsegmentator` (Stage 1).
Stage 3 additionally needs an OpenAI-compatible LLM server (e.g. vLLM) running separately.

## The four stages

```
3D scans ──▶ step1_segment_scans ──▶ seg_scan_clean.nii.gz (per scan)
reports  ──▶ step3_extract_keywords ─▶ {sentence: {keyword: category}}
label maps ▶ step2_unify_labels ─────▶ unified CT+MR id space  (already shipped)
                          │
                          ▼
              step4_build_grounding_dataset
                          │
                          ▼
        rsopid_2_segment_v3_<modality>.json  +  segmented_slices_v3/*.npy
                          │
                          ▼
        refrad2d_detect.py / refrad2d_segment.py  (training loaders)
```

### Stage 1 — segment the scans (`step1_segment_scans.py`)

Runs TotalSegmentator on every scan under a directory tree (`total` for CT, `total_mr` for
MR, chosen from a per-scan metadata file) and writes a multilabel segmentation plus
`segment_info.json`, mirroring the input layout under `--output-dir`.

```bash
python -m radgrounder.dataset.segmentation.grounding_pipeline.step1_segment_scans \
    --scans-dir /path/to/scans \
    --output-dir /path/to/segmentations \
    --scan-name scan_clean.nii.gz --meta-name scan_clean.json \
    --workers 4 --device gpu
```

Each scan directory must contain the scan (`--scan-name`) and a small JSON (`--meta-name`)
with a `"Modality"` field (`"CT"` or `"MR"`).

### Stage 2 — unify the label maps (`step2_unify_labels.py`)

TotalSegmentator's CT and MR tasks use different label ids; Stage 4 needs one id space.
The merged maps for the **default** TotalSegmentator class set already ship in
[`../label_map/`](../label_map/), so you only need this if your TotalSegmentator version
emits a different structure set.

```bash
python -m radgrounder.dataset.segmentation.grounding_pipeline.step2_unify_labels \
    --label-map-dir radgrounder/dataset/segmentation/label_map
```

### Stage 3 — extract keywords with an LLM (`step3_extract_keywords.py`)

For each report sentence, the LLM returns the segmentable anatomical structures it mentions
and the TotalSegmentator category each maps to (see the shipped system prompts
[`keywords_system_prompt_eng.md`](keywords_system_prompt_eng.md) /
[`keywords_system_prompt_de.md`](keywords_system_prompt_de.md)). Start an
OpenAI-compatible server, then run once per language:

```bash
vllm serve gpt-oss-120b --async-scheduling          # separate shell/GPU

python -m radgrounder.dataset.segmentation.grounding_pipeline.step3_extract_keywords \
    --reports sentences_en.json --language english \
    --model gpt-oss-120b --output-dir keywords/english
```

`--reports` is a JSON list of sentences, or a parquet (`--text-column`). Output: a merged
`ExtractedKeywords_<language>_<n>_<timestamp>.json` mapping `{sentence: {keyword: category}}`.
Requests run concurrently and are checkpointed, so re-running resumes.

### Stage 4 — assemble the grounding dataset (`step4_build_grounding_dataset.py`)

Slices each segmentation, extracts boxes, intersects keywords with segmented structures, and
writes the loader-ready file (run once per modality):

```bash
python -m radgrounder.dataset.segmentation.grounding_pipeline.step4_build_grounding_dataset \
    --modality MR --scan-index scans_index_mr.json \
    --segmentations-dir /path/to/segmentations \
    --reports reports.parquet --keywords-eng kw_en.json --keywords-de kw_de.json \
    --output-dir $REFRAD2D_SEGMENT_DIR
```

## Input schema (for your own data)

**Scan index** (`--scan-index`, one JSON per modality) — maps each scan to its slices:

```jsonc
{
  "<scan_id>": {
    "rel_dir": "ab/<scan_id>",          // scan dir relative to --scans-dir / --segmentations-dir
                                         //   (optional; defaults to "<scan_id[:2]>/<scan_id>")
    "file": "scan_clean.nii.gz",        // optional, kept as metadata only
    "slices": {
      "<image_id>": { "slice_axis": 2, "slice_idx": 128 }
    }
  }
}
```

`slice_axis`/`slice_idx` tell Stage 4 which 2D slice of the 3D volume each `image_id`
corresponds to (axis-aware extraction is in [`scan_io.py`](scan_io.py)).

**Reports** (`--reports`) — a parquet indexed by `--id-column` (default `rsopid`) with the
two caption columns. The columns may be plain strings or nested dicts; the defaults
`english.CleanedSentence` / `german.CleanedSentence` use dotted-path access (override with
`--caption-eng-column` / `--caption-de-column`). The caption text must match the sentence
keys produced in Stage 3.

**Label maps** ([`../label_map/`](../label_map/)) — `merged_label_map.json` is the curated
`{class_name: class_id}` map Stage 4 uses for the grounding token ids;
`total_mr_id_2_frerad_id.json` remaps MR segmentation ids into that space.

## Output schema

`rsopid_2_segment_v3_<modality>.json` — one entry per slice, read by
`load_segmentation_dataset` in [`../../dataset_utils.py`](../../dataset_utils.py):

```jsonc
{
  "<image_id>": {
    "scan_rserid": "<scan_id>",
    "scan_file": "scan_clean.nii.gz",
    "slice_axis": 2, "slice_idx": 128,
    "segment_scan_path": ".../seg_scan_clean.nii.gz",
    "segment_slice_path": ".../segmented_slices_v3/ab/<image_id>.npy",
    "keywords_eng": [["liver", "liver", 5], ["right kidney", "kidney_right", 2]],
    "keywords_de":  [["Leber", "liver", 5]],
    "bboxes": [[x1, y1, x2, y2, class_id], ...]
  }
}
```

The training loaders turn each `(keyword, class_name, class_id)` + box into the grounding
tokens the model predicts:

- **Detection** ([`../refrad2d_detect.py`](../refrad2d_detect.py)) —
  `<p bbox=<loc{x1}><loc{y1}><loc{x2}><loc{y2}> id=<seg{class_id:03d}>>keyword</p>`
  (coordinates quantized to 512 bins).
- **Segmentation** ([`../refrad2d_segment.py`](../refrad2d_segment.py)) — `<seg>keyword</seg>`
  spans, with the binary mask read from the cached `.npy`.

## Notes

- Stages 1 and 3 need a GPU (TotalSegmentator) and a running LLM server respectively, so
  they are not runnable in CI; the helper modules `scan_io.py` and `bbox_utils.py` are
  unit-testable on CPU.
- The `v3` suffix in the filenames is the dataset version the released loaders expect; keep
  it unless you also update `load_segmentation_dataset`.
