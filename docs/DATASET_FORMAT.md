# RefRad2D dataset format

The RefRad2D clinical dataset (CT/MR reports, VQA, detection boxes, segmentation
masks) is **private and not distributed** with this repository. This document
describes the exact on-disk layout the loaders expect so you can train on your
**own** data in the same format.

The public benchmarks (SLAKE, VQA-RAD) used for the released eval numbers have
their own simple format and are covered in the [README](../README.md#datasets).

All locations are configured via environment variables (see
[`radgrounder/paths.py`](../radgrounder/paths.py)); defaults live under `./data/refrad2d/`.

| Env var | Default | Holds |
|---|---|---|
| `REFRAD2D_DICOM_DIR` | `data/refrad2d/dicoms_anon` | DICOM slices |
| `REFRAD2D_VQA_PARQUET` | `data/refrad2d/refrad2d_vqa_dataset.parquet` | reports + VQA |
| `REFRAD2D_SEGMENT_DIR` | `data/refrad2d/refrad2d_segment` | bbox/mask index JSONs |
| `REFRAD2D_SPLIT_DIR` | `data/refrad2d/split_v18/generated` | train/val/test split JSONs |
| `REFRAD2D_LABEL_MAP` | `radgrounder/dataset/segmentation/label_map/merged_label_map.json` | class-name → id |

Every sample is keyed by an `rsopid` (a unique slice/image identifier — use any
stable string ID for your own data).

---

## 1. DICOM images

Path pattern (`radgrounder/grounded_gemma/utils.py`, `radgrounder/dataset/segmentation/refrad2d_detect.py`):

```
${REFRAD2D_DICOM_DIR}/{rsopid[:2]}/{rsopid}.dcm.zst
```

i.e. Zstandard-compressed DICOM files, sharded into subdirectories by the first
two characters of the `rsopid`. `read_dicom_as_numpy()` decompresses, reads the
DICOM, applies `RescaleSlope`/`RescaleIntercept`, and returns a 2-D
`float32` array `(H, W)` in Hounsfield units (CT) or raw intensities (MR).

If your images are not DICOM, the simplest path is to replace `read_dicom_as_numpy`
with a function returning the same `(H, W) float32` array.

## 2. Reports + VQA — parquet

`${REFRAD2D_VQA_PARQUET}` is a pandas parquet **indexed by `rsopid`**. Each row has
per-language nested dicts (`english`, `german`) plus `modality` (`"CT"`/`"MR"`) and
`body_part`:

```python
df.loc[rsopid] = {
  "modality": "CT",
  "body_part": "Abdomen",
  "english": {
     "dicom_vqa":  [ {"question": str, "answer": str,
                      "question_type": "open"|"closed"|"multiple",
                      "choices": [str, ...] | None}, ... ],
     "snippet_vqa":[ {... same shape ...}, ... ],
     "ReportFragestellungCleaned":   str,   # report: question/indication
     "ReportKlinischeAngabenCleaned":str,   # report: clinical info
     "CleanedSentence":              str,   # report body (target for report task)
  },
  "german": { ... identical structure ... },
}
```

- **report** `question_types` use the report text fields; **vqa** use the
  `dicom_vqa` + `snippet_vqa` lists. `question_type=="multiple"` adds `choices`.

## 3. Detection — bbox index JSON

`load_segmentation_dataset()` reads, by modality:

```
${REFRAD2D_SEGMENT_DIR}/rsopid_2_segment_v3_ct.json
${REFRAD2D_SEGMENT_DIR}/rsopid_2_segment_v3_mr.json
```

Top-level object keyed by `rsopid`:

```json
{
  "ab12cd34ef56": {
    "bboxes": [[x_min, y_min, x_max, y_max, class_id], ...],
    "keywords_eng": [["liver", "liver_class", 1], ...],
    "keywords_de":  [["Leber", "Leber_class", 1], ...],
    "segment_slice_path": "data/refrad2d/masks/ab12cd34ef56.npy"
  }
}
```

- `bboxes`: **pixel** coordinates (relative to the image `W`/`H`), last element is
  the integer `class_id`.
- `keywords_*`: `(keyword_text, class_name, class_id)` triples — the keyword is the
  span in the report text that the box grounds.

## 4. Segmentation — masks

`segment_slice_path` (from the JSON above) points to a NumPy `.npy` array of shape
`(H, W)` with **per-pixel integer class IDs** (0 = background). Masks are resized to
the model resolution (224) with nearest-neighbour interpolation at load time.

## 5. Train/val/test splits — snippet→rsopid JSON

`load_snippet2rsopids()` reads, by modality and split:

```
${REFRAD2D_SPLIT_DIR}/filtered_ct_splits/snippet2rsopids_{split}_ct_only.json
${REFRAD2D_SPLIT_DIR}/filtered_mr_splits/snippet2rsopids_{split}_mr_only.json
```

with `split ∈ {train, val, test}`. Structure maps a report snippet to the rsopids
that contain it (several rsopids per snippet is fine):

```json
{ "The liver is enlarged": ["ab12cd34ef56", "xy78zw90ab12"], ... }
```

## 6. Label map

`${REFRAD2D_LABEL_MAP}` maps class name → integer id:

```json
{ "liver": 1, "kidney_right": 2, "aorta": 4, "spleen": 5, ... }
```

A shippable copy lives at the default path; replace it with your own class set.

---

## 7. How boxes / masks are encoded in the text target

The model is trained to emit grounding inline in the report text.

**Detection** (`refrad2d_detect.py`) — coordinates are normalised to `[0,1]`,
discretised into **512 bins**, and emitted as `<locNNNN>` tokens (`0000`–`0511`);
the class id becomes a `<segNNN>` token:

```
<p bbox=<loc0100><loc0200><loc0300><loc0400> id=<seg042>>liver</p>
```

So "The liver is enlarged" → "The `<p bbox=…<loc>… id=<seg042>>`liver`</p>` is enlarged".

**Segmentation** (`refrad2d_segment.py`) — the grounded keyword is wrapped in
`<seg>…</seg>`, and each `</seg>` position is paired (in order) with one binary
mask supplied to the segmentation decoder:

```
The <seg>liver</seg> is enlarged and the <seg>spleen</seg> is infarcted.
```

These special tokens (`<p bbox=`, `</p>`, `id=`, `<seg>`, `</seg>`) are added to the
tokenizer at train time (see `train_detectgemma.py` / `train_groundedgemma.py`).

---

## Normalization

`--normalization medgemma` (used for the released models) applies, for CT, three
HU windows (lung / soft-tissue / brain) as a 3-channel image; MR is scaled
symmetrically to `[-1, 1]`. Other options: `dataset_stats`, `min_max` (see
[`radgrounder/dataset/image_preprocessing.py`](../radgrounder/dataset/image_preprocessing.py)).
