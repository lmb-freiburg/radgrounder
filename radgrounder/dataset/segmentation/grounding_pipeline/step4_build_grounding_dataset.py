"""Stage 4 — assemble the per-slice grounding dataset.

Combines the Stage-1 segmentations, the Stage-3 keywords, the unified label maps, and
the report captions into ``rsopid_2_segment_v3_<modality>.json`` (+ cached ``.npy``
mask slices) — exactly the file the ``refrad2d_detect.py`` / ``refrad2d_segment.py``
loaders read via ``load_segmentation_dataset``.

For each image (slice) in the scan index it:
  1. loads the 3D segmentation, extracts the matching 2D slice (scan_io),
  2. remaps MR label ids to the unified id space (for ``--modality MR``),
  3. extracts per-class bounding boxes (bbox_utils),
  4. keeps the report keywords whose category is both segmented in the slice and
     present in the caption (substring-deduplicated),
  5. caches the 2D mask as ``.npy`` and records one entry.

Run once per modality (the scan index is per-modality)::

    python -m radgrounder.dataset.segmentation.grounding_pipeline.step4_build_grounding_dataset \\
        --modality MR --scan-index scans_index_mr.json \\
        --segmentations-dir /path/to/segmentations \\
        --reports reports.parquet --keywords-eng kw_en.json --keywords-de kw_de.json

See README.md for the input/output schema.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from radgrounder import paths
from radgrounder.dataset.segmentation.grounding_pipeline.bbox_utils import extract_all_bboxes
from radgrounder.dataset.segmentation.grounding_pipeline.scan_io import (
    get_slice_from_scan,
    read_scan_from_file,
)

LABEL_MAP_DIR = Path(__file__).resolve().parent.parent / "label_map"


def get_matching_keywords(caption, keywords, segmented_class_names, name_to_id):
    """Keywords whose category is segmented in the slice AND appears in the caption.

    Substring duplicates are dropped (shorter keyword kept first). Returns a list of
    ``(keyword, class_name, class_id)`` tuples.
    """
    matching = []
    for keyword, class_name in keywords.get(caption, {}).items():
        if class_name in segmented_class_names and keyword in caption:
            matching.append((keyword, class_name, name_to_id.get(class_name, -1)))
    matching.sort(key=lambda x: len(x[0]))

    filtered = []
    for keyword, class_name, class_id in matching:
        if not any(keyword in kw[0] for kw in filtered):
            filtered.append((keyword, class_name, class_id))
    return filtered


def get_field(row, spec):
    """Read ``row[spec]`` with dotted-path support (e.g. ``english.CleanedSentence``)."""
    value = row
    for part in spec.split("."):
        value = value[part]
    return value


def build(args):
    with open(args.scan_index) as f:
        scan_index = json.load(f)
    with open(args.keywords_eng, encoding="utf-8") as f:
        keywords_eng = json.load(f)
    with open(args.keywords_de, encoding="utf-8") as f:
        keywords_de = json.load(f)

    with open(os.path.join(args.label_map_dir, "merged_label_map.json")) as f:
        name_to_id = json.load(f)
    id_to_name = {v: k for k, v in name_to_id.items()}

    mr_remap = None
    if args.modality.upper() == "MR":
        with open(os.path.join(args.label_map_dir, "total_mr_id_2_frerad_id.json")) as f:
            mr_remap = {int(k): v for k, v in json.load(f).items()}

    reports = pd.read_parquet(args.reports).set_index(args.id_column)

    segmentations_dir = Path(args.segmentations_dir)
    slice_save_dir = Path(args.output_dir) / "segmented_slices_v3"
    slice_save_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for scan_id, scan_info in tqdm.tqdm(scan_index.items(), desc="Processing scans"):
        rel_dir = scan_info.get("rel_dir", f"{scan_id[:2]}/{scan_id}")
        seg_file = segmentations_dir / rel_dir / "seg_scan_clean.nii.gz"
        if not seg_file.exists():
            continue
        try:
            seg_data, _, _ = read_scan_from_file(seg_file, normalize="none")
        except Exception as e:  # noqa: BLE001
            print(f"Error loading segmentation {seg_file}: {e}")
            continue

        for image_id, slice_info in scan_info["slices"].items():
            if image_id not in reports.index:
                continue
            try:
                slice_seg = get_slice_from_scan(seg_data, slice_info["slice_axis"], slice_info["slice_idx"])
            except Exception as e:  # noqa: BLE001
                print(f"Error slicing {image_id} from {scan_id}: {e}")
                continue

            if mr_remap is not None:
                slice_seg = np.vectorize(lambda x: mr_remap.get(x, 0))(slice_seg)

            labels_on_slice = np.unique(slice_seg)
            bboxes = extract_all_bboxes(slice_seg, merge_classes=True)
            segmented_class_names = [id_to_name.get(int(lbl), "background") for lbl in labels_on_slice]

            row = reports.loc[image_id]
            caption_eng = get_field(row, args.caption_eng_column)
            caption_de = get_field(row, args.caption_de_column)

            matching_eng = get_matching_keywords(caption_eng, keywords_eng, segmented_class_names, name_to_id)
            matching_de = get_matching_keywords(caption_de, keywords_de, segmented_class_names, name_to_id)

            slice_path = slice_save_dir / image_id[:2] / f"{image_id}.npy"
            if not slice_path.exists():
                slice_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(slice_path, slice_seg)

            result[image_id] = {
                "scan_rserid": scan_id,
                "scan_file": scan_info.get("file"),
                "slice_axis": slice_info["slice_axis"],
                "slice_idx": slice_info["slice_idx"],
                "segment_scan_path": str(seg_file),
                "segment_slice_path": str(slice_path),
                "keywords_eng": matching_eng,
                "keywords_de": matching_de,
                "bboxes": bboxes,
            }

    out_path = Path(args.output_dir) / f"rsopid_2_segment_v3_{args.modality.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Wrote {len(result)} slices -> {out_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", required=True, choices=["CT", "MR", "ct", "mr"], help="Modality of this scan index (selects MR id remap + output filename).")
    p.add_argument("--scan-index", required=True, help="Per-modality index json: {scan_id: {rel_dir?, file?, slices: {image_id: {slice_axis, slice_idx}}}}.")
    p.add_argument("--segmentations-dir", required=True, help="Root of Stage-1 segmentations (mirrors the scans layout).")
    p.add_argument("--reports", required=True, help="Parquet of reports indexed by image id (--id-column) with caption columns.")
    p.add_argument("--keywords-eng", required=True, help="Stage-3 English keyword JSON ({sentence: {keyword: category}}).")
    p.add_argument("--keywords-de", required=True, help="Stage-3 German keyword JSON.")
    p.add_argument("--output-dir", default=str(paths.REFRAD2D_SEGMENT_DIR), help="Where rsopid_2_segment_v3_<modality>.json + .npy slices are written.")
    p.add_argument("--label-map-dir", default=str(LABEL_MAP_DIR), help="Dir with merged_label_map.json + total_mr_id_2_frerad_id.json.")
    p.add_argument("--id-column", default="rsopid", help="Reports column / index that identifies an image slice.")
    p.add_argument("--caption-eng-column", default="english.CleanedSentence", help="Reports field for the English caption (dotted path allowed).")
    p.add_argument("--caption-de-column", default="german.CleanedSentence", help="Reports field for the German caption (dotted path allowed).")
    return p.parse_args(argv)


def main(argv=None):
    build(parse_args(argv))


if __name__ == "__main__":
    main()
