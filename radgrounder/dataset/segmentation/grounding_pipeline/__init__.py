"""Grounding dataset generation pipeline.

Four stages turn 3D CT/MR scans + radiology reports into the per-slice grounding
annotations (bounding boxes + segmentation masks tied to report keywords) that the
``refrad2d_detect.py`` / ``refrad2d_segment.py`` loaders consume:

  step1_segment_scans      TotalSegmentator -> seg_scan_clean.nii.gz per scan
  step2_unify_labels       merge CT/MR label maps into one id space
  step3_extract_keywords   LLM (gpt-oss) -> {sentence: {keyword: category}}
  step4_build_grounding_dataset  slice + match keywords + extract bboxes -> JSON + .npy

See README.md in this folder for the full flow, the input/output schema, and how to
run it on your own dataset.
"""
