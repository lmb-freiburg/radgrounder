# Label maps

TotalSegmentator emits **different** label-id sets for its CT (`total`) and MR (`total_mr`)
tasks. RadGrounder harmonizes them into a single schema of **C = 121** anatomical classes
(shared anatomies merged), used both for the grounding token ids and for keyword→structure
matching. These JSONs encode that schema and the mappings between the three id spaces.

| File | Maps | Used by |
|---|---|---|
| `merged_label_map.json` | `{class_name: unified_id}` (the curated C=121 schema) | training/eval grounding tokens; grounding pipeline step 4 |
| `ct_label_map.json` | `{ct_id: class_name}` (TotalSegmentator CT) | label unification (step 2) |
| `mr_label_map.json` | `{mr_id: class_name}` (TotalSegmentator MR) | label unification (step 2) |
| `total_ct_id_2_frerad_id.json` | `{ct_id: unified_id}` | (CT is identity in the unified space) |
| `total_mr_id_2_frerad_id.json` | `{mr_id: unified_id}` | remap MR masks into the unified space (step 4) |
| `class_2_organ_name_english.json` | `{unified_id: english name}` | display / VQA templates |
| `class_2_organ_name_german.json` | `{unified_id: german name}` | display / VQA templates |

The `merged_label_map.json` ships ready to use; regenerate the `total_*` maps only if your
TotalSegmentator version changes the class set — see the grounding pipeline
[step 2](../grounding_pipeline/step2_unify_labels.py). (The `frerad_id` token in two
filenames is the internal name for the unified id and is kept for backward compatibility.)
