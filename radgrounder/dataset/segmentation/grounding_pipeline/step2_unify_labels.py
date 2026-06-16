"""Stage 2 — unify the CT and MR label maps into one class-id space.

TotalSegmentator uses different label ids for its ``total`` (CT) and ``total_mr``
(MR) tasks. Stage 4 needs a single id space so a structure (e.g. ``liver``) gets
the same class id regardless of modality. This script merges the two per-modality
maps (``{id: name}``) into:

    total_label_map.json          {name: unified_id}
    total_ct_id_2_frerad_id.json  {ct_id: unified_id}   (identity for CT)
    total_mr_id_2_frerad_id.json  {mr_id: unified_id}

The released ``label_map/`` already ships these for the default TotalSegmentator
class set, so you only need to re-run this if your TotalSegmentator version emits a
different set of structures.

Example::

    python -m radgrounder.dataset.segmentation.grounding_pipeline.step2_unify_labels \\
        --label-map-dir radgrounder/dataset/segmentation/label_map
"""

import argparse
import json
import os


def unify(label_map_dir):
    with open(os.path.join(label_map_dir, "ct_label_map.json")) as f:
        ct_label_map = json.load(f)  # {ct_id: name}
    with open(os.path.join(label_map_dir, "mr_label_map.json")) as f:
        mr_label_map = json.load(f)  # {mr_id: name}

    # CT ids define the base unified space; MR-only structures get new ids appended.
    name_to_id = {name: int(cid) for cid, name in ct_label_map.items()}
    mr_only = set(mr_label_map.values()) - set(ct_label_map.values())
    next_id = max(name_to_id.values())
    for name in sorted(mr_only):
        next_id += 1
        name_to_id[name] = next_id

    ct_id_2_frerad_id = {int(cid): name_to_id[name] for cid, name in ct_label_map.items()}
    mr_id_2_frerad_id = {int(mid): name_to_id[name] for mid, name in mr_label_map.items()}

    out = {
        "total_label_map.json": name_to_id,
        "total_ct_id_2_frerad_id.json": ct_id_2_frerad_id,
        "total_mr_id_2_frerad_id.json": mr_id_2_frerad_id,
    }
    for fname, data in out.items():
        with open(os.path.join(label_map_dir, fname), "w") as f:
            json.dump(data, f, indent=4)
        print(f"Wrote {fname} ({len(data)} entries)")

    print(
        f"Unified {len(ct_label_map)} CT + {len(mr_label_map)} MR labels "
        f"into {len(name_to_id)} classes ({len(mr_only)} MR-only)."
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "label_map")
    p.add_argument("--label-map-dir", default=default_dir, help="Dir with ct_label_map.json / mr_label_map.json (outputs written here).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    unify(args.label_map_dir)


if __name__ == "__main__":
    main()
