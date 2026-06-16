"""Stage 1 — segment 3D CT/MR scans with TotalSegmentator.

Walks a directory tree of 3D NIfTI scans, runs TotalSegmentator (``total`` for CT,
``total_mr`` for MR) on each, and writes a multilabel segmentation plus a small
metadata file next to a mirror of the input layout::

    <output-dir>/<same relative path as the scan>/seg_scan_clean.nii.gz
                                                  /segment_info.json

Each scan directory must contain the scan file (``--scan-name``) and a small JSON
metadata file (``--meta-name``) with a ``"Modality"`` field ("CT" or "MR").

TotalSegmentator is an optional dependency (heavy). Install the grounding extra::

    uv pip install -e ".[grounding]"     # or: uv sync --extra grounding

Example::

    python -m radgrounder.dataset.segmentation.grounding_pipeline.step1_segment_scans \\
        --scans-dir /path/to/scans --output-dir /path/to/segmentations --workers 4
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

TEMP_PATTERNS = ["/tmp/tmp*", "/tmp/pymp*", "/tmp/nnunet*", "/tmp/repeat_srun.log"]


def cleanup_temp():
    """Remove leftover nnU-Net / multiprocessing temp files TotalSegmentator creates."""
    for pattern in TEMP_PATTERNS:
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    try:
                        os.listdir(path)  # skip if in use / inaccessible
                        shutil.rmtree(path, ignore_errors=True)
                    except (OSError, PermissionError):
                        continue
                elif os.path.isfile(path):
                    os.remove(path)
            except (OSError, PermissionError):
                continue
            except Exception as e:  # noqa: BLE001
                print(f"Failed to remove {path}: {e}")


def run_totalsegmentator(nii_path, task_name, output_dir, device, black_list_path, verbose=False):
    """Run the TotalSegmentator CLI for one scan; return True on success."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "seg_" + os.path.basename(nii_path))
    cmd = [
        "TotalSegmentator",
        "-i", nii_path,
        "-o", output_path,
        "-ta", task_name,
        "--ml",
        "--device", device,
    ]
    if verbose:
        cmd.append("--verbose")
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"Error running TotalSegmentator on {nii_path}:\n{result.stderr}")
            if black_list_path:
                with open(black_list_path, "a") as f:
                    f.write(f"{nii_path}\n")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout: killed TotalSegmentator for {nii_path}")
        if black_list_path:
            with open(black_list_path, "a") as f:
                f.write(f"{nii_path}\n")
        return False


def process_scan(paths, scans_dir, output_root, device, black_list_path, black_list):
    """Worker: segment one (scan, meta) pair, mirroring its path under output_root."""
    nii_path, meta_path = paths
    try:
        if black_list and nii_path in black_list:
            return f"Skipped (blacklisted): {nii_path}"

        # Mirror the scan's location relative to the scans root.
        rel_dir = os.path.relpath(os.path.dirname(nii_path), scans_dir)
        output_dir = os.path.join(output_root, rel_dir)
        final_output_path = os.path.join(output_dir, "seg_" + os.path.basename(nii_path))
        if os.path.exists(final_output_path):
            return f"Skipped (exists): {os.path.basename(nii_path)}"

        with open(meta_path, "r") as f:
            meta_info = json.load(f)
        modality = meta_info.get("Modality", None)
        task_name = "total_mr" if modality == "MR" else "total"

        if not run_totalsegmentator(nii_path, task_name, output_dir, device, black_list_path):
            return f"Failed: {nii_path}"

        with open(os.path.join(output_dir, "segment_info.json"), "w") as f:
            json.dump({"scan_path": nii_path, "task": task_name, "modality": modality}, f, indent=4)
        return f"Success: {nii_path}"
    except Exception as e:  # noqa: BLE001
        print(f"Error processing {nii_path}: {e}")
        return f"Failed: {nii_path} ({e})"


def collect_scans(scans_dir, scan_name, meta_name):
    """Find every (scan_file, meta_file) pair under scans_dir."""
    pairs = []
    for root, _dirs, files in os.walk(scans_dir):
        scan_file = os.path.join(root, scan_name)
        meta_file = os.path.join(root, meta_name)
        if scan_name in files and os.path.exists(meta_file):
            pairs.append((scan_file, meta_file))
    return pairs


def load_blacklist(black_list_path):
    if not black_list_path or not os.path.exists(black_list_path):
        return []
    out = []
    with open(black_list_path, "r") as f:
        for line in f:
            p = line.strip()
            if p.endswith(".nii.gz") and os.path.exists(p):
                out.append(p)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scans-dir", required=True, help="Root dir of 3D NIfTI scans to segment.")
    p.add_argument("--output-dir", required=True, help="Where segmentations are written (mirrors --scans-dir layout).")
    p.add_argument("--scan-name", default="scan_clean.nii.gz", help="Scan file name in each scan dir.")
    p.add_argument("--meta-name", default="scan_clean.json", help="Per-scan metadata json with a 'Modality' field.")
    p.add_argument("--device", default="gpu", help="TotalSegmentator device: 'gpu' or 'cpu'.")
    p.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    p.add_argument("--batch-size", type=int, default=50, help="Scans per batch (temp cleanup happens between batches).")
    p.add_argument("--black-list", default=None, help="File of scan paths to skip / log failures (default: <output-dir>/black_list.txt).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    black_list_path = args.black_list or os.path.join(args.output_dir, "black_list.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    pairs = collect_scans(args.scans_dir, args.scan_name, args.meta_name)
    black_list = load_blacklist(black_list_path)
    print(f"Found {len(pairs)} scans to process ({len(black_list)} blacklisted).")

    worker = partial(
        process_scan,
        scans_dir=args.scans_dir,
        output_root=args.output_dir,
        device=args.device,
        black_list_path=black_list_path,
        black_list=black_list,
    )

    with Pool(processes=args.workers) as pool:
        for batch_start in range(0, len(pairs), args.batch_size):
            batch = pairs[batch_start : batch_start + args.batch_size]
            batch_idx = batch_start // args.batch_size + 1
            results = list(
                tqdm(pool.imap_unordered(worker, batch), total=len(batch), desc=f"Batch {batch_idx}")
            )
            num_failed = sum(1 for r in results if r.startswith("Failed"))
            print(f"Batch {batch_idx}: {num_failed} failures.")
            if num_failed >= max(1, len(results) // 2):
                print(f"Too many failures in batch {batch_idx}; cleaning up and exiting.")
                cleanup_temp()
                sys.exit(1)
            cleanup_temp()

    cleanup_temp()
    print(f"Done. Processed {len(pairs)} scans in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    main()
