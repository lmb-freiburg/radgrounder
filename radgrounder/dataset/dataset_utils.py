from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

import re


from radgrounder.paths import REFRAD2D_SPLIT_DIR, REFRAD2D_SEGMENT_DIR

DEFAULT_SPLIT_BASE_DIR = Path(REFRAD2D_SPLIT_DIR)
DEFAULT_SEGMENT_BASE_DIR = Path(REFRAD2D_SEGMENT_DIR)

REPLACEMENTS = {
    "\u202f": " ",  # narrow no-break space → space
    "\xad": "",  # soft hyphen → remove
    "\u200b": "",  # zero-width space → remove
    "\u2009": " ",  # thin space → space
    "\x96": "-",  # en dash → hyphen
    "\x84": '"',  # low double quote → "
    "\x93": '"',  # left double quote → "
    "\x91": "'",  # left single quote → '
    "\x92": "'",  # right single quote → '
    "\xa0": " ",  # NBSP → space
    "\x1f": "",  # control → remove
    "\x85": "...",  # ellipsis → ...
    "\x7f": "",  # delete → remove
    "\x82": "'",  # low single quote → '
    "\x94": '"',  # right double quote → "
    "\x02": "",  # control → remove
    "\x0b": " ",  # vertical tab → space
    "\u200c": "",  # zero-width non-joiner → remove
    "\u2060": "",  # word joiner → remove
    "\ufeff": "",  # BOM → remove
    "\u200d": "",  # zero-width joiner → remove
    "\u2006": " ",  # narrow space → space
    "\u2003": " ",  # em space → space
    "\x00": "",  # null → remove
    "\u200e": "",  # LTR mark → remove
    "\u2028": "\n",  # line separator → newline
    "\x8f": "",  # control/mis-decoded → remove
    "\x8a": "S",  # Š → ASCII S
    "\x80": "€",  # Euro → keep as symbol
}


def clean_punctuation(text: str) -> str:
    if not hasattr(clean_punctuation, "_compiled_regexes"):
        clean_punctuation._compiled_regexes = {
            "space_before_punct": re.compile(r"\s+([.,:;!?])"),
            "punctuation_spacing": re.compile(r"([.,:;!?])(?![\s\n\r.,:;!?])"),
            "space_before_newline": re.compile(r"\s+\n"),
            "end_punct": re.compile(r"[.!?]$"),
        }
    regexes = clean_punctuation._compiled_regexes

    text = regexes["space_before_punct"].sub(r"\1", text)
    text = regexes["punctuation_spacing"].sub(r"\1 ", text)
    text = regexes["space_before_newline"].sub("\n", text)
    text = text.strip()

    if not regexes["end_punct"].search(text):
        text += "."

    for old_char, new_char in REPLACEMENTS.items():
        text = text.replace(old_char, new_char)

    return text


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file {path} not found.")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_snippet2rsopids(
    split: str,
    modality: Optional[str] = None,
    base_dir: Path = DEFAULT_SPLIT_BASE_DIR,
    report_count: bool = True,
) -> dict:
    """Load snippet-to-rsopid mapping for a split and modality.

    Args:
        split: One of "train", "val", "test", or "all".
        modality: "ct", "mr", "all", or None (defaults to CT).
        base_dir: Base directory containing filtered_*_splits folders.

    Returns:
        Mapping of snippet text to list of rsopids.
    """
    modality_key = str(modality).lower() if modality is not None else "ct"

    if modality_key == "mr":
        path = base_dir / "filtered_mr_splits" / f"snippet2rsopids_{split}_mr_only.json"
        data = load_json(path)
        if report_count:
            print(f"Loaded {len(data)} snippet2rsopids from {path}")
        return data

    if modality_key == "all":
        ct_path = base_dir / "filtered_ct_splits" / f"snippet2rsopids_{split}_ct_only.json"
        mr_path = base_dir / "filtered_mr_splits" / f"snippet2rsopids_{split}_mr_only.json"
        ct_data = load_json(ct_path)
        mr_data = load_json(mr_path)
        merged = {**ct_data, **mr_data}
        if report_count:
            print(
                "Loaded snippet2rsopids (all) \n"
                f"CT={len(ct_data)} from {ct_path} \n"
                f"MR={len(mr_data)} from {mr_path} \n"
                f"Merged={len(merged)}"
            )
        return merged

    path = base_dir / "filtered_ct_splits" / f"snippet2rsopids_{split}_ct_only.json"
    data = load_json(path)
    if report_count:
        print(f"Loaded {len(data)} snippet2rsopids from {path}")
    return data


def load_segmentation_dataset(
    modality: Optional[str] = None,
    base_dir: Path = DEFAULT_SEGMENT_BASE_DIR,
    report_count: bool = True,
) -> dict:
    modality_key = str(modality).lower() if modality is not None else "ct"

    if modality_key == "mr":
        path = base_dir / "rsopid_2_segment_v3_mr.json"
        data = load_json(path)
        if report_count:
            print(f"Loaded {len(data)} segmentation samples from {path}")
        return data

    if modality_key == "all":
        ct_path = base_dir / "rsopid_2_segment_v3_ct.json"
        mr_path = base_dir / "rsopid_2_segment_v3_mr.json"
        ct_data = load_json(ct_path)
        mr_data = load_json(mr_path)
        merged = {**ct_data, **mr_data}
        if report_count:
            print(
                "Loaded segmentation (all) \n"
                f"CT={len(ct_data)} from {ct_path} \n"
                f"MR={len(mr_data)} from {mr_path} \n"
                f"Merged={len(merged)}"
            )
        return merged

    path = base_dir / "rsopid_2_segment_v3_ct.json"
    data = load_json(path)
    if report_count:
        print(f"Loaded {len(data)} segmentation samples from {path}")
    return data


def load_sampled_dataset(
    modality: Optional[str] = None,
    base_dir: Path = DEFAULT_SEGMENT_BASE_DIR,
    report_count: bool = True,
) -> dict:
    modality_key = str(modality).lower() if modality is not None else "ct"

    if modality_key == "mr":
        path = base_dir / "refrad2d_sampled_dataset_v3_mr.json"
        data = load_json(path)
        if report_count:
            print(f"Loaded {len(data)} sampled slices from {path}")
        return data

    if modality_key == "all":
        ct_path = base_dir / "refrad2d_sampled_dataset_v3_ct.json"
        mr_path = base_dir / "refrad2d_sampled_dataset_v3_mr.json"
        ct_data = load_json(ct_path)
        mr_data = load_json(mr_path)
        merged = {**ct_data, **mr_data}
        if report_count:
            print(
                "Loaded sampled dataset (all) \n"
                f"CT={len(ct_data)} from {ct_path} \n"
                f"MR={len(mr_data)} from {mr_path} \n"
                f"Merged={len(merged)}"
            )
        return merged

    path = base_dir / "refrad2d_sampled_dataset_v3_ct.json"
    data = load_json(path)
    if report_count:
        print(f"Loaded {len(data)} sampled slices from {path}")
    return data


def tokenize_text(processor, prefixes: List[str], suffixes: List[str], output_kwargs: dict):
    IMAGE_TOKEN = "<image>"
    expanded_samples = []
    for sample in prefixes:
        expanded_sample = sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * processor.image_seq_length)
        bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
        bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
        expanded_sample = (
            expanded_sample[:bos_index] + processor.tokenizer.bos_token + expanded_sample[bos_index:]
        )
        expanded_samples.append(expanded_sample)
    input_strings = [f"{sample}\n" for sample in expanded_samples]

    if suffixes is not None:
        suffixes = [sfx + processor.tokenizer.eos_token for sfx in suffixes]
        return_token_type_ids = True
    else:
        return_token_type_ids = False

    if output_kwargs["text_kwargs"].get("max_length", None) is not None:
        output_kwargs["text_kwargs"]["max_length"] += processor.image_seq_length

    inputs = processor.tokenizer(
        input_strings,
        text_pair=suffixes,
        return_token_type_ids=return_token_type_ids,
        **output_kwargs["text_kwargs"],
    )

    if return_token_type_ids:
        labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
        inputs.update({"labels": labels})

    return inputs
