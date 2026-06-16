import copy
import json
import math
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tqdm

from radgrounder.grounded_gemma.augmentations import build_detect_aug_pipeline, build_augmentation_pipeline
from radgrounder.grounded_gemma.utils import read_dicom_as_numpy
from radgrounder.dataset.dataset_manager import USE_GROUNDED_REPORT_PROMPT
from radgrounder.dataset.image_preprocessing import (
    DEFAULT_NORMALIZATION,
    NormalizationConfig,
    NormalizationType,
    apply_normalization,
)
from radgrounder.dataset.dataset_utils import (
    clean_punctuation,
    load_segmentation_dataset,
    load_snippet2rsopids,
    tokenize_text,
)


GRAYSCALE_PROB = 0.2

class RefRad2DDetect(Dataset):
    def __init__(
        self,
        split="train",
        img_size=224,
        eval_mode=False,
        language="all",
        augment=False,
        dataset_size=None,
        max_length=200,
        body_part=None,
        modality=None,
        prefix_style=None,
        only_segmented=False,
        tokenizer=None,
        question_types="all",
        normalization: Optional[NormalizationConfig] = None,
    ):
        print("Initializing RefRad2DDetect dataset")
        self.split = split
        self.eval_mode = eval_mode
        self.normalization = normalization or DEFAULT_NORMALIZATION
        print("Normalization strategy:", self.normalization.strategy.value)

        if tokenizer is None:
            raise ValueError("Tokenizer must be provided in RefRad2DDetect")
        self.tokenizer = tokenizer

        self._setup_random_generators()

        snippet2rsopids = load_snippet2rsopids(split, modality)
        print(f"Loaded snippet2rsopids with {len(snippet2rsopids)} caption for split '{split}' and modality '{modality}'")

        from radgrounder.paths import REFRAD2D_VQA_PARQUET
        merged_vqa_dataset_path = str(REFRAD2D_VQA_PARQUET)
        refrad2d_dataset = pd.read_parquet(merged_vqa_dataset_path)
        refrad2d_dataset = refrad2d_dataset.set_index('rsopid')

        all_rsopids = {rsopid for rsopids in snippet2rsopids.values() for rsopid in rsopids}
        refrad2d_dataset = refrad2d_dataset[refrad2d_dataset.index.isin(all_rsopids)]
        rsopid_to_data = refrad2d_dataset.to_dict(orient="index")

        self.segmentation_dataset = load_segmentation_dataset(modality)
        print(f"Loaded segmentation dataset with {len(self.segmentation_dataset)} samples.")
        self.segmentation_dataset = {
            k: v for k, v in self.segmentation_dataset.items()
            if len(v["keywords_eng"]) > 0 and len(v["keywords_de"]) > 0
        }
        print(f"Filtered segmentation dataset to {len(self.segmentation_dataset)} samples with keywords.")

        filtered_snippet2rsopids = {}
        self.filtered_dataset = {}
        print(f"Body part filter: {body_part}, Modality filter: {modality}, Language filter: {language}")
        for caption, rsopids in tqdm.tqdm(snippet2rsopids.items(), desc="Filtering dataset"):
            valid_rsopids = []
            for rsopid in rsopids:
                data = rsopid_to_data.get(rsopid)
                if data is None:
                    continue
                if only_segmented and rsopid not in self.segmentation_dataset:
                    continue
                if not self._matches_filters(data, body_part, modality):
                    continue
                valid_rsopids.append(rsopid)
                self.filtered_dataset[rsopid] = data
            if valid_rsopids:
                filtered_snippet2rsopids[caption] = valid_rsopids

        if not filtered_snippet2rsopids:
            raise ValueError("No samples left after applying the provided filters.")

        if dataset_size is not None:
            if dataset_size > len(filtered_snippet2rsopids):
                print(
                    f"Dataset size {dataset_size} is larger than the actual dataset size {len(filtered_snippet2rsopids)}. "
                    "Using the actual dataset size."
                )
                dataset_size = len(filtered_snippet2rsopids)
            rng = np.random.default_rng(42)
            selected_keys = rng.choice(list(filtered_snippet2rsopids.keys()), size=dataset_size, replace=False)
            filtered_snippet2rsopids = {k: filtered_snippet2rsopids[k] for k in selected_keys}
            allowed_rsopids = {
                rsopid
                for values in filtered_snippet2rsopids.values()
                for rsopid in values
            }
            self.filtered_dataset = {rsopid: self.filtered_dataset[rsopid] for rsopid in allowed_rsopids}

        self.snippet2rsopids = filtered_snippet2rsopids
        self.keys = list(self.snippet2rsopids.keys())
        print(f"Filtered dataset size after grouping: {len(self.snippet2rsopids)} captions")

        from radgrounder.paths import REFRAD2D_DICOM_DIR
        self.slices_path = str(REFRAD2D_DICOM_DIR)
        self.image_shape = (img_size, img_size)
        self.dataset_stats = {
            "avr_ct_mean": -610.1908535827807,
            "avr_ct_std": 737.3882466073229,
            "avr_mr_mean": 141.1154960399739,
            "avr_mr_std": 255.40429635138338,
        }

        self.num_bins = 512
        self.augment = augment
        print(f"augment: {self.augment}")
        if self.augment:
            self.default_augmentation = build_augmentation_pipeline()
            self.detect_augmentation = build_detect_aug_pipeline()

        self.torch_dtype = torch.bfloat16
        self.language = language
        if self.language not in ["english", "german", "all"]:
            raise ValueError("Invalid language. Must be one of ['english', 'german', 'all'].")
        self.max_length = max_length

        self.prefix_style = prefix_style
        self.prefix_options = ["none", "klinische_angaben", "fragestellung", "both"]

        print(f"Prefix style: {self.prefix_style}, Question types: {question_types}")
        self.question_types = ["vqa"]
        if question_types == "all":
            self.question_types = ["vqa", "report"]
        elif question_types == "report":
            self.question_types = ["report"]

    def __len__(self):
        return len(self.snippet2rsopids)
    
    def __getitem__(self, idx):
        caption = self.keys[idx]
        candidates = self.snippet2rsopids[caption]
        rsopid = candidates[0] if self.eval_mode else self._sample(candidates)

        data = self.filtered_dataset[rsopid]

        # Load image
        image_path = os.path.join(self.slices_path, rsopid[:2], rsopid + ".dcm.zst")
        modality = data["modality"]


        if rsopid in self.segmentation_dataset:
            bboxes, labels = self.load_bbox_and_labels(rsopid)
            #the augmentation can crop the image thats why first load the image and the augmented bboxes
            image, img_size, bboxes, labels = self.load_image(image_path, modality, bboxes, labels)
            if len(bboxes) > 0:
                prefix, prompt, suffix, language, question_type = self.load_detection(data, rsopid, img_size, bboxes, labels)
            else:
                prefix, prompt, suffix, language, question_type = self.load_text(data)
        else:
            image, img_size, bboxes, labels = self.load_image(image_path, modality)
            prefix, prompt, suffix, language, question_type =  self.load_text(data)

        
        image_tensor = image.type(self.torch_dtype)
        
        prefix_full = f"<image>\n{prefix}\n{prompt} "
        info = {}
        if self.eval_mode:
            info = {
                "dicom_path": image_path,
                "modality": modality,
                "language": language,
                "question_type": question_type,
                "bboxes": bboxes,
                "labels": labels
            }
        
        return image_tensor, prefix_full, suffix, info, self.eval_mode

    def load_bbox_and_labels(self, rsopid):
        sample = self.segmentation_dataset[rsopid]
        bboxes_w_labels = sample["bboxes"]
        labels = [bbox[4] for bbox in bboxes_w_labels]
        labels = np.array(labels, dtype=np.int64)
        bboxes = [[bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in bboxes_w_labels]
        bboxes = np.array(bboxes, dtype=np.float32)
        return bboxes, labels

    def load_image(self, dicom_path, modality, bboxes=None, labels=None):
        try:
            dicom_array = read_dicom_as_numpy(dicom_path)
        except Exception as e:
            print(f"Error loading image {dicom_path}: {e}")
            return None

        dicom_array = torch.from_numpy(dicom_array)
        if modality == "CT":
            dicom_array = torch.clip(dicom_array, min=-1000)
        dicom_array = dicom_array.unsqueeze(0)  # Add channel dimension
        
        # print(f"Loaded image {dicom_path} with shape {dicom_array.shape}")
        if self.augment:
            if bboxes is not None and labels is not None:
                # print(dicom_array.shape, bboxes, labels)
                data = {"image": dicom_array, "boxes": bboxes, "labels": labels}
                aug_data = self.detect_augmentation(data)
                dicom_array = aug_data["image"]
                bboxes = aug_data["boxes"]
                labels = aug_data["labels"]
            else:
                dicom_array = self.default_augmentation(dicom_array)


        image = self.normalize_image(dicom_array, modality, eval_mode=self.eval_mode)
        org_img_size = image.shape[-2:]
        
        #resize to 224 x 224
        image = F.interpolate(image.unsqueeze(0), size=self.image_shape, mode='bilinear', align_corners=False)
        image = image[0]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if bboxes is not None and bboxes.shape[0] > 0:
            # print(image.shape[2], org_img_size[1])
            scale_x = image.shape[2] / org_img_size[1]
            scale_y = image.shape[1] / org_img_size[0]
            bboxes *= np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
            bboxes = bboxes.astype(int)

        return image, org_img_size, bboxes, labels

    def normalize_image(self, image, modality=None, eval_mode=False):
        norm_img = apply_normalization(
            image,
            modality,
            self.normalization,
            dataset_stats=self.dataset_stats,
        )

        if self.normalization.strategy == NormalizationType.MEDGEMMA and not eval_mode and self.augment:
            if self._random_float() < GRAYSCALE_PROB:
                channel_idx = self.random_choice_function(np.arange(norm_img.shape[0]))
                norm_img = norm_img[channel_idx, :, :].unsqueeze(0).repeat(3, 1, 1)

        return norm_img
    
    def load_text(self, data):
        language = self.language
        if language == "all":
            language = self._sample(["english", "german"])
        
        quest_type = self._sample(self.question_types)
            
        if quest_type == "report":
            prefix, prompt, suffix, question_type = self.load_report(data, language)
        elif quest_type == "vqa":
            prefix, prompt, suffix, question_type = self.load_vqa(data, language)

        # adaptively cut prefix and suffix to fit within max_length
        prefix, suffix = self.adaptively_cut_text(prompt, prefix, suffix)

        return prefix, prompt, suffix, language, question_type

    def load_detection(self, data, rsopid, img_size, bboxes, labels):
        sample = self.segmentation_dataset[rsopid]
        language = self.language
        if language == "all":
            language = self._sample(["english", "german"])

        question_type = "segmentation"
        if language == "english":
            keywords = sample["keywords_eng"]
            prompt = "Caption this image and detect keywords:" if len(keywords) > 0 else "Caption this image:"
        elif language == "german":
            keywords = sample["keywords_de"]
            prompt = "Beschreibe dieses Bild und detektiere Schlüsselwörter:" if len(keywords) > 0 else "Beschreibe dieses Bild:"

        prefix = self.load_prefix(data, language)
        suffix = self.get_report_content(data[language], "CleanedSentence")

        suffix_w_detection_tokens = self.add_detection_tokens(suffix, keywords, bboxes, labels)
        # filtered_seg_map = self.load_segm_map(sample, segmented_ids)

        return prefix, prompt, suffix_w_detection_tokens, language, question_type

    def add_detection_tokens(self, suffix, keywords, bboxes, labels):
        # filtered_keywords = []
        # segmented_ids = set()
        # Filter keywords to only include those dont contain others
        # for keyword, class_name, class_id in keywords:
        #     if keyword not in suffix:
        #         continue
        #     if not any(keyword in kw[0] for kw in filtered_keywords):
        #         filtered_keywords.append((keyword, class_name, class_id))
        #         segmented_ids.add(class_id)

        #Find bboxes for the filtered keywords
        class_id_to_bbox = {}
        #bbox (x, y, x + w, y + h, class_id)
        for keyword, class_name, class_id in keywords:
            #dont override the existing bbox
            if class_id not in class_id_to_bbox:
                for i, bbox in enumerate(bboxes):
                    if labels[i] == class_id:
                        normalized_bbox = [
                            bbox[0] / self.image_shape[1],  # x_min
                            bbox[1] / self.image_shape[0],  # y_min
                            bbox[2] / self.image_shape[1],  # x_max
                            bbox[3] / self.image_shape[0],  # y_max
                        ]

                        #make sure the bbox is between 0 and 1
                        normalized_bbox = [
                            np.clip(coord, 0, 1) for coord in normalized_bbox
                        ]

                        descrete_bbox = [
                            int(normalized_bbox[0] * self.num_bins),
                            int(normalized_bbox[1] * self.num_bins),
                            int(normalized_bbox[2] * self.num_bins),
                            int(normalized_bbox[3] * self.num_bins)
                        ]
                        # print(f"Normalized bbox for class_id {class_id}: {normalized_bbox}")
                        # print(f"Descrete bbox for class_id {class_id}: {descrete_bbox}")
                        class_id_to_bbox[class_id] = descrete_bbox
                        break


        for keyword, class_name, class_id in keywords:
            # suffix = suffix.replace(keyword, f"<p class_name='{class_name}'>{keyword}</p>")
            if class_id in class_id_to_bbox:
                bbox = class_id_to_bbox[class_id]
                bbox_tokens = [f"<loc{coord:04d}>" for coord in bbox]
                class_token = f"<seg{class_id:03d}>"
                keyword_pattern = fr"\b{re.escape(keyword)}\b"
                keyword_w_bbox = f"<p bbox={bbox_tokens[0]}{bbox_tokens[1]}{bbox_tokens[2]}{bbox_tokens[3]} id={class_token}>{keyword}</p>"
                suffix = re.sub(keyword_pattern, keyword_w_bbox, suffix)

        #Test if it can learn the dummy token
        # dummy_token = f"<loc0001>"
        # keyword = "dummy_keyword"
        # suffix = f"<p bbox={dummy_token}{dummy_token}{dummy_token}{dummy_token} id=\"{0}\">{keyword}</p>"
        # # print(f"Suffix: {suffix}")
        # print(f"Segmented IDs: {segmented_ids}")

        return suffix

    def load_segm_map(self, sample, segmented_ids):
        segment_path = sample["segment_slice_path"]
        seg_map = np.load(segment_path, allow_pickle=True)
        #filter the seg_map to only include the segmented ids
        filtered_seg_map = np.vectorize(lambda x: x if x in segmented_ids else 0)(seg_map)
        
        filtered_seg_map = torch.from_numpy(filtered_seg_map).unsqueeze(0).float()
        filtered_seg_map = F.interpolate(filtered_seg_map.unsqueeze(0), size=self.image_shape, mode='nearest')
        filtered_seg_map = filtered_seg_map[0].squeeze(0)  # Remove batch and channel dimensions
        return filtered_seg_map

    def load_report(self, data, language):
        question_type = "report"
        if USE_GROUNDED_REPORT_PROMPT:
            if language == "english":
                prompt = "Caption this image and detect keywords:"
            else:
                prompt = "Beschreibe dieses Bild und detektiere Schlüsselwörter:"
        else:
            if language == "english":
                prompt = "Caption this image:"
            elif language == "german":
                prompt = "Beschreibe dieses Bild:"

        prefix = self.load_prefix(data, language)
        suffix = self.get_report_content(data[language], "CleanedSentence")

        return prefix, prompt, suffix, question_type
    
    def load_vqa(self, data, language):
        vqa_list = list(data[language]["dicom_vqa"]) + list(data[language]["snippet_vqa"])
        if not vqa_list:
            return None
        
        selected_vqa = self._sample(vqa_list)
        
        prefix = selected_vqa["question"]
        question_type = selected_vqa["question_type"]
        if question_type == "multiple":
            prefix += "\n" + "\n".join(selected_vqa["choices"])
            
        suffix = selected_vqa["answer"]
        if language == "english":
            prompt = "Answer:"
        elif language == "german":
            prompt = "Antwort:"

        return prefix, prompt, suffix, question_type

    def load_prefix(self, data, language):
        # Determine which fields to include (none, one, or both)
        if self.prefix_style == "random" or self.prefix_style is None:
            prefix_type = self._sample(self.prefix_options)
        else:
            prefix_type = self.prefix_style
        lan_data = data[language]

        # Build prefix content
        if prefix_type == "none":
            prefix_content = ""
        elif prefix_type == "both":
            content_1 = self.get_report_content(lan_data, "klinische_angaben").strip()
            content_1_name = self.get_content_name("klinische_angaben", language)
            content_2 = self.get_report_content(lan_data, "fragestellung").strip()
            content_2_name = self.get_content_name("fragestellung", language)
            if content_1 == content_2:
                prefix_content = f"{content_1_name} {content_1}".strip()
            else:
                prefix_content = f"{content_1_name} {content_1}\n{content_2_name} {content_2}".strip()
        else:
            prefix_content = self.get_report_content(lan_data, prefix_type).strip()
            content_name = self.get_content_name(prefix_type, language)
            prefix_content = f"{content_name} {prefix_content}" if prefix_content else ""

        return prefix_content

    def get_content_name(self, prefix_type, language):
        if prefix_type == "fragestellung":
            if language == "english":
                return "Question:"
            else:
                return "Fragestellung:"
        elif prefix_type == "klinische_angaben":
            if language == "english":
                return "Clinical information:"
            else:
                return "Klinische angaben:"

    def get_report_content(self, lan_data, prefix_type):
        if prefix_type == "fragestellung":
            key = "ReportFragestellungCleaned"
        elif prefix_type == "klinische_angaben":
            key = "ReportKlinischeAngabenCleaned"
        else:
            key = prefix_type

        content = lan_data.get(key, "")
        if content:
            content = clean_punctuation(content)

        return content

    def _setup_random_generators(self):
        if self.eval_mode:
            self._rng = np.random.default_rng(42)
            self.random_choice_function = lambda seq: seq[self._rng.integers(len(seq))]
            self._random_float = lambda: self._rng.random()
        else:
            self.random_choice_function = lambda seq: np.random.choice(seq)
            self._random_float = lambda: np.random.rand()

    def _sample(self, options):
        if isinstance(options, set):
            options = list(options)
        choice = self.random_choice_function(options)
        return choice.item() if hasattr(choice, "item") else choice

    def _matches_filters(self, data, body_part, modality):
        if modality and modality.lower() != "all":
            sample_mod = data.get("modality")
            sample_mod = sample_mod.lower() if isinstance(sample_mod, str) else None
            if sample_mod != modality.lower():
                return False

        if body_part and body_part.lower() != "all":
            sample_body_part = data.get("body_part")
            sample_body_part = sample_body_part.lower() if isinstance(sample_body_part, str) else None
            if sample_body_part != body_part.lower():
                return False

        return True

    # def load_vqa(self, data, language):
    #     vqa_list = list(data[language]["dicom_vqa"]) + list(data[language]["snippet_vqa"])
    #     if not vqa_list:
    #         return None
    #     selected_vqa = np.random.choice(vqa_list)
    #     prefix = selected_vqa["question"]
    #     question_type = selected_vqa["question_type"]
    #     if question_type == "multiple":
    #         prefix += "\n" + "\n".join(selected_vqa["choices"])

    #     suffix = selected_vqa["answer"]
    #     prompt = "Answer:"
        
    #     return prefix, prompt, suffix, question_type
    def adaptively_cut_text(self, prompt: str, prefix: str, suffix: str):
        """
        Adaptively cut prefix and suffix to fit within effective_max_length.
        TODO this would better be done on token level after tokenizing each text.

        Args:
            prompt: The prompt text
            prefix: The prefix text to potentially cut
            suffix: The suffix text to potentially cut
    
        Returns:
            tuple: (new_prefix, new_suffix) with lengths adjusted to fit within effective_max_length
        """
        tokenized_prompt = self.tokenizer.encode(prompt)
        tokenized_prefix = self.tokenizer.encode(prefix)
        tokenized_suffix = self.tokenizer.encode(suffix)
        
        effective_max_length = self.max_length - len(tokenized_prompt)
        pref_len = len(tokenized_prefix)
        suff_len = len(tokenized_suffix)
        if pref_len + suff_len > effective_max_length:
            num_to_cut = pref_len + suff_len - effective_max_length
            new_prefix_len = max(math.floor(effective_max_length / 2), pref_len - num_to_cut)

            tokenized_prefix = tokenized_prefix[:new_prefix_len]
            prefix = self.tokenizer.decode(tokenized_prefix, skip_special_tokens=True)

            remaining_to_cut = new_prefix_len + suff_len - effective_max_length
            if remaining_to_cut > 0:
                new_suffix_len = suff_len - remaining_to_cut
                tokenized_suffix = tokenized_suffix[:new_suffix_len]
                suffix = self.tokenizer.decode(tokenized_suffix, skip_special_tokens=True)

        # prefix_split = prefix.split()
        # prefix_len = len(prefix_split)
        # suffix_split = suffix.split()
        # suffix_len = len(suffix_split)

        # if prefix_len + suffix_len > effective_max_length:
        #     # cut prefix up to half of the maximum length
        #     num_to_cut = prefix_len + suffix_len - effective_max_length
        #     new_prefix_len = max(math.floor(effective_max_length / 2), prefix_len - num_to_cut)
        #     prefix = " ".join(prefix_split[:new_prefix_len])
        #     remaining_to_cut = new_prefix_len + suffix_len - effective_max_length
        #     if remaining_to_cut > 0:
        #         new_suffix_len = suffix_len - remaining_to_cut
        #         suffix = " ".join(suffix_split[:new_suffix_len])
        return prefix, suffix


def get_collate_fn(processor, seq_len, torch_dtype, return_infos=False):
    def collate_fn(batch):
        images, prefixes, suffixes, infos, eval_mode = zip(*batch)

        output_kwargs = {"text_kwargs": {"max_length": seq_len, "truncation": False,
                                          "return_tensors": "pt", "padding": "longest"}}
        if eval_mode[0]:
            text_inputs = tokenize_text(processor, prefixes, None, output_kwargs)
        else:    
            text_inputs = tokenize_text(processor, prefixes, suffixes, output_kwargs)

        image_inputs = torch.stack(images).to(torch_dtype)
        # seg_map = torch.stack(seg_map).to(torch_dtype)
        # image_inputs = torch.stack(images)
        # print(image_inputs.shape)

        inputs = {"pixel_values": image_inputs, **text_inputs}
        if return_infos:
            return inputs, prefixes, suffixes, infos
            
        return inputs    
    return collate_fn


if __name__ == "__main__":
    # Example usage
    from transformers import PaliGemmaProcessor
    from radgrounder.dataset.segmentation.total_segmentator.read_segmentation import visualize_slices_with_segmentation
    # from radgrounder.dataset.image_preprocessing import NormalizationConfig
    MODEL_ID ="google/paligemma2-3b-pt-224"
    cache_dir = None
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)
    language = "all"  # or "german" or "all"
    normalization_config = NormalizationConfig(strategy=NormalizationType.MEDGEMMA)
    dataset = RefRad2DDetect(split="train", img_size=224, eval_mode=True, language=language, augment=True, max_length=100, prefix_style="none",
                            body_part="ALL", modality="all", only_segmented=True, tokenizer=processor.tokenizer, normalization=normalization_config)

    g = torch.Generator().manual_seed(43)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, generator=g,
                            collate_fn=get_collate_fn(processor, 100, torch.bfloat16, return_infos=True), num_workers=4, pin_memory=True)

    import json
    with open("radgrounder/dataset/segmentation/label_map/merged_label_map.json", "r") as f:
        label_map = json.load(f)
    label_map = {int(v): k for k, v in label_map.items()}

    from radgrounder.grounded_gemma.detection_augmentations import plot_img_with_boxes
    from matplotlib import pyplot as plt
    import re
    
    print(f"label_map: {label_map}")
    values = 0
    val_squares = 0
    num_elements = 0
    for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
        # image_tensor, prefix_full, suffix, info, eval_mode = batch
        inputs, prefixes, suffixes, infos = batch 
        for i in range(len(suffixes)):
            suffix = suffixes[i]
            # print(f"Prefix: {prefixes[i]}")
            # print(f"Suffix: {suffix}")
            # print(f"Info: {infos[i]}")
            bboxes = infos[i]["bboxes"]
            labels = infos[i]["labels"]
            # print("bboxes:", bboxes)
            # print("labels:", labels)
            img = inputs["pixel_values"][i]
            img = img.type(torch.float32)
            img = img.permute(1, 2, 0).cpu().numpy()

            # Find all matches in the suffix string
            matches = re.finditer(r"<p bbox=((<loc\d{4}>){4}) id=<seg(\d{3})>>", suffix)

            bbox_list = []
            label_list = []
            for match in matches:
                bbox_tokens = re.findall(r"<loc(\d{4})>", match.group(1))
                bbox = [int(224 * int(token) / 512) for token in bbox_tokens]
                label = int(match.group(3))
                bbox_list.append(bbox)
                label_list.append(label)

            print("Extracted bboxes:", bbox_list)
            print("Extracted labels:", label_list)
            print("suffix:", suffix)
            save_path = f"./samples_from_dataloader/detect/detect_image_{i}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plot_img_with_boxes(
                img,
                bbox_list,
                label_list,
                save_path=save_path,
                title=f"Detection Grounded RefRad2D Caption Sample",
                caption=suffix,
                prefix=prefixes[i] if prefixes else None,
                font_size=18,
                label_map=label_map,
                show=False,
            )

        exit()




        # values += pixel_values.sum().item()
        # val_squares += (pixel_values ** 2).sum().item()
        # num_elements += pixel_values.numel()
        # print(inputs.keys())

        # for i in range(len(prefixes)):
        #     print(f"Prefix: {prefixes[i]}")
        #     print(f"Suffix: {suffixes[i]}")
        #     print("*" * 20)
       
        # for i in range(len(prefix_full)):
        #     print(f"Image shape: {image_tensor[i].shape}\nPrefix: {prefix_full[i]}\nSuffix: {suffix[i]}")
        #     q_type = info["question_type"][i]
        #     print(f"Info: {q_type}")
        # exit()
    # avr_mean = values / num_elements if num_elements > 0 else 0
    # avr_std = np.sqrt((val_squares / num_elements) - (avr_mean ** 2)) if num_elements > 0 else 0
    # print(f"Mean: {avr_mean}, Std: {avr_std}")
    # print(f"Total elements: {num_elements}")