from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import re
import tqdm

from radgrounder.grounded_gemma.augmentations import build_segment_aug_pipeline, build_augmentation_pipeline
from radgrounder.dataset.segmentation.refrad2d_segment import SEG_START, SEG_END, get_collate_fn
from radgrounder.dataset.image_preprocessing import (
    DEFAULT_NORMALIZATION,
    NormalizationConfig,
    NormalizationType,
    apply_normalization,
)
from radgrounder.dataset.dataset_utils import (
    load_sampled_dataset,
    load_snippet2rsopids,
)

GRAYSCALE_PROB = 0.2



class RefRad2DSegmentVQA(Dataset):
    def __init__(
        self,
        split="train",
        img_size=224,
        eval_mode=False,
        language="all",
        augment=False,
        dataset_size=None,
        max_length=100,
        tokenizer=None,
        normalization: Optional[NormalizationConfig] = None,
        modality=None,
    ):
        print("Initializing RefRad2DSegmentVQA dataset")
        self.seed = 42
        self.eval_mode = eval_mode
        self.normalization = normalization or DEFAULT_NORMALIZATION
        print("Normalization strategy:", self.normalization.strategy.value)
        
        self.sampled_dataset = load_sampled_dataset(modality=modality)


        script_dir = os.path.dirname(os.path.abspath(__file__))
        class_name_2_organ_name_path = os.path.join(script_dir, "label_map", "class_2_organ_name_english.json")
        with open(class_name_2_organ_name_path, "r", encoding="utf-8") as f:
            self.class_name_2_organ_name_eng = json.load(f)
        self.organ_name_2_class_name_eng = {v: k for k, v in self.class_name_2_organ_name_eng.items()}

        class_name_2_organ_name_path = os.path.join(script_dir, "label_map", "class_2_organ_name_german.json")
        with open(class_name_2_organ_name_path, "r", encoding="utf-8") as f:
            self.class_name_2_organ_name_german = json.load(f)
        self.organ_name_2_class_name_german = {v: k for k, v in self.class_name_2_organ_name_german.items()}


        template_path = os.path.join(script_dir, "detect_vqa", "vqa_templates_english.json")
        with open(template_path, "r", encoding="utf-8") as f:
            self.question_templates_eng = json.load(f)

        template_path = os.path.join(script_dir, "detect_vqa", "vqa_templates_german.json")
        with open(template_path, "r", encoding="utf-8") as f:
            self.question_templates_german = json.load(f)

        with open(os.path.join(script_dir, "label_map", "merged_label_map.json"), "r", encoding="utf-8") as f:
            self.class_name_2_id = json.load(f)

        self.id_2_class_name = {v: k for k, v in self.class_name_2_id.items()}


        # Split into train/validation/test using snippet-based splits
        print(f"Using split: {split}, eval_mode: {eval_mode}")

        self.split = split
        self.snippet2rsopids = load_snippet2rsopids(split, None)
        self.filtered_dataset = self._filter_dataset_by_split(self.snippet2rsopids, dataset_size)

        if not self.filtered_dataset:
            raise ValueError(f"No samples available for split '{split}' after filtering.")

        print(f"{self.split} dataset size: {len(self.filtered_dataset)}")

        self.keys = list(self.filtered_dataset.keys())
        self.image_shape = (img_size, img_size)    
        self.dataset_stats = {"avr_ct_mean": -610.1908535827807, "avr_ct_std": 737.3882466073229,
                        "avr_mr_mean": 141.1154960399739, "avr_mr_std": 255.40429635138338}
        self.augment = augment
        print(f"augment: {self.augment}")
        if self.augment:
            self.default_augmentation = build_augmentation_pipeline()
            self.seg_aug_pipeline = build_segment_aug_pipeline()

        self.torch_dtype = torch.bfloat16
        self.language = language
        if self.language not in ["english", "german", "all"]:
            raise ValueError(f"Invalid language: {self.language}. Must be one of ['english', 'german', 'all'].")
        self.max_length = max_length
        self._setup_random_generators()
        
        self.replace_braces_pattern = re.compile(r"\{[^}]*\}")
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided in RefRad2DSegmentVQA")
        self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        slice_name = self.keys[idx]

        sample = self.filtered_dataset[slice_name]

        # Load image
        seg_path = sample["segment_slice_path"]
        image_path = sample["scan_slice_path"]
        modality = sample["modality"]
        image, seg_mask = self.load_image_and_seg_mask(image_path, seg_path, modality)

        question, answer, language, question_type, segmented_keywords = self.load_segmentation_qa(sample, seg_mask)
        question, answer = self.adaptively_cut_text(question, answer, self.max_length)

        image_tensor = image.type(self.torch_dtype)
        
        binary_masks = self.convert_mask_to_binary(seg_mask, segmented_keywords)
        prefix_full = f"<image>\n{question} "
        info = {
            "image_path": image_path,
            "modality": modality,
            "language": language,
            "question_type": question_type,
            "binary_mask_keywords": segmented_keywords,
        }
        
        return image_tensor, binary_masks, prefix_full, answer, info, self.eval_mode

    def load_image_and_seg_mask(self, slice_path, segment_path=None, modality=None):
        slice_img = np.load(slice_path)
        slice_img = torch.from_numpy(slice_img)
        if modality == "CT":
            slice_img = torch.clip(slice_img, min=-1000)
        slice_img = slice_img.unsqueeze(0)  # Add channel dimension

        seg_mask = None
        if segment_path:
            seg_mask = np.load(segment_path, allow_pickle=True)
            seg_mask = torch.from_numpy(seg_mask).unsqueeze(0).float()

        if self.augment:
            if seg_mask is not None:
                aug_input = {"image": slice_img, "seg_mask": seg_mask}
                aug_output = self.seg_aug_pipeline(aug_input)
                slice_img = aug_output["image"]
                seg_mask = aug_output["seg_mask"]
            else:
                slice_img = self.default_augmentation(slice_img)

        slice_img = self.normalize_image(slice_img, modality=modality, eval_mode=self.eval_mode)
        
        #resize to 224 x 224
        image = F.interpolate(slice_img.unsqueeze(0), size=self.image_shape, mode='bilinear', align_corners=False)
        image = image[0]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if seg_mask is not None:
            seg_mask = F.interpolate(seg_mask.unsqueeze(0), size=self.image_shape, mode='nearest')

        return image, seg_mask

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


    def load_segmentation_qa(self, sample, seg_mask):
        language = self.language
        if language == "all":
            language = self._sample(["english", "german"])

        if language == "english":
            class_name_2_organ_name = self.class_name_2_organ_name_eng
            question_templates = self.question_templates_eng
        else:
            class_name_2_organ_name = self.class_name_2_organ_name_german
            question_templates = self.question_templates_german
        
        segmented_class_names = []
        segmented_organ_names = []

        # gives sorted unique classes, thats why the order of the organs stays the same
        present_labels = np.unique(seg_mask.numpy().astype(int))
        present_labels = present_labels[present_labels != 0]

        for label in present_labels:
            class_name = self.id_2_class_name.get(label)
            if class_name:
                segmented_class_names.append(class_name)
                organ_name = class_name_2_organ_name.get(class_name)
                if organ_name:
                    segmented_organ_names.append(organ_name)


        selected_template = self._sample(question_templates)

        question_type = selected_template["question_type"]
        question_temp = selected_template["question"]
        answer_temp = selected_template["answer"]
        all_organ_names = set(class_name_2_organ_name.values())
        segmented_keywords = []

        force_negative_response = len(segmented_organ_names) == 0

        if "negative" in question_type or force_negative_response:
            negative_organ_names = all_organ_names - set(segmented_organ_names)
            if not negative_organ_names: # handle case where all organs are present
                selected_organ = self._sample(list(all_organ_names))
            else:
                selected_organ = self._sample(list(negative_organ_names))
            question = self.replace_braces(question_temp, selected_organ)
            answer = self.replace_braces(answer_temp, selected_organ)
        elif "affirmative" in question_type:
            affirmative_organ_names = set(segmented_organ_names)
            if not affirmative_organ_names: # Should be handled by force_negative_response
                 # Fallback to a negative question if no organs are segmented
                negative_organ_names = all_organ_names
                selected_organ = self._sample(list(negative_organ_names))
                question = self.replace_braces(question_temp, selected_organ)
                answer = self.replace_braces(answer_temp, selected_organ)
                answer = answer.replace(selected_organ, f"{SEG_START}{selected_organ}{SEG_END}")
            else:
                selected_organ = self._sample(list(affirmative_organ_names))
                question = self.replace_braces(question_temp, selected_organ)
                answer = self.replace_braces(answer_temp, selected_organ)
                answer, segmented_keywords = self.add_segmentation_tokens(answer, selected_organ, language)
        else: # "all" or other types
            question = question_temp
            organ_list_str = ", ".join(segmented_organ_names)
            answer = self.replace_braces(answer_temp, organ_list_str)
            answer, segmented_keywords = self.add_segmentation_tokens(answer, segmented_organ_names, language)

        return question, answer, language, question_type, segmented_keywords


    def replace_braces(self, template, value):
        if isinstance(value, list):
            value = ", ".join(value)
        return self.replace_braces_pattern.sub(value, template)

    def add_segmentation_tokens(self, answer, organ_names, language):
        if not isinstance(organ_names, list):
            organ_names = [organ_names]

        if language == "english":
            organ_name_2_class_name = self.organ_name_2_class_name_eng
        else:
            organ_name_2_class_name = self.organ_name_2_class_name_german

        segmented_keywords = []
        for organ_name in organ_names:
            if organ_name in answer:
                class_name = organ_name_2_class_name.get(organ_name)
                if class_name:
                    class_id = self.class_name_2_id.get(class_name)
                    if class_id is not None:
                        count = answer.count(organ_name)
                        answer = answer.replace(organ_name, f"{SEG_START}{organ_name}{SEG_END}")
                        for _ in range(count):
                            segmented_keywords.append((organ_name, class_id))
        return answer, segmented_keywords

    def convert_mask_to_binary(self, seg_mask, segmented_keywords):
        if seg_mask is None or not segmented_keywords:
            dtype = seg_mask.dtype if seg_mask is not None else torch.float32
            return torch.empty((0, *self.image_shape), dtype=dtype)
        binary_masks = torch.zeros((len(segmented_keywords), *self.image_shape), dtype=seg_mask.dtype)
        for i, (keyword, class_id) in enumerate(segmented_keywords):
            binary_masks[i] = (seg_mask == class_id).float()
        return binary_masks

    def adaptively_cut_text(self, prefix: str, suffix: str, effective_max_length: int):
        """
        Adaptively cut prefix and suffix to fit within effective_max_length.
        """

        tokenized_prefix = self.tokenizer.encode(prefix)
        tokenized_suffix = self.tokenizer.encode(suffix)
        pref_len = len(tokenized_prefix)
        suff_len = len(tokenized_suffix)
        if pref_len + suff_len > effective_max_length:
            max_length_for_suffix = effective_max_length - pref_len
            tokenized_suffix = tokenized_suffix[:max_length_for_suffix]
            tokenized_prefix = tokenized_prefix[:effective_max_length]
            prefix = self.tokenizer.decode(tokenized_prefix)
            suffix = self.tokenizer.decode(tokenized_suffix)

            # find the last </seg> in suffix and discard the rest of the text
            last_p_index = suffix.rfind(SEG_END)
            if last_p_index != -1:
                suffix = suffix[:last_p_index + len(SEG_END)]
            

        return prefix, suffix

    def _setup_random_generators(self):
        if self.eval_mode:
            self._rng = np.random.default_rng(self.seed)
            self.random_choice_function = lambda seq: seq[self._rng.integers(len(seq))]
            self._random_float = lambda: float(self._rng.random())
        else:
            self.random_choice_function = lambda seq: seq[int(np.random.randint(len(seq)))]
            self._random_float = lambda: float(np.random.random())

    def _sample(self, options):
        if isinstance(options, set):
            options = list(options)
        choice = self.random_choice_function(options)
        return choice.item() if hasattr(choice, "item") else choice

    def _filter_dataset_by_split(self, snippet2rsopids, dataset_size):
        allowed_rsopids = {rsopid for rsopids in snippet2rsopids.values() for rsopid in rsopids}
        rsopid_to_slice_names = {}
        for slice_name, sample in self.sampled_dataset.items():
            rsopid = sample.get("rsopid")
            if rsopid is None or rsopid not in allowed_rsopids:
                continue
            rsopid_to_slice_names.setdefault(rsopid, []).append(slice_name)

        ordered_slice_names = []
        for rsopids in snippet2rsopids.values():
            for rsopid in rsopids:
                ordered_slice_names.extend(rsopid_to_slice_names.get(rsopid, []))

        seen = set()
        unique_slice_names = []
        for name in ordered_slice_names:
            if name not in seen:
                seen.add(name)
                unique_slice_names.append(name)

        if dataset_size is not None:
            if dataset_size > len(unique_slice_names):
                print(
                    f"Dataset size {dataset_size} is larger than the filtered dataset size {len(unique_slice_names)}. "
                    "Using the actual dataset size."
                )
                dataset_size = len(unique_slice_names)
            unique_slice_names = unique_slice_names[:dataset_size]

        return {k: self.sampled_dataset[k] for k in unique_slice_names}




if __name__ == "__main__":
    # Example usage
    import os
    from collections import defaultdict
    from transformers import PaliGemmaProcessor
    from radgrounder.dataset.segmentation.total_segmentator.read_segmentation import visualize_slices_with_segmentation
    MODEL_ID ="google/paligemma2-3b-pt-224"
    cache_dir = None
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)
    processor.tokenizer.add_tokens([SEG_START, SEG_END])
    
    language = "english"
    dataset = RefRad2DSegmentVQA(img_size=224, eval_mode=True, language=language, augment=False, max_length=200,
                                tokenizer=processor.tokenizer, split="val", dataset_size=1000)
    
    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, generator=g, collate_fn=get_collate_fn(processor, 200, torch.bfloat16, return_infos=True), num_workers=4, pin_memory=True)
    
    with open("radgrounder/dataset/segmentation/label_map/merged_label_map.json", "r") as f:
        label_map = json.load(f)
    label_map = {int(v): k for k, v in label_map.items()}

    for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
        inputs, prefixes, suffixes, infos, binary_gt_masks, seg_token_pos, mask_labels = batch
        pixel_values = inputs["pixel_values"]

        # for mask_id, (binary_map, seg_pos) in enumerate(zip(binary_gt_masks, seg_token_pos)):
        #     i = seg_pos[0].item()
        #     slice_name = infos[i]["image_path"].split("/")[-1].split(".")[0]
        #     save_path = f"./samples_from_dataloader/segment_vqa_{slice_name}_{seg_pos[1]}.png"
        #     image_slice = pixel_values[i].cpu().float().permute(1, 2, 0).numpy()
        #     binary_seg_slice = binary_map.cpu().int().numpy()

        #     class_id = mask_labels[mask_id]
            
        #     seg_slice = binary_seg_slice * class_id
            
        #     # Visualize the slices with segmentation
        #     visualize_slices_with_segmentation(image_slice, seg_slice, label_map, slice_idx=0, modality=infos[i]["language"], save_path=save_path, caption=suffixes[i])
        
        samples_with_masks = []
        segment_maps = defaultdict(list)
        for mask_id, (binary_map, seg_pos) in enumerate(zip(binary_gt_masks, seg_token_pos)):
            i = seg_pos[0].item()
            samples_with_masks.append(i)
            slice_name = infos[i]["image_path"].split("/")[-1].split(".")[0]
            segment_maps[slice_name].append((mask_id, binary_map, seg_pos))

        for slice_name, masks in segment_maps.items():
            first_mask_id, first_binary_map, first_seg_pos = masks[0]
            image_index = first_seg_pos[0].item()

            combined_seg_slice = torch.zeros_like(first_binary_map, dtype=torch.float32)
            for mask_id, binary_map, seg_pos in masks:
                class_id = mask_labels[mask_id]
                class_value = int(class_id.item()) if hasattr(class_id, "item") else int(class_id)
                combined_seg_slice = torch.maximum(
                    combined_seg_slice,
                    binary_map.to(dtype=combined_seg_slice.dtype) * class_value,
                )

            combined_seg_slice = combined_seg_slice.cpu().numpy().astype(int)
            image_slice = pixel_values[image_index].cpu().float().permute(1, 2, 0).numpy()
            save_path = f"./samples_from_dataloader/segment_vqa/{language}/{slice_name}_combined.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            caption = suffixes[image_index]
            #example <seg>liver</seg> extract liver
            # pattern = r"<seg>(.*?)</seg>"
            # regex = r"<.*?>|</.*?>"
            # caption = re.sub(regex, "", caption)
            print(caption)
            visualize_slices_with_segmentation(
                image_slice,
                combined_seg_slice,
                label_map,
                slice_idx=0,
                title="RefRad2D G-VQA Segmentation Sample",
                save_path=save_path,
                caption=caption,
                prefix=prefixes[image_index],
                font_size=20
            )
            
        #save some negative samples without masks
        negative_samples = [i for i in range(len(infos)) if i not in samples_with_masks]
        negative_samples = negative_samples[:10]  # Save only first 5 negative samples
        for image_index in negative_samples:
            slice_name = infos[image_index]["image_path"].split("/")[-1].split(".")[0]
            image_slice = pixel_values[image_index].cpu().float().permute(1, 2, 0).numpy()
            combined_seg_slice = np.zeros(image_slice.shape[:2], dtype=int)  # Empty segmentation
            save_path = f"./samples_from_dataloader/segment_vqa/{language}/negative_{slice_name}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            caption = suffixes[image_index]
            #example <seg>liver</seg> extract liver
            # pattern = r"<seg>(.*?)</seg>"
            # regex = r"<.*?>|</.*?>"
            # caption = re.sub(regex, "", caption)
            print(caption)
            visualize_slices_with_segmentation(
                image_slice,
                combined_seg_slice,
                label_map,
                slice_idx=0,
                title="RefRad2D G-VQA Segmentation Sample - No Masks",
                save_path=save_path,
                caption=caption,
                prefix=prefixes[image_index],
                font_size=20
            )

        exit()
