import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import torch.nn.functional as F
from typing import Optional

from radgrounder.grounded_gemma.augmentations import build_augmentation_pipeline
from radgrounder.dataset.image_preprocessing import (
    DEFAULT_NORMALIZATION,
    NormalizationConfig,
    NormalizationType,
    apply_normalization,
)

from PIL import Image

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
SLAKE_DATASET_MEAN = (0.2526479, 0.2526479, 0.2526479)
SLAKE_DATASET_STD = (0.2969230, 0.2969230, 0.2969230)

def apply_prompt_template(question: str, template_type: str) -> str:
    if template_type == "medgemma":
        prompt = "You may write out your argument before stating your final very short, \
definitive, and concise answer (if possible, a single word or the letter corresponding to your answer \
choice) X in the format 'Final Answer: X': "
        full_prompt = f"<image>\n{question}\n{prompt}"
    else:
        full_prompt = f"<image>\n{question}\nAnswer: "
    return full_prompt

class SlakeVQA(Dataset):
    def __init__(
        self,
        split="train",
        img_size=224,
        eval_mode=True,
        augment=False,
        max_length=100,
        language="en",
        question_types="all",
        return_dummy_mask=False,
        normalization: Optional[NormalizationConfig] = None,
        prompt_template_type: str = "refrad2d",
    ):
        print("Initializing SlakeVQA Dataset")
        self.normalization = normalization or DEFAULT_NORMALIZATION
        self.prompt_template_type = prompt_template_type
        print("Normalization strategy:", self.normalization.strategy.value)
        print("Prompt template type:", self.prompt_template_type)
        
        from radgrounder.paths import SLAKE_ROOT
        self.image_folder = os.path.join(str(SLAKE_ROOT), "imgs")
        if split == "val":
            split = "validate"

        if split not in ["train", "validate", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'validate', 'test'].")

        dataset_path = os.path.join(str(SLAKE_ROOT), f"{split}.json")
        with open(dataset_path, "r") as f:
            vqa_dataset = json.load(f)

        if question_types != "all":
            if question_types == "vqa_open":
                vqa_dataset = [item for item in vqa_dataset if item["answer_type"] == "OPEN"]
            elif question_types == "vqa_closed":
                vqa_dataset = [item for item in vqa_dataset if item["answer_type"] == "CLOSED"]
            else:
                raise ValueError(f"Invalid question_types: {question_types}. Must be one of ['all', 'vqa_open', 'vqa_closed'].")

        self.vqa_dataset = vqa_dataset
        self.split = split
        self.torch_dtype = torch.bfloat16
        self.eval_mode = eval_mode
        self.max_length = max_length
        self.image_shape = (img_size, img_size)
        self.augment = augment
        self.return_dummy_mask = return_dummy_mask
        self.channel_stats = {"mean": SLAKE_DATASET_MEAN, "std": SLAKE_DATASET_STD}
    
        # Remove non-English items from the dataset
        self.vqa_dataset = [item for item in self.vqa_dataset if item["q_lang"] == language]

        print(f"Using split: {split}, eval_mode: {eval_mode}")
        print(f"Dataset size: {len(self.vqa_dataset)}")

        if self.augment and not self.eval_mode:
            self.augmentation_pipeline = build_augmentation_pipeline(pixel_offset=0.1)




    def __len__(self):
        return len(self.vqa_dataset)

    def __getitem__(self, idx):
        # print(self.vqa_dataset[idx])
        item = self.vqa_dataset[idx]

        image_name = item["img_name"]
        image_path = os.path.join(self.image_folder, image_name)
        image = self.load_image(image_path)

        if self.augment and not self.eval_mode:
            image = self.augmentation_pipeline(image)
            
        image = F.interpolate(image.unsqueeze(0), size=self.image_shape, mode='bilinear', align_corners=False)
        image = image[0]

        image_tensor = image.type(self.torch_dtype)

        info = item
        info["image_path"] = image_path

            
        template_type = self.prompt_template_type
        prefix = apply_prompt_template(item["question"], template_type=template_type)
        suffix = item["answer"]
        info["system_instruction"] = "You are a helpful medical assistant."
        
        if self.return_dummy_mask:
            dummy_mask = None
            return image_tensor, dummy_mask, prefix, suffix, info, self.eval_mode

        return image_tensor, prefix, suffix, info, self.eval_mode

    def load_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.image_shape)
                img = torch.from_numpy(np.array(img))
                image = img.permute(2, 0, 1).float()
                image = image / 255.0
                image = apply_normalization(
                    image,
                    modality=None,
                    config=self.normalization,
                    channel_stats=self.channel_stats,
                )
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        return image


if __name__ == "__main__":
    SEQLEN = 200
    TORCH_DTYPE = torch.bfloat16

    norm_config = NormalizationConfig(NormalizationType.MEDGEMMA)
    dataset = SlakeVQA(split="validate",
                       img_size=224,
                       eval_mode=True,
                       augment=False,
                       language="en",
                       question_types="vqa_closed",
                       normalization=norm_config)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    # from radgrounder.grounded_gemma.train_gemma3 import get_collate_fn
    from radgrounder.dataset.segmentation.refrad2d_detect_merged import RefRad2DDetectMerged, get_collate_fn
    from transformers import AutoProcessor
    model_id = "google/medgemma-4b-it"
    cache_dir = None
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                            collate_fn=get_collate_fn(processor, SEQLEN, TORCH_DTYPE, return_infos=True))
    total_sum = 0.0
    total_count = 0
    for batch in dataloader:
        # images, questions, answers, infos, eval_modes = batch
        print(batch[0]["input_ids"].shape)
        message = processor.decode(batch[0]["input_ids"][0], skip_special_tokens=True)
        print("Message:", message)
        
        break
        # print(images.shape)
        # for answer, question in zip(answers, questions):
        #     print(f"Q: {question} | A: {answer}")
            
        

    # for batch in dataloader:
    #     images, questions, answers, infos, eval_modes = batch
    #     print("Questions:", questions)
    #     print("Answers:", answers)
    #     total_sum += images.sum().item()
    #     total_count += images.numel()
    #     # For std calculation
    #     if total_count == images.numel():
    #         sum_squared = (images ** 2).sum().item()
    #     else:
    #         sum_squared += (images ** 2).sum().item()

    # mean = total_sum / total_count
    # std = (sum_squared / total_count - mean ** 2) ** 0.5
    # print(f"Mean: {mean}, Std: {std}")
        # for i in range(len(questions)):
        #     print(f"Sample {i}:")
        #     print(f"  Image shape: {images[i].shape}")
        #     print(f"  Question: {questions[i]}")
        #     print(f"  Answer: {answers[i]}")
        #     # print(f"  Info: {infos[i]}")
        #     print(f"  Eval mode: {eval_modes[i]}")
        #     print("-" * 40)
        # exit()