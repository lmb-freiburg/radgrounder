import inspect

USE_GROUNDED_REPORT_PROMPT = False

class DatasetManager:
    def __init__(self, dataset_name: str, split: str = "train", use_grounded_prompt: bool = False, **kwargs):
        global USE_GROUNDED_REPORT_PROMPT
        USE_GROUNDED_REPORT_PROMPT = use_grounded_prompt
        self.dataset_name = dataset_name
        self.split = split
        if dataset_name == "refrad2d_v18":
            from radgrounder.dataset.vqa_dataset.refrad2d_v18 import RefRad2DV18, get_collate_fn
            refrad2dv18_params = inspect.signature(RefRad2DV18.__init__).parameters
            allowed_keys = set(refrad2dv18_params) - {"self", "split"}
            refrad2dv18_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            self.dataset = RefRad2DV18(split=split, **refrad2dv18_kwargs)
            self.get_collate_fn = get_collate_fn
        elif dataset_name == "refrad2d_vqa":
            raise DeprecationWarning("refrad2d_vqa is deprecated, please use refrad2d_v18 instead.")
            from radgrounder.dataset.vqa_dataset.refrad2d_vqa_dataset import RefRad2DVQA, get_collate_fn
            refrad2dvqa_params = inspect.signature(RefRad2DVQA.__init__).parameters
            allowed_keys = set(refrad2dvqa_params) - {"self", "split"}
            refrad2dvqa_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            self.dataset = RefRad2DVQA(split=split, **refrad2dvqa_kwargs)
            self.get_collate_fn = get_collate_fn
        elif dataset_name == "slake_vqa":
            from radgrounder.dataset.vqa_dataset.slake_vqa import SlakeVQA
            from radgrounder.dataset.vqa_dataset.refrad2d_vqa_dataset import get_collate_fn
            # Remove 'dataset_size' from kwargs if present
            slake_params = inspect.signature(SlakeVQA.__init__).parameters
            allowed_keys = set(slake_params) - {"self", "split"}
            slake_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            slake_kwargs["language"] = "en"
            self.dataset = SlakeVQA(split=split, **slake_kwargs)
            self.get_collate_fn = get_collate_fn
        elif dataset_name == "vqa_rad":
            from radgrounder.dataset.vqa_dataset.vqa_rad import VqaRad
            from radgrounder.dataset.vqa_dataset.refrad2d_vqa_dataset import get_collate_fn
            vqarad_params = inspect.signature(VqaRad.__init__).parameters
            allowed_keys = set(vqarad_params) - {"self", "split"}
            vqarad_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            self.dataset = VqaRad(split=split, **vqarad_kwargs)
            self.get_collate_fn = get_collate_fn
        elif dataset_name == "refrad2d_detect": 
            from radgrounder.dataset.segmentation.refrad2d_detect import RefRad2DDetect, get_collate_fn
            refrad2d_detect_params = inspect.signature(RefRad2DDetect.__init__).parameters
            allowed_keys = set(refrad2d_detect_params) - {"self", "split"}
            refrad2d_detect_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            self.dataset = RefRad2DDetect(split=split, **refrad2d_detect_kwargs)
            self.get_collate_fn = get_collate_fn
        elif dataset_name == "refrad2d_detect_vqa":
            from radgrounder.dataset.segmentation.refrad2d_detect_vqa import RefRad2DDetectVQA, get_collate_fn
            refrad2d_detect_vqa_params = inspect.signature(RefRad2DDetectVQA.__init__).parameters
            allowed_keys = set(refrad2d_detect_vqa_params) - {"self", "split"}
            refrad2d_detect_vqa_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            self.dataset = RefRad2DDetectVQA(split=split, **refrad2d_detect_vqa_kwargs)
            self.get_collate_fn = get_collate_fn
        elif dataset_name == "refrad2d_detect_merged":
            from radgrounder.dataset.segmentation.refrad2d_detect_merged import RefRad2DDetectMerged, get_collate_fn
            refrad2d_detect_merged_params = inspect.signature(RefRad2DDetectMerged.__init__).parameters
            allowed_keys = set(refrad2d_detect_merged_params) - {"self", "split"}
            refrad2d_detect_merged_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            self.dataset = RefRad2DDetectMerged(split=split, **refrad2d_detect_merged_kwargs)
            self.get_collate_fn = get_collate_fn
        elif dataset_name == "refrad2d_segment":
            from radgrounder.dataset.segmentation.refrad2d_segment import RefRad2DSegment, get_collate_fn
            refrad2d_segment_params = inspect.signature(RefRad2DSegment.__init__).parameters
            allowed_keys = set(refrad2d_segment_params) - {"self", "split"}
            refrad2d_segment_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            self.dataset = RefRad2DSegment(split=split, **refrad2d_segment_kwargs)
            self.get_collate_fn = get_collate_fn
        elif dataset_name == "refrad2d_segment_merged":
            from radgrounder.dataset.segmentation.refrad2d_segment_merged import RefRad2DSegmentMerged
            from radgrounder.dataset.segmentation.refrad2d_segment import get_collate_fn
            refrad2d_segment_merged_params = inspect.signature(RefRad2DSegmentMerged.__init__).parameters
            allowed_keys = set(refrad2d_segment_merged_params) - {"self", "split"}
            refrad2d_segment_merged_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
            self.dataset = RefRad2DSegmentMerged(split=split, **refrad2d_segment_merged_kwargs)
            self.get_collate_fn = get_collate_fn
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    
    
