# Prompt for Medical Keyword Extraction and Classification

**You are an expert medical annotator specializing in the segmentation of anatomical structures from MR and CT images.**

Your task is to carefully analyze a given medical sentence and identify all mentions of anatomical structures that can be segmented. You must then classify each identified structure into one of the predefined categories provided below.

---

### **Instructions**

1.  **Identify Keywords**: From the input sentence, extract the specific words or phrases that refer to an anatomical structure present in the category list.

2.  **Classify Keywords**: For each extracted keyword, assign it to the most precise category from the provided list.
    *   Pay close attention to laterality (e.g., "left" or "right"). For example, "the left kidney" should be classified as `kidney_left`.
    *   If a general term is used, classify it accordingly (e.g., "heart" should be `heart`, and "lungs" should be classified as `lung_left` and `lung_right` if no specific lobe is mentioned).

3.  **Handle Specificity**: When both a general organ and a specific part of it are mentioned, prioritize the most specific category available. For example, in "the upper lobe of the left lung," the keyword is "upper lobe of the left lung" and the category is `lung_upper_lobe_left`.

4.  **Output Format**: The output must be a JSON array of objects. Each object should contain two keys:
    *   `"keyword"`: The exact text of the anatomical structure identified in the sentence.
    *   `"category"`: The corresponding category from the provided list.

5.  **No Matches**: If the sentence does not contain any of the anatomical structures listed in the categories, return an empty array `[]`.

---

### **Anatomical Categories List**

[
    "adrenal_gland_left", "adrenal_gland_right", "aorta", "atrial_appendage_left",
    "autochthon_left", "autochthon_right", "brachiocephalic_trunk",
    "brachiocephalic_vein_left", "brachiocephalic_vein_right", "brain",
    "clavicula_left", "clavicula_right", "colon", "common_carotid_artery_left",
    "common_carotid_artery_right", "costal_cartilages", "duodenum", "esophagus",
    "femur_left", "femur_right", "gallbladder", "gluteus_maximus_left",
    "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right", "heart", "hip_left", "hip_right",
    "humerus_left", "humerus_right", "iliac_artery_left", "iliac_artery_right",
    "iliac_vena_left", "iliac_vena_right", "iliopsoas_left", "iliopsoas_right",
    "inferior_vena_cava", "intervertebral_discs", "kidney_cyst_left",
    "kidney_cyst_right", "kidney_left", "kidney_right", "liver", "lung_left",
    "lung_lower_lobe_left", "lung_lower_lobe_right", "lung_middle_lobe_right",
    "lung_right", "lung_upper_lobe_left", "lung_upper_lobe_right", "pancreas",
    "portal_vein_and_splenic_vein", "prostate", "pulmonary_vein", "rib_left_1",
    "rib_left_10", "rib_left_11", "rib_left_12", "rib_left_2", "rib_left_3",
    "rib_left_4", "rib_left_5", "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9",
    "rib_right_1", "rib_right_10", "rib_right_11", "rib_right_12", "rib_right_2",
    "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6", "rib_right_7",
    "rib_right_8", "rib_right_9", "sacrum", "scapula_left", "scapula_right", "skull",
    "small_bowel", "spinal_cord", "spleen", "sternum", "stomach",
    "subclavian_artery_left", "subclavian_artery_right", "superior_vena_cava",
    "thyroid_gland", "trachea", "urinary_bladder", "vertebrae", "vertebrae_C1",
    "vertebrae_C2", "vertebrae_C3", "vertebrae_C4", "vertebrae_C5", "vertebrae_C6",
    "vertebrae_C7", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4",
    "vertebrae_L5", "vertebrae_S1", "vertebrae_T1", "vertebrae_T10", "vertebrae_T11",
    "vertebrae_T12", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
    "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9"
]

### **Examples**
Input:"Diffuse liver Metastases. A Metastase in liver segment IV measures 12 mm, another central liver Metastase measures 2.2 cm."
Output:
[
    {
        "keyword": "liver",
        "category": "liver"
    },
    {
        "keyword": "liver segment IV",
        "category": "liver"
    },
    {
        "keyword": "central liver",
        "category": "liver"
    }
]

Input:"Inhomogeneous, smoothly marginated structure in the area of the dorsal left kidney measuring 1.1 x 0.6 cm."
Output:
[
    {
        "keyword": "left kidney",
        "category": "kidney_left"
    }
]

Input:"A lymph node measuring 1.6 x 1.1 cm below the pancreas."
Output:
[
    {
        "keyword": "pancreas",
        "category": "pancreas"
    }
]

Input:"Size-regressing lymph nodes deep cervical with 8 mm"
Output:
[]

Input:"From the right dorsolateral side projecting calcification into the spinal canal at the level of L5 with spinal stenosis."
Output:
[
    {
        "keyword": "spinal canal",
        "category": "spinal_cord"
    },
    {
        "keyword": "L5",
        "category": "vertebrae_L5"
    }
]

Input:"Intense radionuclide uptake in the bed of the resected T12 vertebra with spinal soft tissue component, in the adjacent partially resected T11 vertebra, and focally in the right ilium."
Output:
[
    {
        "keyword": "T12 vertebra",
        "category": "vertebrae_T12"
    },
    {
        "keyword": "spinal soft tissue component",
        "category": "spinal_cord"
    },
    {
        "keyword": "T11 vertebra",
        "category": "vertebrae_T11"
    },
    {
        "keyword": "right ilium",
        "category": "hip_right"
    }
]

Input:"Intensive involvement at the level of the 10th and 11th ribs on the right."
Output:
[
    {
        "keyword": "10th",
        "category": "rib_right_10"
    },
    {
        "keyword": "11th ribs on the right",
        "category": "rib_right_11"
    }
]

Input:"Mediolateral disc herniation at the L4/5 motion segment on the left with stenosis of the lateral recess and irritation of the L5 nerve root."
Output:
[
    {
        "keyword": "disc",
        "category": "intervertebral_discs"
    },
    {
        "keyword": "L4",
        "category": "vertebrae_L4"
    },
    {
        "keyword": "L5",
        "category": "vertebrae_L5"
    },
    {
        "keyword": "lateral recess",
        "category": "spinal_cord"
    },
    {
        "keyword": "L5 nerve root",
        "category": "spinal_cord"
    }
]

--

Now find and classify the keywords in the following text: