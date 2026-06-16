import re
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from ovqa.metrics.simple import compare_f1, get_metric_is_equal_default_prep

class DetectGroundingEvaluator:
    def __init__(self, model_name="google/embeddinggemma-300m", cache_folder="./models", special_token="p"):
        """
        Initializes the GroundingEvaluator with a sentence-transformer model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"GroundingEvaluator - Using device: {self.device}")
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        self.model.to(self.device)
        self.special_token = special_token
        self.detect_pattern = re.compile(rf"<{self.special_token} bbox=<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)> id=<seg(\d+)>>(.*?)</{self.special_token}>", re.IGNORECASE)
        
    def extract_keywords(self, text: str) -> list[str]:
        """
        Extracts and cleans keywords from text wrapped in special tags.
        Example: "<p bbox=... id=...>liver segment II</p>" -> "liver segment II"
        """
        # Find all content within the special tags

        # Remove all nested tags (like bbox, loc, id) and clean up
        # The inner sub removes any tag-like <> structures.
        # keywords = [re.sub(r"<[^>]+>", "", tag_content).strip() for tag_content in pattern.findall(text)]
        
        #Find the bbox coordinates of the keywords
        keywords = []
        bboxes = []
        # if text is None:
        #     return keywords, bboxes
        if type(text) != str:
            text = str(text)
            
        for tag_content in self.detect_pattern.findall(text):
            # Extract bbox numbers from the tag content
            bbox = [int(num) / 512 for num in tag_content[:4]]
            bbox_id = int(tag_content[4])
            keyword = tag_content[5].strip().lower()
            keywords.append(keyword)
            bboxes.append((bbox, bbox_id))

        # Return a list of non-empty, lowercase keywords
        return keywords, bboxes

    def match_keywords(self, gt_keywords, pred_keywords, gt_bboxes, pred_bboxes):
        # Generate embeddings for both lists
        gt_embeddings = self.model.encode(gt_keywords, convert_to_tensor=True)
        pred_embeddings = self.model.encode(pred_keywords, convert_to_tensor=True)

        # Ensure embeddings are 2D (add batch dimension if needed)
        if gt_embeddings.dim() == 1:
            gt_embeddings = gt_embeddings.unsqueeze(0)
        if pred_embeddings.dim() == 1:
            pred_embeddings = pred_embeddings.unsqueeze(0)

        # Normalize embeddings for cosine similarity calculation
        gt_embeddings_norm = F.normalize(gt_embeddings, p=2, dim=1)
        pred_embeddings_norm = F.normalize(pred_embeddings, p=2, dim=1)

        # Calculate the similarity matrix
        similarity_matrix = torch.mm(gt_embeddings_norm, pred_embeddings_norm.T)

        results = []
        # Iterate through each ground truth keyword to find its best match
        while similarity_matrix.numel() > 0:
            # Find the highest similarity score and its index for the current GT keyword
            best_match_score = torch.max(similarity_matrix)
            best_match_index = torch.argmax(similarity_matrix)
            row, col = np.unravel_index(best_match_index.item(), similarity_matrix.shape)
            # print((row, col))  # Output: (1, 0)
            # print(best_match_index)
            # Get the corresponding predicted keyword
            
            # print("Len GT keywords:", len(gt_keywords), "Len Pred keywords:", len(pred_keywords))
            result = {
                "gt_keyword": gt_keywords[row],
                "gt_bbox": gt_bboxes[row],
                "best_pred_keyword": pred_keywords[col],
                "best_pred_bbox": pred_bboxes[col],
                "score": best_match_score.item()
            }
            results.append(result)
            
            # Remove the matched row and column to avoid re-matching
            similarity_matrix = torch.cat((similarity_matrix[:row], similarity_matrix[row+1:]), dim=0)  # Remove row
            similarity_matrix = torch.cat((similarity_matrix[:, :col], similarity_matrix[:, col+1:]), dim=1)  # Remove column
            gt_keywords.pop(row)
            pred_keywords.pop(col)
            # print(f"GT Keyword: '{gt_keyword}'")
            # print(f" -> Best Match: '{best_pred_keyword}' (Score: {best_match_score:.4f})\n")
            
        for remaining_gt in gt_keywords:
            results.append({
                "gt_keyword": remaining_gt,
                "gt_bbox": (None, None),
                "best_pred_keyword": None,
                "best_pred_bbox": (None, None),
                "score": 0.0
            })
            
        for remaining_pred in pred_keywords:
            results.append({
                "gt_keyword": None,
                "gt_bbox": (None, None),
                "best_pred_keyword": remaining_pred,
                "best_pred_bbox": (None, None),
                "score": 0.0
            })
        
        return results
        
    def evaluate(self, prediction_text: str, ground_truth_text: str):
        """
        Performs semantic matching between keywords from prediction and ground truth texts and calculates IoU and matching scores.
        """
        pred_keywords, pred_bboxes = self.extract_keywords(prediction_text)
        gt_keywords, gt_bboxes = self.extract_keywords(ground_truth_text)

        # print(f"Predicted Keywords: {pred_keywords}")
        # print(f"Predicted BBoxes: {pred_bboxes}")
        # print(f"Ground Truth Keywords: {gt_keywords}")
        # print(f"Ground Truth BBoxes: {gt_bboxes}\n")

        results = self.match_keywords(gt_keywords, pred_keywords, gt_bboxes, pred_bboxes)
        
        #calculate the IOU for each matched pair
        for res in results:
            gt_bbox, _ = res["gt_bbox"]
            pred_bbox, _ = res["best_pred_bbox"]
            if gt_bbox is None or pred_bbox is None:
                res["iou"] = 0.0
                res["grounding_iou"] = 0.0
                res['matched'] = False
                continue
            # Calculate intersection
            xA = max(gt_bbox[0], pred_bbox[0])
            yA = max(gt_bbox[1], pred_bbox[1])
            xB = min(gt_bbox[2], pred_bbox[2])
            yB = min(gt_bbox[3], pred_bbox[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            # Calculate union
            boxAArea = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            boxBArea = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
            unionArea = boxAArea + boxBArea - interArea
            # Compute IoU
            iou = interArea / unionArea if unionArea > 0 else 0
            res["iou"] = iou
            # Compute grounding IoU (IoU weighted by semantic score)
            res["grounding_iou"] = iou * res["score"]
            res['matched'] = True
        return results

    def evaluate_dataset(self, prediction_texts: list[str], ground_truth_texts: list[str]):
        """
        Evaluates the whole dataset by processing lists of prediction and ground truth texts.
        
        Args:
            prediction_texts: List of prediction text strings
            ground_truth_texts: List of ground truth text strings
            
        Returns:
            dict: Dictionary containing overall metrics and detailed results
        """
        if len(prediction_texts) != len(ground_truth_texts):
            raise ValueError("Prediction and ground truth lists must have the same length")
        
        all_results = []
        total_samples = len(prediction_texts)
        total_samples_with_gt_keywords = 0
        total_gt_keywords = 0
        total_pred_keywords = 0
        total_matched_keywords = 0
        total_iou_at_05 = 0  # Count of predictions with IoU >= 0.5
        print("Starting grounding evaluation...")
        print(f"Evaluating {total_samples} samples...")
        num_samples_with_matches = 0
        
        for i, (pred_text, gt_text) in enumerate(zip(prediction_texts, ground_truth_texts)):
            if i % 100 == 0:  # Progress indicator
                print(f"Processing sample {i+1}/{total_samples}")
            
            # Extract keywords directly to get accurate counts
            pred_keywords, pred_bboxes = self.extract_keywords(pred_text)
            gt_keywords, gt_bboxes = self.extract_keywords(gt_text)
            
            # Count actual keywords extracted
            gt_keywords_count = len(gt_keywords)
            pred_keywords_count = len(pred_keywords)
            
            if gt_keywords_count > 0:
                total_samples_with_gt_keywords += 1

            if pred_keywords_count == 0 or gt_keywords_count == 0:
                all_results.append({
                    "sample_index": 0,
                    "gt_keywords_count": gt_keywords_count,
                    "pred_keywords_count": pred_keywords_count,
                    "matched_keywords_count": 0,
                    "iou_at_05_count": 0,
                    "sample_iou": 0.0,
                    "sample_semantic_score": 0.0,
                    "sample_grounding_iou": 0.0,
                    "results": []
                })
                continue
                    
            # Evaluate individual sample
            sample_results = self.evaluate(pred_text, gt_text)


            # Count matches for this sample (only from valid results)
            matched_keywords_count = len([r for r in sample_results if r["matched"]])
            
            if matched_keywords_count > 0:
                num_samples_with_matches += 1
            
            
            # Count IoU@0.5 matches for this sample
            iou_at_05_count = len([r for r in sample_results if r["iou"] >= 0.5])
            
            # Accumulate totals
            total_gt_keywords += gt_keywords_count
            total_pred_keywords += pred_keywords_count
            total_matched_keywords += matched_keywords_count
            total_iou_at_05 += iou_at_05_count

            sample_iou = np.mean([r["iou"] for r in sample_results if r["matched"]]) if any(r["matched"] > 0 for r in sample_results) else 0.0
            sample_semantic = np.mean([r["score"] for r in sample_results if r["matched"]]) if any(r["matched"] > 0 for r in sample_results) else 0.0
            sample_grounding_iou = np.mean([r["grounding_iou"] for r in sample_results])

            all_results.append({
                "sample_index": i,
                "gt_keywords_count": gt_keywords_count,
                "pred_keywords_count": pred_keywords_count,
                "matched_keywords_count": matched_keywords_count,
                "iou_at_05_count": iou_at_05_count,
                "sample_iou": sample_iou,
                "sample_semantic_score": sample_semantic,
                "sample_grounding_iou": sample_grounding_iou,
                "results": sample_results
            })
        
        # Calculate overall metrics
        precision = total_matched_keywords / total_pred_keywords if total_pred_keywords > 0 else 0.0
        recall = total_matched_keywords / total_gt_keywords if total_gt_keywords > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # IoU@0.5 metrics
        precision_at_05 = total_iou_at_05 / total_pred_keywords if total_pred_keywords > 0 else 0.0
        recall_at_05 = total_iou_at_05 / total_gt_keywords if total_gt_keywords > 0 else 0.0
        f1_at_05 = 2 * (precision_at_05 * recall_at_05) / (precision_at_05 + recall_at_05) if (precision_at_05 + recall_at_05) > 0 else 0.0

        if len(all_results) == 0:
            print("No valid samples with keywords found for evaluation.")
            return {
                "overall_metrics": {},
                "detailed_results": []
            }
            
        total_iou_score = sum(r["sample_iou"] for r in all_results if r["matched_keywords_count"] > 0)
        total_semantic_score = sum(r["sample_semantic_score"] for r in all_results if r["matched_keywords_count"] > 0)
        total_grounding_iou_score = sum(r["sample_grounding_iou"] for r in all_results if r["gt_keywords_count"] > 0)

        avg_iou = total_iou_score / num_samples_with_matches if num_samples_with_matches > 0 else 0.0
        avg_semantic_score = total_semantic_score / num_samples_with_matches if num_samples_with_matches > 0 else 0.0
        avg_grounding_iou = total_grounding_iou_score / total_samples_with_gt_keywords if total_samples_with_gt_keywords > 0 else 0.0 

        overall_metrics = {
            "total_samples": total_samples,
            "total_samples_with_gt_keywords": total_samples_with_gt_keywords,
            "total_gt_keywords": total_gt_keywords,
            "total_pred_keywords": total_pred_keywords,
            "total_matched_keywords": total_matched_keywords,
            "total_iou_at_05": total_iou_at_05,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "precision_at_05": precision_at_05,
            "recall_at_05": recall_at_05,
            "f1_at_05": f1_at_05,
            "average_iou": avg_iou,
            "average_semantic_score": avg_semantic_score,
            "average_grounding_iou": avg_grounding_iou,
        }
        
        print("\n--- Grounding Evaluation Results ---")
        print(f"Total samples: {total_samples}")
        print(f"Total GT keywords: {total_gt_keywords}")
        print(f"Total predicted keywords: {total_pred_keywords}")
        print(f"Total matched keywords: {total_matched_keywords}")
        print(f"Total IoU@0.5 correct: {total_iou_at_05}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"Precision@0.5: {precision_at_05:.4f}")
        print(f"Recall@0.5: {recall_at_05:.4f}")
        print(f"F1@0.5: {f1_at_05:.4f}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Semantic Score: {avg_semantic_score:.4f}")
        print(f"Average Grounding IoU: {avg_grounding_iou:.4f}")
        
        return {
            "overall_metrics": overall_metrics,
            "detailed_results": all_results
        }
        
    def eval_keyword_metrics(self, predictions, ground_truth_texts):
        
        closed_samples = []
        open_samples = []

        for pred, suf in zip(predictions, ground_truth_texts):
            if any(x in suf.lower() for x in ['yes', 'no', 'ja', 'nein']):
                closed_samples.append((pred, suf))
            else:
                open_samples.append((pred, suf))

        f1_scores = []
        for i, (pred_text, gt_text) in enumerate(tqdm.tqdm(open_samples, desc="Processing open samples")):
            # Extract keywords directly to get accurate counts
            pred_keywords, pred_bboxes = self.extract_keywords(pred_text)
            gt_keywords, gt_bboxes = self.extract_keywords(gt_text)
            # print(f"Sample {i+1}:")
            # print(f"Prediction Text: {pred_text}")
            # print(f"Ground Truth Text: {gt_text}")
            # print(f"Predicted Keywords: {pred_keywords}")
            # print(f"Ground Truth Keywords: {gt_keywords}\n")

            gt_keywords = [k_word.replace(" ", "_").strip() for k_word in gt_keywords if len(k_word.strip()) > 0]
            pred_keywords = [k_word.replace(" ", "_").strip() for k_word in pred_keywords if len(k_word.strip()) > 0]
            
            # if len(gt_keywords) == 0 and len(pred_keywords) == 0:
            #     # f1_scores.append(1.0)
            #     continue
            
            gt_keywords = " ".join(gt_keywords)
            pred_keywords = " ".join(pred_keywords)
            f1 = compare_f1(gt_keywords, pred_keywords)
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        print(f"Open keyword F1 Score: {mean_f1:.4f}")

        binary_pred = []
        binary_gt = []
        for i, (pred_text, gt_text) in enumerate(tqdm.tqdm(closed_samples, desc="Processing closed samples")):
            if 'yes' in gt_text.lower() or 'ja' in gt_text.lower():
                binary_gt.append('yes')
            elif 'no' in gt_text.lower() or 'nein' in gt_text.lower():
                binary_gt.append('no')
            else:
                binary_gt.append('Undefined') 

            if 'yes' in str(pred_text).lower() or 'ja' in str(pred_text).lower():
                binary_pred.append('yes')
            elif 'no' in str(pred_text).lower() or 'nein' in str(pred_text).lower():
                binary_pred.append('no')
            else:
                binary_pred.append('None')  
                
        acc_metric = get_metric_is_equal_default_prep()
        acc_metric.reset() 
        acc_metric.update(binary_pred, binary_gt) 
        scores = acc_metric.compute_per_datapoint()
        mean_acc = torch.mean(scores).item()
        print(f"Closed Accuracy: {mean_acc:.4f}")
        results = {
            "mean_open_keyword_f1": mean_f1,
            "mean_closed_accuracy": mean_acc
        }
        return results

if __name__ == "__main__":
    # --- 1. Single Sample Evaluation ---
    prediction_text = "In the early arterial contrast phase, a small <p bbox=<loc0096><loc0093><loc0333><loc0422> id=<seg005>>segment 3</p> of 5 mm is seen subcapsular in <p bbox=<loc0089><loc0093><loc0333><loc0425> id=<seg005>>liver segment II</p>."
    ground_truth_text = "<p bbox=<loc0089><loc0096><loc0333><loc0422> id=<seg005>>Liver</p> hemangioma <p bbox=<loc0089><loc0096><loc0333><loc0422> id=<seg005>>segment 7</p> and <p bbox=<loc0089><loc0096><loc0333><loc0422> id=<seg005>>segment 2</p>."

    evaluator = DetectGroundingEvaluator()
    matching_results = evaluator.evaluate(prediction_text, ground_truth_text)

    if matching_results:
        print("\n--- Single Sample Results ---")
        for res in matching_results:
            print(f"GT: '{res['gt_keyword']}' -> Pred: '{res['best_pred_keyword']}' (Score: {res['score']:.4f}), IoU: {res['iou']:.4f}, Grounding IoU: {res['grounding_iou']:.4f}")

    # --- 2. Dataset Evaluation Example ---
    print("\n" + "="*50)
    print("DATASET EVALUATION EXAMPLE")
    print("="*50)
    
    # Example dataset (you would replace this with your actual data)
    prediction_texts = [
        "In the early arterial contrast phase, a small <p bbox=<loc0096><loc0093><loc0333><loc0422> id=<seg005>>segment 3</p> of 5 mm is seen subcapsular in <p bbox=<loc0089><loc0093><loc0333><loc0425> id=<seg005>>liver segment II</p>.",
        "The <p bbox=<loc0100><loc0100><loc0300><loc0300> id=<seg001>>kidney</p> shows normal appearance with <p bbox=<loc0150><loc0150><loc0250><loc0250> id=<seg002>>cortex</p> enhancement.",
        "A <p bbox=<loc0200><loc0200><loc0400><loc0400> id=<seg003>>lung nodule</p> is visible in the <p bbox=<loc0180><loc0180><loc0420><loc0420> id=<seg004>>right lower lobe</p>, <p bbox=<loc0180><loc0180><loc0420><loc0420> id=<seg004>>left lower lobe</p>, <p bbox=<loc0180><loc0180><loc0420><loc0420> id=<seg004>>right upper lobe</p>.",
        "No abnormal findings detected in the chest region.",  # No keywords in prediction
        "The patient shows normal <p bbox=<loc0300><loc0300><loc0400><loc0400> id=<seg006>>heart</p> function.",  # Has keywords in prediction
        "A small <p bbox=<loc0050><loc0050><loc0150><loc0150> id=<seg007>>liver lesion</p> is noted.",  # Poor overlap case - bbox far from ground truth (IoU = 0)
        "A <p bbox=<loc0100><loc0100><loc0300><loc0300> id=<seg008>>spleen cyst</p> is visible."  # Moderate overlap case - IoU ≈ 0.25
    ]
    
    ground_truth_texts = [
        "<p bbox=<loc0089><loc0096><loc0333><loc0422> id=<seg005>>Liver</p> hemangioma <p bbox=<loc0089><loc0096><loc0333><loc0422> id=<seg005>>segment 7</p> and <p bbox=<loc0089><loc0096><loc0333><loc0422> id=<seg005>>segment 2</p>.",
        "Normal <p bbox=<loc0105><loc0105><loc0295><loc0295> id=<seg001>>kidney</p> with enhanced <p bbox=<loc0155><loc0155><loc0245><loc0245> id=<seg002>>cortex</p>.",
        "Small <p bbox=<loc0195><loc0195><loc0405><loc0405> id=<seg003>>pulmonary nodule</p> in <p bbox=<loc0175><loc0175><loc0425><loc0425> id=<seg004>>right lower lobe</p>.",
        "Normal chest examination with no abnormalities.",  # No keywords in ground truth
        "The examination was unremarkable.",  # No keywords in ground truth, but prediction has keywords
        "A <p bbox=<loc0400><loc0400><loc0500><loc0500> id=<seg007>>liver mass</p> is present.",  # Poor overlap case - bbox far from prediction (IoU = 0)
        "A <p bbox=<loc0200><loc0200><loc0400><loc0400> id=<seg008>>splenic lesion</p> detected."  # Moderate overlap case - IoU ≈ 0.25 (partial overlap with prediction)
    ]
    
    # Evaluate the dataset
    dataset_results = evaluator.evaluate_dataset(prediction_texts, ground_truth_texts)
    
    # Access the results
    overall_metrics = dataset_results["overall_metrics"]
    detailed_results = dataset_results["detailed_results"]
    
    for i, sample in enumerate(detailed_results):
        print(f"GT Keywords: {sample['gt_keywords_count']}, Pred Keywords: {sample['pred_keywords_count']}, Matched: {sample['matched_keywords_count']}, IoU@0.5 Correct: {sample['iou_at_05_count']}")
        for res in sample["results"]:
            print(f"  GT: '{res['gt_keyword']}' -> Pred: '{res['best_pred_keyword']}' (Score: {res['score']:.4f}), IoU: {res['iou']:.4f}, Grounding IoU: {res['grounding_iou']:.4f}")