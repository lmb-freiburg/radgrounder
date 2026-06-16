import re
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from ovqa.metrics.simple import compare_f1, get_metric_is_equal_default_prep

class SegmentGroundingEvaluator:
    def __init__(self, model_name="google/embeddinggemma-300m", cache_folder="./models", 
                 seg_start_token="<seg>", seg_end_token="</seg>", iou_threshold=0.5):
        """
        Initializes the SegmentGroundingEvaluator for evaluating binary segmentation map groundings.
        
        Args:
            model_name: Name of the sentence transformer model for text similarity
            cache_folder: Cache folder for the model
            seg_start_token: Start token for segmentation (default: "<seg>")
            seg_end_token: End token for segmentation (default: "</seg>")
            iou_threshold: IoU threshold for considering a segmentation mask match as correct
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SegmentGroundingEvaluator - Using device: {self.device}")
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        self.model.to(self.device)
        self.seg_start_token = seg_start_token
        self.seg_end_token = seg_end_token
        self.iou_threshold = iou_threshold
        
        # Create regex pattern to extract segmented keywords
        escaped_start = re.escape(seg_start_token)
        escaped_end = re.escape(seg_end_token)
        self.segment_pattern = re.compile(rf"{escaped_start}(.*?){escaped_end}", re.IGNORECASE)

    def extract_segmented_keywords(self, text: str) -> list[str]:
        """
        Extracts keywords from text wrapped in segmentation tokens.
        Example: "The <seg>liver</seg> shows <seg>enhancement</seg>" -> ["liver", "enhancement"]
        
        Args:
            text: Input text containing segmented keywords
            
        Returns:
            List of extracted keywords (lowercase, stripped)
        """
        matches = self.segment_pattern.findall(text)
        keywords = [match.strip().lower() for match in matches if match.strip()]
        return keywords

    def compute_mask_iou(self, mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        """
        Compute Intersection over Union (IoU) between two binary masks.
        
        Args:
            mask1: First binary mask tensor (H, W)
            mask2: Second binary mask tensor (H, W)
            
        Returns:
            IoU value as float
        """
        if mask1.shape != mask2.shape:
            raise ValueError(f"Mask shapes don't match: {mask1.shape} vs {mask2.shape}")
        
        # Ensure masks are binary
        mask1 = (mask1 > 0.5).float()
        mask2 = (mask2 > 0.5).float()
        
        intersection = (mask1 * mask2).sum()
        union = (mask1 + mask2 - mask1 * mask2).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return (intersection / union).item()

    def match_keywords_and_masks(self, gt_keywords, pred_keywords, gt_masks, pred_masks):
        """
        Match ground truth and predicted keywords with their corresponding masks.
        Keywords and masks are assumed to be in the same order.
        
        Args:
            gt_keywords: List of ground truth keywords
            pred_keywords: List of predicted keywords
            gt_masks: Ground truth binary masks list (N_gt, H, W)
            pred_masks: Predicted binary masks list (N_pred, H, W)

        Returns:
            Dictionary containing matching results and metrics
        """
        if len(gt_keywords) == 0 and len(pred_keywords) == 0:
            return {
                "matches": [],
                "gt_keywords_count": 0,
                "pred_keywords_count": 0,
                "matched_keywords_count": 0,
                "text_similarity_scores": [],
                "mask_ious": [],
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "mean_iou": 1.0,
            }
            
        if len(gt_keywords) == 0:
            return {
                "matches": [],
                "gt_keywords_count": 0,
                "pred_keywords_count": len(pred_keywords),
                "matched_keywords_count": 0,
                "text_similarity_scores": [],
                "mask_ious": [],
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "mean_iou": 0.0,
            }
            
        if len(pred_keywords) == 0:
            return {
                "matches": [],
                "gt_keywords_count": len(gt_keywords),
                "pred_keywords_count": 0,
                "matched_keywords_count": 0,
                "text_similarity_scores": [],
                "mask_ious": [],
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "mean_iou": 0.0,
            }

        # Ensure we have the right number of masks
        # Ensure we have the right number of masks
        try:
            if len(gt_masks) != len(gt_keywords):
                raise ValueError(f"Number of GT masks ({len(gt_masks)}) doesn't match GT keywords ({len(gt_keywords)})")
            if len(pred_masks) != len(pred_keywords):
                raise ValueError(f"Number of pred masks ({len(pred_masks)}) doesn't match pred keywords ({len(pred_keywords)}) -- GT masks: {len(gt_masks)}, GT keywords: {len(gt_keywords)}")
        except ValueError as e:
            print(f"Warning: {e}")
            # Pad or truncate gt_masks to match gt_keywords
            if len(gt_masks) < len(gt_keywords):
                pad_shape = gt_masks[0].shape if gt_masks else (1, 1)
                for _ in range(len(gt_keywords) - len(gt_masks)):
                    gt_masks.append(torch.zeros(pad_shape, dtype=torch.float32))
            elif len(gt_masks) > len(gt_keywords):
                gt_masks = gt_masks[:len(gt_keywords)]
            # Pad or truncate pred_masks to match pred_keywords
            if len(pred_masks) < len(pred_keywords):
                pad_shape = pred_masks[0].shape if pred_masks else (1, 1)
                for _ in range(len(pred_keywords) - len(pred_masks)):
                    pred_masks.append(torch.zeros(pad_shape, dtype=torch.float32))
            elif len(pred_masks) > len(pred_keywords):
                pred_masks = pred_masks[:len(pred_keywords)]

        # Generate embeddings for text similarity
        gt_embeddings = self.model.encode(gt_keywords, convert_to_tensor=True)
        pred_embeddings = self.model.encode(pred_keywords, convert_to_tensor=True)

        # Ensure embeddings are 2D
        if gt_embeddings.dim() == 1:
            gt_embeddings = gt_embeddings.unsqueeze(0)
        if pred_embeddings.dim() == 1:
            pred_embeddings = pred_embeddings.unsqueeze(0)

        # Compute text similarity matrix for keyword matching
        text_similarity_matrix = torch.cosine_similarity(
            gt_embeddings.unsqueeze(1), pred_embeddings.unsqueeze(0), dim=2
        )
        # Clamp to [0, 1] to handle floating-point precision errors
        text_similarity_matrix = torch.clamp(text_similarity_matrix, 0.0, 1.0)

        # Match keywords using text similarity only (greedy matching)
        matches = []
        used_pred_indices = set()
        used_gt_indices = set()
        matched_keywords_count = 0
        text_similarities = []
        mask_ious = []

        # Greedy matching based on text similarity
        for _ in range(min(len(gt_keywords), len(pred_keywords))):
            # Find the best remaining match based on text similarity
            best_score = -1
            best_gt_idx = -1
            best_pred_idx = -1
            
            for i in range(len(gt_keywords)):
                for j in range(len(pred_keywords)):
                    if j not in used_pred_indices and text_similarity_matrix[i, j] > best_score:
                        best_score = text_similarity_matrix[i, j].item()
                        best_gt_idx = i
                        best_pred_idx = j
            
            if best_gt_idx != -1 and best_pred_idx != -1:
                text_sim = text_similarity_matrix[best_gt_idx, best_pred_idx].item()

                pred_mask = pred_masks[best_pred_idx].to(gt_masks[best_gt_idx].device)
                mask_iou = self.compute_mask_iou(gt_masks[best_gt_idx], pred_mask)

                match_info = {
                    "gt_keyword": gt_keywords[best_gt_idx],
                    "pred_keyword": pred_keywords[best_pred_idx],
                    "gt_idx": best_gt_idx,
                    "pred_idx": best_pred_idx,
                    "text_similarity": text_sim,
                    "mask_iou": mask_iou,
                    "grounding_iou": mask_iou * text_sim,
                    "combined_score": 0.5 * text_sim + 0.5 * mask_iou,
                    "is_correct_mask": mask_iou >= self.iou_threshold
                }
                
                matches.append(match_info)
                used_pred_indices.add(best_pred_idx)
                used_gt_indices.add(best_gt_idx)
                matched_keywords_count += 1
                text_similarities.append(text_sim)
                mask_ious.append(mask_iou)
                
                # Set this pair to -1 to avoid reusing
                text_similarity_matrix[best_gt_idx, :] = -1
                text_similarity_matrix[:, best_pred_idx] = -1

        # Add unmatched GT keywords (penalize missing predictions)
        for i, gt_keyword in enumerate(gt_keywords):
            if i not in used_gt_indices:  # Not matched yet
                match_info = {
                    "gt_keyword": gt_keyword,
                    "pred_keyword": None,
                    "gt_idx": i,
                    "pred_idx": None,
                    "text_similarity": 0.0,
                    "mask_iou": 0.0,
                    "grounding_iou": 0.0,
                    "combined_score": 0.0,
                    "is_correct_mask": False
                }
                matches.append(match_info)
                text_similarities.append(0.0)
                mask_ious.append(0.0)

        # Add unmatched predicted keywords (penalize false positives)
        for j, pred_keyword in enumerate(pred_keywords):
            if j not in used_pred_indices:
                match_info = {
                    "gt_keyword": None,
                    "pred_keyword": pred_keyword,
                    "gt_idx": None,
                    "pred_idx": j,
                    "text_similarity": 0.0,
                    "mask_iou": 0.0,
                    "grounding_iou": 0.0,
                    "combined_score": 0.0,
                    "is_correct_mask": False
                }
                matches.append(match_info)
                text_similarities.append(0.0)
                mask_ious.append(0.0)

        # Calculate metrics
        precision = matched_keywords_count / len(pred_keywords) if len(pred_keywords) > 0 else 1.0
        recall = matched_keywords_count / len(gt_keywords) if len(gt_keywords) > 0 else 1.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "matches": matches,
            "gt_keywords_count": len(gt_keywords),
            "pred_keywords_count": len(pred_keywords),
            "matched_keywords_count": matched_keywords_count,
            "text_similarity_scores": text_similarities,
            "mask_ious": mask_ious,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def evaluate(self, pred_text: str, gt_text: str, pred_masks: torch.Tensor, 
                 gt_masks: torch.Tensor) -> dict:
        """
        Main evaluation function for segmentation grounding.
        
        Args:
            pred_text: Predicted text with segmentation tokens
            gt_text: Ground truth text with segmentation tokens
            pred_masks: Predicted binary segmentation masks (N_pred, H, W)
            gt_masks: Ground truth binary segmentation masks (N_gt, H, W)
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        # Extract keywords from texts
        gt_keywords = self.extract_segmented_keywords(gt_text)
        pred_keywords = self.extract_segmented_keywords(pred_text)
        
        # Match keywords and masks
        results = self.match_keywords_and_masks(
            gt_keywords, pred_keywords, gt_masks, pred_masks
        )
        
        # Add additional evaluation info
        results.update({
            "gt_text": gt_text,
            "pred_text": pred_text,
            "gt_keywords": gt_keywords,
            "pred_keywords": pred_keywords,
            "iou_threshold": self.iou_threshold
        })
        
        return results

    def evaluate_dataset(self, pred_texts: list, gt_texts: list, pred_masks_batch: list, 
                        gt_masks_batch: list, batch_size: int = 32) -> dict:
        """
        Evaluates the whole dataset by processing in batches and aggregating results.
        
        Args:
            pred_texts: List of prediction text strings
            gt_texts: List of ground truth text strings
            pred_masks_batch: List of predicted mask tensors
            gt_masks_batch: List of ground truth mask tensors
            batch_size: Size of batches to process (default: 32)
            
        Returns:
            dict: Dictionary containing overall metrics and detailed results
        """
        if len(pred_texts) != len(gt_texts):
            raise ValueError("Prediction and ground truth lists must have the same length")
        if len(pred_texts) != len(pred_masks_batch):
            raise ValueError("Prediction texts and masks lists must have the same length")
        if len(gt_texts) != len(gt_masks_batch):
            raise ValueError("Ground truth texts and masks lists must have the same length")
        
        total_samples = len(pred_texts)
        batch_results = []  # Local variable to accumulate all results
        
        print("Starting segment grounding evaluation...")
        print(f"Evaluating {total_samples} samples in batches of {batch_size}...")
        
        # Process data in batches
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            print(f"Processing batch {start_idx//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} (samples {start_idx+1}-{end_idx})")
            
            # Extract batch
            batch_pred_texts = pred_texts[start_idx:end_idx]
            batch_gt_texts = gt_texts[start_idx:end_idx]
            batch_pred_masks = pred_masks_batch[start_idx:end_idx]
            batch_gt_masks = gt_masks_batch[start_idx:end_idx]
            
            # Evaluate batch and append to batch_results
            self.evaluate_batch(
                batch_pred_texts, batch_gt_texts, 
                batch_pred_masks, batch_gt_masks, batch_results
            )
            
            # Add sample indices to the newly added results
            batch_start_idx = len(batch_results) - len(batch_pred_texts)
            for i in range(len(batch_pred_texts)):
                batch_results[batch_start_idx + i]["sample_index"] = start_idx + i
        
        # Generate overall report from all batch results
        return self.generate_dataset_report(batch_results)

    def generate_dataset_report(self, all_sample_results: list) -> dict:
        """
        Generate dataset-level report from individual sample results.
        
        Args:
            all_sample_results: List of individual sample evaluation results
            
        Returns:
            dict: Dictionary containing overall metrics and detailed results in detection evaluator format
        """
        total_samples = len(all_sample_results)
        total_gt_keywords = 0
        total_pred_keywords = 0
        total_matched_keywords = 0
        total_iou_score = 0.0
        total_semantic_score = 0.0
        total_grounding_iou_score = 0.0
        total_iou_at_05 = 0  # Count of predictions with IoU >= 0.5
        num_samples_with_matched_keywords = 0
        detailed_results = []
        
        for sample_result in all_sample_results:
            # Count keywords
            gt_keywords_count = sample_result["gt_keywords_count"]
            pred_keywords_count = sample_result["pred_keywords_count"]
            matched_keywords_count = sample_result["matched_keywords_count"]
            
            if matched_keywords_count > 0:
                num_samples_with_matched_keywords += 1
            
            total_gt_keywords += gt_keywords_count
            total_pred_keywords += pred_keywords_count
            total_matched_keywords += matched_keywords_count
            
            # Process matches to convert to detection evaluator format
            sample_matches = sample_result["matches"]
            converted_results = []
            sample_iou_scores = []
            sample_semantic_scores = []
            sample_grounding_iou_scores = []
            iou_at_05_count = 0
            
            for match in sample_matches:
                # Convert to detection evaluator format
                result_entry = {
                    "gt_keyword": match["gt_keyword"],
                    "gt_bbox": (None, None),  # No bbox for segment evaluator
                    "best_pred_keyword": match["pred_keyword"],
                    "score": match["text_similarity"],  # Use text similarity as main score
                    "iou": match["mask_iou"],  # Mask IoU instead of bbox IoU
                    "grounding_iou": match["grounding_iou"]
                }
                converted_results.append(result_entry)
                
                # Collect scores for averaging
                sample_iou_scores.append(match["mask_iou"])
                sample_semantic_scores.append(match["text_similarity"])
                sample_grounding_iou_scores.append(match["grounding_iou"])
                
                # Count IoU@0.5 matches
                if match["mask_iou"] >= 0.5:
                    iou_at_05_count += 1
            
            total_iou_at_05 += iou_at_05_count
            
            # Calculate sample-level averages (only from matched keywords)
            if matched_keywords_count > 0:
                sample_iou = np.sum(sample_iou_scores) / matched_keywords_count
                sample_semantic = np.sum(sample_semantic_scores) / matched_keywords_count
                sample_grounding_iou = np.sum(sample_grounding_iou_scores) / matched_keywords_count
            else:
                sample_iou = 0.0
                sample_semantic = 0.0
                sample_grounding_iou = 0.0
            
            # Handle empty case
            if gt_keywords_count == 0 and pred_keywords_count == 0:
                sample_iou = 1.0
                sample_semantic = 1.0
                sample_grounding_iou = 1.0
            
            total_iou_score += sample_iou
            total_semantic_score += sample_semantic
            total_grounding_iou_score += sample_grounding_iou
            
            # Add to detailed results in detection evaluator format
            detailed_results.append({
                "gt_keywords_count": gt_keywords_count,
                "pred_keywords_count": pred_keywords_count,
                "matched_keywords_count": matched_keywords_count,
                "iou_at_05_count": iou_at_05_count,
                "sample_iou": sample_iou,
                "sample_semantic_score": sample_semantic,
                "sample_grounding_iou": sample_grounding_iou,
                "results": converted_results
            })
        
        # Calculate overall metrics
        precision = total_matched_keywords / total_pred_keywords if total_pred_keywords > 0 else 0.0
        recall = total_matched_keywords / total_gt_keywords if total_gt_keywords > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # IoU@0.5 metrics
        precision_at_05 = total_iou_at_05 / total_pred_keywords if total_pred_keywords > 0 else 0.0
        recall_at_05 = total_iou_at_05 / total_gt_keywords if total_gt_keywords > 0 else 0.0
        f1_at_05 = 2 * (precision_at_05 * recall_at_05) / (precision_at_05 + recall_at_05) if (precision_at_05 + recall_at_05) > 0 else 0.0

        # Average across all samples (including those with no matches, which contribute 0)
        avg_iou = total_iou_score / total_samples if total_samples > 0 else 0.0
        avg_semantic_score = total_semantic_score / total_samples if total_samples > 0 else 0.0
        avg_grounding_iou = total_grounding_iou_score / total_samples if total_samples > 0 else 0.0

        overall_metrics = {
            "total_samples": total_samples,
            "num_samples_with_matched_keywords": num_samples_with_matched_keywords,
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
        
        print("\n--- Segment Grounding Evaluation Results ---")
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
            "detailed_results": detailed_results
        }

    def evaluate_batch(self, pred_texts: list, gt_texts: list, pred_masks_batch: list, 
                      gt_masks_batch: list, batch_results: list) -> None:
        """
        Evaluate a batch of samples and append individual results to the provided list.
        
        Args:
            pred_texts: List of predicted texts
            gt_texts: List of ground truth texts
            pred_masks_batch: List of predicted mask tensors
            gt_masks_batch: List of ground truth mask tensors
            batch_results: List to append individual sample evaluation results to
            
        Returns:
            None (modifies batch_results in place)
        """
        for i in range(len(pred_texts)):
            result = self.evaluate(
                pred_texts[i], gt_texts[i], 
                pred_masks_batch[i], gt_masks_batch[i]
            )
            batch_results.append(result)

    def print_detailed_results(self, results: dict, show_matches=True):
        """
        Print detailed evaluation results.
        
        Args:
            results: Results dictionary from evaluate(), generate_dataset_report() or legacy format
            show_matches: Whether to show individual matches
        """
        if "overall_metrics" in results and "detailed_results" in results:
            # New dataset report format
            metrics = results["overall_metrics"]
            print(f"\n=== Segmentation Grounding Evaluation Results ===")
            print(f"Total samples: {metrics['total_samples']}")
            print(f"Total GT Keywords: {metrics['total_gt_keywords']}")
            print(f"Total Pred Keywords: {metrics['total_pred_keywords']}")
            print(f"Total Matched Keywords: {metrics['total_matched_keywords']}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision@0.5: {metrics['precision_at_05']:.4f}")
            print(f"Recall@0.5: {metrics['recall_at_05']:.4f}")
            print(f"F1@0.5: {metrics['f1_at_05']:.4f}")
            print(f"Average IoU: {metrics['average_iou']:.4f}")
            print(f"Average Semantic Score: {metrics['average_semantic_score']:.4f}")
            print(f"Average Grounding IoU: {metrics['average_grounding_iou']:.4f}")
            
            if show_matches:
                print(f"\n=== Individual Sample Results ===")
                for i, sample_result in enumerate(results["detailed_results"]):
                    print(f"\nSample {i+1}:")
                    print(f"  GT Keywords: {sample_result['gt_keywords_count']}")
                    print(f"  Pred Keywords: {sample_result['pred_keywords_count']}")
                    print(f"  Matched: {sample_result['matched_keywords_count']}")
                    print(f"  IoU@0.5 Count: {sample_result['iou_at_05_count']}")
                    print(f"  Sample IoU: {sample_result['sample_iou']:.3f}")
                    print(f"  Sample Semantic: {sample_result['sample_semantic_score']:.3f}")
                    print(f"  Sample Grounding IoU: {sample_result['sample_grounding_iou']:.3f}")
                    
                    for match in sample_result["results"]:
                        status = "✓" if match["iou"] >= self.iou_threshold else "✗"
                        print(f"    {status} '{match['gt_keyword']}' -> '{match['best_pred_keyword']}' "
                              f"(Score: {match['score']:.3f}, IoU: {match['iou']:.3f}, Grounding: {match['grounding_iou']:.3f})")
        elif "batch_results" in results:
            # Legacy batch results format
            metrics = results["overall_metrics"]
            print(f"\n=== Segmentation Grounding Evaluation Results ===")
            print(f"Samples: {metrics['samples_count']}")
            print(f"Total GT Keywords: {metrics['total_gt_keywords']}")
            print(f"Total Pred Keywords: {metrics['total_pred_keywords']}")
            print(f"Total Matched Keywords: {metrics['total_matched_keywords']}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"Mean Text Similarity: {metrics['mean_text_similarity']:.4f}")
            print(f"Mask Accuracy (IoU>={self.iou_threshold}): {metrics['mask_accuracy']:.4f}")
            
            if show_matches:
                print(f"\n=== Individual Sample Results ===")
                for i, sample_result in enumerate(results["batch_results"]):
                    print(f"\nSample {i+1}:")
                    print(f"  GT: {sample_result['gt_text']}")
                    print(f"  Pred: {sample_result['pred_text']}")
                    print(f"  GT Keywords: {sample_result['gt_keywords']}")
                    print(f"  Pred Keywords: {sample_result['pred_keywords']}")
                    print(f"  Matches: {len(sample_result['matches'])}")
                    
                    for match in sample_result["matches"]:
                        status = "✓" if match["is_correct_mask"] else "✗"
                        print(f"    {status} '{match['gt_keyword']}' -> '{match['pred_keyword']}' "
                              f"(Text: {match['text_similarity']:.3f}, IoU: {match['mask_iou']:.3f})")
        else:
            # Single sample results  
            print(f"\n=== Segmentation Grounding Evaluation Results ===")
            print(f"GT Text: {results['gt_text']}")
            print(f"Pred Text: {results['pred_text']}")
            print(f"GT Keywords: {results['gt_keywords']}")
            print(f"Pred Keywords: {results['pred_keywords']}")
            print(f"Matched Keywords: {results['matched_keywords_count']}/{results['gt_keywords_count']}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1 Score: {results['f1_score']:.4f}")
            print(f"Mean IoU: {results['mean_iou']:.4f}")
            
            if show_matches and results["matches"]:
                print(f"\n=== Matches ===")
                for match in results["matches"]:
                    status = "✓" if match["is_correct_mask"] else "✗"
                    print(f"  {status} '{match['gt_keyword']}' -> '{match['pred_keyword']}' "
                          f"(Text: {match['text_similarity']:.3f}, IoU: {match['mask_iou']:.3f})")


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
                pred_keywords = self.extract_segmented_keywords(pred_text)
                gt_keywords = self.extract_segmented_keywords(gt_text)
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
    """
    Example usage of SegmentGroundingEvaluator class for evaluating binary segmentation map groundings.
    """
    
    import torch
    import numpy as np
    
    def create_sample_data():
        """Create sample data for testing the evaluator."""
        
        # Sample texts with segmentation tokens
        gt_text = "The <seg>liver</seg> shows enhancement and the <seg>spleen</seg> appears normal."
        pred_text = "The <seg>liver</seg> is enhanced and <seg>spleen</seg> looks healthy."
        
        # Create sample binary masks (32x32 for simplicity)
        height, width = 32, 32
        
        # Ground truth masks - 2 masks for "liver" and "spleen"
        gt_masks = torch.zeros((2, height, width))
        gt_masks[0, 8:16, 8:16] = 1.0  # liver mask
        gt_masks[1, 20:28, 20:28] = 1.0  # spleen mask
        
        # Predicted masks - 2 masks with slight variations
        pred_masks = torch.zeros((2, height, width))
        pred_masks[0, 9:17, 9:17] = 1.0  # liver mask (slightly shifted)
        pred_masks[1, 19:27, 19:27] = 1.0  # spleen mask (slightly shifted)
        
        return gt_text, pred_text, gt_masks, pred_masks
    
    print("=== SegmentGroundingEvaluator Example ===\n")
    
    # Initialize the evaluator
    evaluator = SegmentGroundingEvaluator(
        model_name="google/embeddinggemma-300m",
        cache_folder="./models",
        iou_threshold=0.5
    )


    def create_square_mask(size: int, top: int, left: int, width: int) -> torch.Tensor:
        mask = torch.zeros((1, size, size))
        mask[0, top:top + width, left:left + width] = 1.0
        return mask

    def stack_masks(masks: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(masks, dim=0)

    def run_miou_edge_case_tests() -> None:
        print("\n=== mIoU Edge Case Tests ===")
        size = 8
        width = 3

        base_gt_text = "Findings show <seg>liver</seg> and <seg>spleen</seg> involvement."
        base_pred_text = "Report describes <seg>liver</seg> with change and <seg>spleen</seg> involvement."

        gt_masks_base = stack_masks([
            create_square_mask(size, 0, 0, width),
            create_square_mask(size, 5, 5, width),
        ])
        pred_masks_base = stack_masks([
            create_square_mask(size, 0, 1, width),  # horizontal shift for 0.5 IoU
            create_square_mask(size, 4, 5, width),  # vertical shift for 0.5 IoU
        ])

        base_result = evaluator.evaluate(base_pred_text, base_gt_text, pred_masks_base, gt_masks_base)
        base_mean_iou = float(np.mean(base_result["mask_ious"]))
        print(f"Baseline mean IoU (two keywords, perfect matching except 0.5 overlap): {base_mean_iou:.2f}")
        print("Mask IoUs for base case:", base_result["mask_ious"])
        print("Matched Keywords:", base_result["matched_keywords_count"])
        print("="*50)

        extra_pred_text = base_pred_text + " Additional <seg>kidney</seg> changes are suspected."
        extra_pred_masks = stack_masks([
            pred_masks_base,
            create_square_mask(size, 2, 2, 2),  # extra keyword mask with no GT partner
        ])

        extra_result = evaluator.evaluate(extra_pred_text, base_gt_text, extra_pred_masks, gt_masks_base)
        extra_mean_iou = float(np.mean(extra_result["mask_ious"]))
        print(
            "Mean IoU with extra prediction (expected drop due to unmatched keyword): "
            f"{extra_mean_iou:.2f}"
        )
        print("Mask IoUs for extra case:", extra_result["mask_ious"])
        print("Matched Keywords:", extra_result["matched_keywords_count"])
        print("="*50)

        extended_gt_text = base_gt_text + " The <seg>kidney</seg> also shows involvement."
        extended_gt_masks = stack_masks([
            gt_masks_base,
            create_square_mask(size, 2, 2, 2),
        ])

        missing_pred_result = evaluator.evaluate(base_pred_text, extended_gt_text, pred_masks_base, extended_gt_masks)
        missing_mean_iou = float(np.mean(missing_pred_result["mask_ious"]))
        print(
            "Mean IoU with missing prediction (expected drop due to unmatched GT keyword): "
            f"{missing_mean_iou:.2f}"
        )
        print("Mask IoUs for missing case:", missing_pred_result["mask_ious"])
        print("Matched Keywords:", missing_pred_result["matched_keywords_count"])
        mIoU =  np.sum(missing_pred_result["mask_ious"]) / missing_pred_result["matched_keywords_count"]
        print("mIoU:", mIoU)
        
        mIoU = np.sum([m["mask_iou"] for m in missing_pred_result["matches"]]) / missing_pred_result["matched_keywords_count"]
        print("mIoU:", mIoU)
        
        print("="*50)

    run_miou_edge_case_tests()

    def run_dataset_evaluation_tests() -> None:
        print("\n=== evaluate_dataset Demo ===")

        size = 8
        width = 3

        # Sample 1: good alignment between prediction and ground truth
        sample1_gt_text = "Findings show <seg>liver</seg> and <seg>spleen</seg> involvement."
        sample1_pred_text = "Report describes <seg>liver</seg> with change and <seg>spleen</seg> involvement."
        sample1_gt_masks = stack_masks([
            create_square_mask(size, 0, 0, width),
            create_square_mask(size, 5, 5, width),
        ])
        sample1_pred_masks = stack_masks([
            create_square_mask(size, 0, 1, width),
            create_square_mask(size, 4, 5, width),
        ])

        # Sample 2: missing predicted keyword to exercise recall aggregation
        sample2_gt_text = sample1_gt_text + " The <seg>kidney</seg> also shows involvement."
        sample2_pred_text = sample1_pred_text
        sample2_gt_masks = stack_masks([
            create_square_mask(size, 0, 0, width),
            create_square_mask(size, 5, 5, width),
            create_square_mask(size, 2, 2, 2),
        ])
        sample2_pred_masks = sample1_pred_masks

        pred_texts = [sample1_pred_text, sample2_pred_text]
        gt_texts = [sample1_gt_text, sample2_gt_text]
        pred_masks_batch = [sample1_pred_masks, sample2_pred_masks]
        gt_masks_batch = [sample1_gt_masks, sample2_gt_masks]

        dataset_results = evaluator.evaluate_dataset(
            pred_texts,
            gt_texts,
            pred_masks_batch,
            gt_masks_batch,
            batch_size=1,
        )

        metrics = dataset_results["overall_metrics"]
        print("Total samples:", metrics["total_samples"])
        print("Total matched keywords:", metrics["total_matched_keywords"])

        for idx, sample in enumerate(dataset_results["detailed_results"], start=1):
            print(f"Sample {idx} IoU: {sample['sample_iou']:.3f}")
            print(f"Sample {idx} G-IoU: {sample['sample_grounding_iou']:.3f}")

    run_dataset_evaluation_tests()