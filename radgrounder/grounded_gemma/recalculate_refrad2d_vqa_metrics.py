import numpy as np

def calculate_refrad2dvqa_open_closed(metrics_dict, infos):
    """
    Evaluate RefRad2DVQA results separately for open-ended and closed-ended questions.

    Args:
        metrics_dict (dict): Dictionary containing overall evaluation metrics.
        infos (list): List of dictionaries containing information about each question, including 'question_type'.
    """

    llm_scores = {"open":[], "closed":[]}
    f1_scores = {"open":[], "closed":[]}
    acc_scores = {"open":[], "closed":[]}
    for idx, info in enumerate(infos):
        q_type = info.get("question_type", "")
        if q_type in ["yesno", "multiple"]:
            llm_scores["closed"].append(metrics_dict["llm_score"][idx])
            f1_scores["closed"].append(metrics_dict["f1"][idx])
            acc_scores["closed"].append(metrics_dict["accuracy"][idx])
        elif q_type == "open":
            llm_scores["open"].append(metrics_dict["llm_score"][idx])
            f1_scores["open"].append(metrics_dict["f1"][idx])
            acc_scores["open"].append(metrics_dict["accuracy"][idx])

    open_closed_scores = {}
    for q_type in ["open", "closed"]:
        mean_f1 = np.mean(f1_scores[q_type]) if f1_scores[q_type] else 0
        mean_f1 = round(mean_f1, 4)
        
        mean_acc = np.mean(acc_scores[q_type]) if acc_scores[q_type] else 0
        mean_acc = round(mean_acc, 4)
        
        mean_llm_score = np.mean(llm_scores[q_type]) if llm_scores[q_type] else 0
        mean_llm_score = (mean_llm_score - 1) / 4  # Normalize LLM score from [1,5] to [0,1]
        mean_llm_score = round(mean_llm_score, 4)

        new_metrics = {
            f'{q_type}_count': len(f1_scores[q_type]),
            f"f1_{q_type}": float(mean_f1),
            f"accuracy_{q_type}": float(mean_acc),
            f"llm_score_{q_type}": float(mean_llm_score),
        }
        open_closed_scores.update(new_metrics)
    return open_closed_scores

#test in main 
if __name__ == "__main__":
    # Example usage
    metrics_dict = {
        "llm_score": [0.9, 0.8, 0.7, 0.6],
        "f1": [0.85, 0.75, 0.65, 0.55],
        "accuracy": [0.9, 0.8, 0.7, 0.6],
    }
    infos = [
        {"question_type": "yesno"},
        {"question_type": "open"},
        {"question_type": "multiple"},
        {"question_type": "open"},
    ]

    results = calculate_refrad2dvqa_open_closed(metrics_dict, infos)
    print(results)