"""Metrics calculation for evaluation."""

from typing import List, Dict, Any
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', '', s)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s.strip()


def exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Calculate exact match score."""
    if not prediction:
        return 0.0

    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if norm_pred == norm_gt:
            return 1.0
    return 0.0


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Calculate F1 score."""
    if not prediction:
        return 0.0

    def get_tokens(s):
        return normalize_answer(s).split()

    pred_tokens = get_tokens(prediction)
    scores = []

    for gt in ground_truths:
        gt_tokens = get_tokens(gt)

        if not pred_tokens and not gt_tokens:
            scores.append(1.0)
            continue
        if not pred_tokens or not gt_tokens:
            scores.append(0.0)
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            scores.append(0.0)
            continue

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        scores.append(f1)

    return max(scores) if scores else 0.0


def calculate_metrics(results: List[Dict[str, Any]], metrics_list: List[str]) -> Dict[str, float]:
    """Calculate all metrics for results."""
    metrics_results = {metric: [] for metric in metrics_list}

    for item in results:
        prediction = item.get('prediction', '')
        ground_truths = item.get('ground_truths', [])

        for metric in metrics_list:
            if metric == 'exact_match':
                score = exact_match(prediction, ground_truths)
            elif metric == 'f1':
                score = f1_score(prediction, ground_truths)
            else:
                continue

            metrics_results[metric].append(score)

    # Calculate averages
    avg_metrics = {}
    for metric, scores in metrics_results.items():
        avg_metrics[metric] = sum(scores) / len(scores) if scores else 0.0

    # Add search statistics
    search_queries = [len(item.get('search_queries', [])) for item in results]
    iterations = [item.get('iterations', 0) for item in results]

    avg_metrics['avg_searches'] = sum(search_queries) / len(search_queries) if search_queries else 0.0
    avg_metrics['avg_iterations'] = sum(iterations) / len(iterations) if iterations else 0.0

    return avg_metrics