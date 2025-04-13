# Import Auxiliary modules
from collections import defaultdict

class TextEvaluatorByCategory:
    def __init__(self, ground_truths, predictions, categories=None):
        """
        :param ground_truths: List of ground truth strings
        :param predictions: List of predicted strings
        :param categories: List of category labels corresponding to each item (optional)
        """
        assert len(ground_truths) == len(predictions), "GT and Prediction length mismatch"
        if categories:
            assert len(categories) == len(predictions), "Categories length mismatch"
        self.gt = ground_truths
        self.pred = predictions
        self.cat = categories
        self.n = len(ground_truths)

    # Compute Jaccard similarity between two strings
    def _jaccard(self, str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c)) if (len(a) + len(b) - len(c)) > 0 else 0.0
        # Return Jaccard score, or 0 if denominator is zero

    # Compute precision, recall, and F1 score given predicted and ground truth strings
    def _precision_recall_f1(self, str1, str2):
        pred_tokens = set(str1.lower().split())
        truth_tokens = set(str2.lower().split())
        tp = len(pred_tokens & truth_tokens)
        precision = tp / len(pred_tokens) if pred_tokens else 0
        recall = tp / len(truth_tokens) if truth_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        return precision, recall, f1

    # Main evaluation function: calculates metrics per category
    def evaluate(self):
        # Dictionary to accumulate metric totals per category
        total = defaultdict(lambda: {
            "jaccard": 0, "precision": 0, "recall": 0, "f1": 0, "correct": 0, "count": 0
        })
        # Loop through each prediction and ground truth
        for i in range(self.n):
            gt_i = self.gt[i]   # Ground truth
            pred_i = self.pred[i]   # Predicted
            cat = self.cat[i] if self.cat else "ALL"
            # Calculate metrics for this pair
            jaccard = self._jaccard(gt_i, pred_i)
            precision, recall, f1 = self._precision_recall_f1(gt_i, pred_i)
            is_correct = int(gt_i.strip().lower() == pred_i.strip().lower())

            total[cat]["jaccard"] += jaccard
            total[cat]["precision"] += precision
            total[cat]["recall"] += recall
            total[cat]["f1"] += f1
            total[cat]["correct"] += is_correct
            total[cat]["count"] += 1
        # Compute average metrics per category
        results = {}
        for cat in total:
            count = total[cat]["count"]
            results[cat] = {
                "Accuracy": total[cat]["correct"] / count,
                "Jaccard": total[cat]["jaccard"] / count,
                "Precision": total[cat]["precision"] / count,
                "Recall": total[cat]["recall"] / count,
                "F1 Score": total[cat]["f1"] / count
            }

        return results
        # return metrics dictionary by category

