"""
评估指标（分类 + 掌握度）
"""
import torch
import numpy as np
import logging
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, balanced_accuracy_score,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:

    @staticmethod
    def _to_ac_probs(logits):
        logits = logits.detach().cpu()
        return torch.sigmoid(logits).numpy().flatten()

    @staticmethod
    def auc(logits, targets):
        try:
            probs = MetricsCalculator._to_ac_probs(logits)
            tgts = targets.detach().cpu().numpy().flatten()
            if len(np.unique(tgts)) < 2:
                return float('nan')
            return roc_auc_score(tgts, probs)
        except Exception as e:
            logger.warning(f"AUC 计算失败: {e}")
            return float('nan')

    @staticmethod
    def binary_accuracy(logits, targets, threshold=0.5):
        probs = MetricsCalculator._to_ac_probs(logits)
        preds = (probs >= threshold).astype(float)
        tgts = targets.detach().cpu().numpy().flatten()
        return accuracy_score(tgts, preds)

    @staticmethod
    def f1(logits, targets, threshold=0.5):
        probs = MetricsCalculator._to_ac_probs(logits)
        preds = (probs >= threshold).astype(float)
        tgts = targets.detach().cpu().numpy().flatten()
        return f1_score(tgts, preds, zero_division=0)

    @staticmethod
    def precision_recall(logits, targets, threshold=0.5):
        probs = MetricsCalculator._to_ac_probs(logits)
        preds = (probs >= threshold).astype(float)
        tgts = targets.detach().cpu().numpy().flatten()
        p = precision_score(tgts, preds, zero_division=0)
        r = recall_score(tgts, preds, zero_division=0)
        return p, r

    @staticmethod
    def mastery_mae(mastery_pred, mastery_target):
        pred = mastery_pred.detach().cpu().numpy()
        tgt = mastery_target.detach().cpu().numpy()
        return float(np.mean(np.abs(pred - tgt)))

    @staticmethod
    def compute_all(ac_logits, ac_targets,
                    mastery_pred=None, mastery_target=None,
                    threshold=0.5):
        probs = MetricsCalculator._to_ac_probs(ac_logits)
        preds = (probs >= threshold).astype(float)
        tgts = ac_targets.detach().cpu().numpy().flatten()

        metrics = {
            'auc': MetricsCalculator.auc(ac_logits, ac_targets),
            'accuracy': MetricsCalculator.binary_accuracy(ac_logits, ac_targets, threshold),
            'balanced_accuracy': float(balanced_accuracy_score(tgts, preds)),
            'f1': MetricsCalculator.f1(ac_logits, ac_targets, threshold),
        }
        p, r = MetricsCalculator.precision_recall(ac_logits, ac_targets, threshold)
        metrics['precision'] = p
        metrics['recall'] = r
        metrics['num_pos'] = int(tgts.sum())
        metrics['num_neg'] = int(len(tgts) - tgts.sum())

        if mastery_pred is not None and mastery_target is not None:
            metrics['mastery_mae'] = MetricsCalculator.mastery_mae(mastery_pred, mastery_target)

        return metrics

    @staticmethod
    def format_metrics(metrics, prefix=""):
        lines = [f"{prefix} Metrics:"]
        lines.append(f"  AUC:          {metrics['auc']:.4f}")
        lines.append(f"  Accuracy:     {metrics['accuracy']:.4f}")
        lines.append(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        lines.append(f"  F1:           {metrics['f1']:.4f}")
        lines.append(f"  Precision:    {metrics['precision']:.4f}")
        lines.append(f"  Recall:       {metrics['recall']:.4f}")
        if 'mastery_mae' in metrics:
            lines.append(f"  Mastery MAE:  {metrics['mastery_mae']:.4f}")
        return "\n".join(lines)
