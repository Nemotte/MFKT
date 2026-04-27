"""
共享的训练-评估流程与通用工具函数

包含 set_seed / k_fold_split / train_one_fold / print_comparison_table。
"""
import copy
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import StudentTimelineDataset, collate_fn
from models.model import OJKnowledgeTracingModel
from utils.trainer import Trainer
from utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


def set_seed(seed):
    """设置全局随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def k_fold_split(student_data, k, seed=42):
    """按学生 K 折划分"""
    rng = random.Random(seed)
    keys = list(student_data.keys())
    rng.shuffle(keys)

    fold_size = len(keys) // k
    folds = []
    for i in range(k):
        if i < k - 1:
            val_keys = keys[i * fold_size: (i + 1) * fold_size]
        else:
            val_keys = keys[i * fold_size:]
        train_keys = [key for key in keys if key not in set(val_keys)]
        train_data = {key: student_data[key] for key in train_keys}
        val_data = {key: student_data[key] for key in val_keys}
        folds.append((train_data, val_data))
    return folds


def train_one_fold(fold_idx, train_tl, val_tl, model_name, model_info,
                   preprocessor, config, num_kp, num_problems, init_q_matrix):
    """训练单个 fold 并返回验证指标

    适用于 baseline 和 Ours 模型的统一训练流程。
    """
    cfg = copy.deepcopy(config)
    for k, v in model_info.get('config_override', {}).items():
        setattr(cfg, k, v)
    cfg.validate()

    problem_difficulty = preprocessor.compute_problem_difficulty(train_tl)
    train_ds = StudentTimelineDataset(train_tl, preprocessor, cfg, problem_difficulty=problem_difficulty)
    val_ds = StudentTimelineDataset(val_tl, preprocessor, cfg, problem_difficulty=problem_difficulty)

    pos, neg = train_ds.get_class_distribution()
    pos_weight = train_ds.get_pos_weight()
    logger.info(f"    样本: 训练={len(train_ds)}, 验证={len(val_ds)}, "
                f"正/负={pos}/{neg}, pos_weight={pos_weight:.4f}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn,
        num_workers=getattr(cfg, 'NUM_WORKERS', 0),
        pin_memory=getattr(cfg, 'PIN_MEMORY', False),
        persistent_workers=getattr(cfg, 'NUM_WORKERS', 0) > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn,
        num_workers=getattr(cfg, 'NUM_WORKERS', 0),
        pin_memory=getattr(cfg, 'PIN_MEMORY', False),
        persistent_workers=getattr(cfg, 'NUM_WORKERS', 0) > 0,
    )

    if model_info.get('type') == 'baseline':
        baseline_cls = model_info['class']
        model = baseline_cls(num_problems=num_problems, num_kp=num_kp,
                             **model_info.get('params', {}))
    else:
        model = OJKnowledgeTracingModel(cfg, num_kp, num_problems, init_q_matrix)

    # Vanilla 模型不使用 pos_weight（标准 BCE）
    use_pos_weight = pos_weight if not model_info.get('vanilla', False) else None

    trainer = Trainer(model, cfg, train_loader, val_loader, pos_weight=use_pos_weight)
    trainer.train()

    # Evaluate best model
    if trainer.best_model_state:
        model.load_state_dict(trainer.best_model_state)

    model.eval()
    model.to(cfg.DEVICE)
    all_ac_logits, all_ac_targets = [], []
    all_mastery_pred, all_mastery_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            device_batch = {
                k: v.to(cfg.DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            ac_logits, mastery_pred = model(device_batch)
            all_ac_logits.append(ac_logits.cpu())
            all_ac_targets.append(device_batch['targets'].cpu())
            all_mastery_pred.append(mastery_pred.cpu())
            all_mastery_targets.append(device_batch['mastery_targets'].cpu())

    threshold = trainer.best_threshold
    fold_metrics = MetricsCalculator.compute_all(
        torch.cat(all_ac_logits), torch.cat(all_ac_targets),
        torch.cat(all_mastery_pred), torch.cat(all_mastery_targets),
        threshold=threshold,
    )
    fold_metrics['threshold'] = threshold
    return fold_metrics


def print_comparison_table(all_results, model_col_width=30, save_csv=None):
    """打印模型对比表"""
    metric_keys = ['auc', 'accuracy', 'balanced_accuracy', 'f1',
                   'precision', 'recall', 'mastery_mae']

    header = f"{'Model':<{model_col_width}s}"
    for key in metric_keys:
        header += f"  {key:<22s}"
    logger.info(header)
    logger.info("=" * len(header))

    for model_name, fold_metrics_list in all_results.items():
        row = f"{model_name:<{model_col_width}s}"
        for key in metric_keys:
            values = [m[key] for m in fold_metrics_list
                      if key in m and not np.isnan(m[key])]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                row += f"  {mean_val:.4f}±{std_val:.4f}       "
            else:
                row += f"  {'N/A':<22s}"
        logger.info(row)

    if save_csv:
        import csv
        import os
        dir_name = os.path.dirname(save_csv)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(save_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['model'] + metric_keys)
            for model_name, fold_metrics_list in all_results.items():
                row = [model_name]
                for key in metric_keys:
                    values = [m[key] for m in fold_metrics_list
                              if key in m and not np.isnan(m[key])]
                    row.append(f"{np.mean(values):.4f}±{np.std(values):.4f}" if values else "N/A")
                writer.writerow(row)
        logger.info(f"结果已保存到 {save_csv}")
