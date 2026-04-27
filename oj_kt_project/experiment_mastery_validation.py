#!/usr/bin/env python
"""Mastery 估计有效性验证

验证思路：如果 mastery 估计有效，那么 mastery 值高的知识点，
学生后续在该知识点上的 AC 率应该更高。

方法：
1. 用训练好的模型对验证集每个样本输出 mastery [K]
2. 对每个样本，取 next_problem 涉及的知识点的 mastery 值
3. 按 mastery 值分桶（0-0.2, 0.2-0.4, ...），统计每个桶内的实际 AC 率
4. 如果 mastery 越高 → AC 率越高，说明 mastery 估计有效
"""
import os
import sys
import torch
import numpy as np
import logging
import argparse
import copy
from collections import defaultdict
from torch.utils.data import DataLoader

from config import Config
from data.preprocessor import OJDataPreprocessor
from data.dataset import StudentTimelineDataset, collate_fn
from models.model import OJKnowledgeTracingModel
from utils.evaluation import set_seed, k_fold_split
from utils.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def run_mastery_validation(dataset_variant='gold', k_folds=5, num_bins=5):
    config = Config(DATASET_VARIANT=dataset_variant)
    set_seed(config.SEED)

    preprocessor = OJDataPreprocessor(config)
    submissions, knowledge_data, problem_data = preprocessor.load_data()
    preprocessor.build_vocabularies(knowledge_data, problem_data, submissions=submissions)
    num_kp = len(preprocessor.knowledge_to_idx)
    num_problems = len(preprocessor.problem_to_idx)
    student_timelines = preprocessor.create_student_timelines(submissions)
    init_q_matrix = preprocessor.build_q_matrix_init()

    folds = k_fold_split(student_timelines, k_folds, seed=config.SEED)

    # 收集所有 fold 的 (mastery_value, actual_ac) 对
    all_pairs = []  # [(mastery_val, ac_label), ...]

    for fold_idx, (train_tl, val_tl) in enumerate(folds):
        set_seed(config.SEED + fold_idx)
        logger.info(f"Fold {fold_idx + 1}/{k_folds}")

        cfg = copy.deepcopy(config)
        cfg.MODEL_SAVE_PATH = f"checkpoints/mastery_val/fold_{fold_idx}"

        problem_difficulty = preprocessor.compute_problem_difficulty(train_tl)
        train_ds = StudentTimelineDataset(train_tl, preprocessor, cfg, problem_difficulty=problem_difficulty)
        val_ds = StudentTimelineDataset(val_tl, preprocessor, cfg, problem_difficulty=problem_difficulty)

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=getattr(cfg, 'NUM_WORKERS', 0),
                                  pin_memory=getattr(cfg, 'PIN_MEMORY', False))
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn,
                                num_workers=getattr(cfg, 'NUM_WORKERS', 0),
                                pin_memory=getattr(cfg, 'PIN_MEMORY', False))

        model = OJKnowledgeTracingModel(cfg, num_kp, num_problems, init_q_matrix)
        pos_weight = train_ds.get_pos_weight()
        trainer = Trainer(model, cfg, train_loader, val_loader, pos_weight=pos_weight)
        trainer.train()

        if trainer.best_model_state:
            model.load_state_dict(trainer.best_model_state)

        model.eval()
        model.to(cfg.DEVICE)

        # 获取 Q-matrix 用于查找 next_problem 涉及的知识点
        q_matrix = model.q_matrix.annotated_q.detach().cpu().numpy()  # [num_problems, num_kp]

        with torch.no_grad():
            for batch in val_loader:
                device_batch = {
                    k: v.to(cfg.DEVICE) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                _, mastery_pred = model(device_batch)  # [B, K]
                mastery_np = mastery_pred.cpu().numpy()
                targets = batch['targets'].numpy()  # [B]
                next_pids = batch['next_problem_ids'].numpy()  # [B]

                for i in range(len(targets)):
                    pid = next_pids[i]
                    ac = targets[i]
                    # 找 next_problem 涉及的知识点
                    kp_weights = q_matrix[pid]  # [K]
                    active_kps = np.where(kp_weights > 0.5)[0]

                    if len(active_kps) == 0:
                        continue

                    # 取涉及知识点的平均 mastery
                    avg_mastery = mastery_np[i, active_kps].mean()
                    all_pairs.append((float(avg_mastery), float(ac)))

    # 分桶统计
    all_pairs = np.array(all_pairs)
    logger.info(f"\n总样本数: {len(all_pairs)}")

    bin_edges = np.linspace(0, 1, num_bins + 1)
    logger.info(f"\n{'='*60}")
    logger.info("Mastery 估计有效性验证")
    logger.info(f"{'='*60}")
    logger.info(f"{'Mastery Range':<20s}  {'Samples':<10s}  {'AC Rate':<10s}  {'Avg Mastery':<12s}")
    logger.info("-" * 55)

    bin_results = []
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < num_bins - 1:
            mask = (all_pairs[:, 0] >= lo) & (all_pairs[:, 0] < hi)
        else:
            mask = (all_pairs[:, 0] >= lo) & (all_pairs[:, 0] <= hi)

        count = mask.sum()
        if count > 0:
            ac_rate = all_pairs[mask, 1].mean()
            avg_m = all_pairs[mask, 0].mean()
            bin_results.append((lo, hi, count, ac_rate, avg_m))
            logger.info(f"[{lo:.1f}, {hi:.1f}{']' if i == num_bins-1 else ')'}{'':>12s}  {count:<10d}  {ac_rate:<10.4f}  {avg_m:<12.4f}")
        else:
            logger.info(f"[{lo:.1f}, {hi:.1f}{']' if i == num_bins-1 else ')'}{'':>12s}  {'0':<10s}  {'N/A':<10s}  {'N/A':<12s}")

    # 计算相关系数
    if len(all_pairs) > 0:
        from scipy import stats
        corr, p_value = stats.pearsonr(all_pairs[:, 0], all_pairs[:, 1])
        spearman_corr, sp_p = stats.spearmanr(all_pairs[:, 0], all_pairs[:, 1])

        logger.info(f"\n{'='*60}")
        logger.info(f"Pearson  相关系数: {corr:.4f}  (p={p_value:.2e})")
        logger.info(f"Spearman 相关系数: {spearman_corr:.4f}  (p={sp_p:.2e})")
        logger.info(f"{'='*60}")

        if corr > 0 and p_value < 0.05:
            logger.info("结论: Mastery 估计与实际 AC 率显著正相关，验证了 mastery 估计的有效性")
        else:
            logger.info("结论: Mastery 估计与实际 AC 率相关性不显著")

    # 保存结果
    output_file = f"mastery_validation_{dataset_variant}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Mastery Estimation Validity Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {dataset_variant}\n")
        f.write(f"K-folds: {k_folds}\n")
        f.write(f"Total samples: {len(all_pairs)}\n\n")

        f.write(f"{'Mastery Range':<20s}  {'Samples':<10s}  {'AC Rate':<10s}  {'Avg Mastery':<12s}\n")
        f.write("-" * 55 + "\n")
        for lo, hi, count, ac_rate, avg_m in bin_results:
            f.write(f"[{lo:.1f}, {hi:.1f}]{'':>14s}  {count:<10d}  {ac_rate:<10.4f}  {avg_m:<12.4f}\n")

        f.write(f"\nPearson r = {corr:.4f} (p = {p_value:.2e})\n")
        f.write(f"Spearman rho = {spearman_corr:.4f} (p = {sp_p:.2e})\n")

    logger.info(f"\n结果已保存到 {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mastery 估计有效性验证')
    parser.add_argument('--dataset', type=str, default='gold',
                        choices=['gold', 'standard', 'raw'])
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--bins', type=int, default=5)
    args = parser.parse_args()

    run_mastery_validation(
        dataset_variant=args.dataset,
        k_folds=args.k_folds,
        num_bins=args.bins,
    )
