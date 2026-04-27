"""
超参数搜索 — 2-fold 快速筛选最优超参组合
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy
import itertools
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from data.preprocessor import OJDataPreprocessor
from data.dataset import StudentTimelineDataset, collate_fn
from models.model import OJKnowledgeTracingModel
from utils.evaluation import set_seed, k_fold_split
from utils.trainer import Trainer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_config(param_dict, folds, preprocessor, num_kp, num_problems, init_q_matrix):
    """用 2-fold 评估一组超参，返回平均 AUC"""
    aucs = []
    for fold_idx, (train_tl, val_tl) in enumerate(folds):
        cfg = Config(STUDENT_FEATURE_INPUT_DIM=2)  # V1
        for k, v in param_dict.items():
            setattr(cfg, k, v)
        cfg.MODEL_TYPE = 'transformer'
        cfg.validate()

        set_seed(42 + fold_idx)

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
        trainer = Trainer(model, cfg, train_loader, val_loader,
                          pos_weight=train_ds.get_pos_weight())
        trainer.train()

        aucs.append(trainer.best_val_auc)

    return np.mean(aucs), np.std(aucs)


def main():
    # 加载数据（只做一次）
    config = Config()
    preprocessor = OJDataPreprocessor(config)
    submissions, knowledge_data, problem_data = preprocessor.load_data()
    preprocessor.build_vocabularies(knowledge_data, problem_data, submissions)
    timelines = preprocessor.create_student_timelines(submissions)
    init_q_matrix = preprocessor.build_q_matrix_init()
    num_kp = len(preprocessor.knowledge_to_idx)
    num_problems = len(preprocessor.problem_to_idx)

    # 2-fold 快速评估
    folds = k_fold_split(timelines, k=2, seed=42)

    # 搜索空间
    search_space = {
        'HIDDEN_DIM': [32, 48, 64],
        'DROPOUT': [0.25, 0.35, 0.45],
        'NUM_TRANSFORMER_LAYERS': [1, 2],
        'FF_DIM': [64, 96],
        'LEARNING_RATE': [0.001, 0.0005],
    }

    # 生成所有组合
    keys = list(search_space.keys())
    values = list(search_space.values())
    all_combos = list(itertools.product(*values))
    logger.info(f"共 {len(all_combos)} 组超参组合")

    # 过滤：HIDDEN_DIM 必须能被 NUM_HEADS(2) 整除
    valid_combos = []
    for combo in all_combos:
        param_dict = dict(zip(keys, combo))
        if param_dict['HIDDEN_DIM'] % 2 == 0:
            valid_combos.append(param_dict)
    logger.info(f"有效组合: {len(valid_combos)}")

    results = []
    for idx, param_dict in enumerate(valid_combos):
        logger.info(f"\n[{idx+1}/{len(valid_combos)}] {param_dict}")
        try:
            mean_auc, std_auc = evaluate_config(
                param_dict, folds, preprocessor, num_kp, num_problems, init_q_matrix
            )
            results.append((mean_auc, std_auc, param_dict))
            logger.info(f"  AUC = {mean_auc:.4f} ± {std_auc:.4f}")
        except Exception as e:
            logger.error(f"  失败: {e}")
            continue

    # 排序输出
    results.sort(key=lambda x: -x[0])
    logger.info("\n" + "=" * 70)
    logger.info("超参搜索结果 (Top 10)")
    logger.info("=" * 70)
    for rank, (auc, std, params) in enumerate(results[:10], 1):
        logger.info(f"#{rank}  AUC={auc:.4f}±{std:.4f}  {params}")

    # 保存完整结果
    with open('hyperparam_search_results.txt', 'w') as f:
        for auc, std, params in results:
            f.write(f"AUC={auc:.4f}±{std:.4f}  {params}\n")
    logger.info("\n结果已保存到 hyperparam_search_results.txt")


if __name__ == '__main__':
    main()
