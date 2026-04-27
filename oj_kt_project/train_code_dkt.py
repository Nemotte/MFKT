"""
Code-DKT 在 CodeWorkout 数据集上的 K 折交叉验证

独立脚本，不修改现有 dataset/collate_fn，自带 Code-DKT 专用数据管线。
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import logging
import argparse
import copy
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.code_dkt_preprocessor import CodeDKTPreprocessor, MAX_CODE_TOKENS
from data.dataset import compute_mastery_labels
from models.code_dkt import CodeDKTBaseline
from utils.evaluation import set_seed, k_fold_split
from utils.trainer import Trainer
from utils.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


class CodeDKTDataset(Dataset):
    """Code-DKT 专用 Dataset，额外输出 code_token_ids 和 code_token_lens"""

    def __init__(self, student_timelines, preprocessor, config, problem_difficulty=None):
        self.config = config
        self.samples = []
        self.num_pos = 0
        self.num_neg = 0

        self.problem_difficulty = problem_difficulty or {}
        if self.problem_difficulty:
            all_diffs = np.stack(list(self.problem_difficulty.values()))
            self.default_difficulty = np.mean(all_diffs, axis=0)
        else:
            self.default_difficulty = np.array([0.5, 1.0], dtype=np.float32)

        num_kp = len(preprocessor.knowledge_to_idx)
        max_T = config.MAX_PROBLEM_SEQ_LEN
        window_size = config.MASTERY_WINDOW_SIZE

        for student_id, timeline in student_timelines.items():
            if len(timeline) < 2:
                continue

            mastery_labels = compute_mastery_labels(timeline, num_kp, window_size)

            for i in range(1, len(timeline)):
                history = timeline[:i]
                if len(history) > max_T:
                    history = history[-max_T:]

                problem_ids = []
                verdict_types = []
                code_token_ids_list = []
                code_token_lens_list = []

                for attempt in history:
                    problem_ids.append(attempt['problem_idx'])
                    verdict_types.append(attempt['verdict_type'])
                    code_token_ids_list.append(attempt.get(
                        'code_token_ids',
                        np.zeros(MAX_CODE_TOKENS, dtype=np.int64)
                    ))
                    code_token_lens_list.append(attempt.get('code_token_len', 0))

                next_attempt = timeline[i]
                target = 1.0 if next_attempt['session_ac'] else 0.0

                if target > 0.5:
                    self.num_pos += 1
                else:
                    self.num_neg += 1

                self.samples.append({
                    'problem_ids': problem_ids,
                    'verdict_types': verdict_types,
                    'code_token_ids': code_token_ids_list,
                    'code_token_lens': code_token_lens_list,
                    'seq_len': len(history),
                    'next_problem_id': next_attempt['problem_idx'],
                    'target': target,
                    'mastery_target': mastery_labels[i - 1],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_pos_weight(self):
        if self.num_pos == 0:
            return 1.0
        return self.num_neg / self.num_pos

    def get_class_distribution(self):
        return self.num_pos, self.num_neg


def code_dkt_collate_fn(batch):
    """Code-DKT 专用 collate，额外处理 code_token_ids [B, T, L]"""
    batch_size = len(batch)
    T_max = max(item['seq_len'] for item in batch)
    K = batch[0]['mastery_target'].shape[0]

    problem_ids = torch.zeros(batch_size, T_max, dtype=torch.long)
    verdict_types = torch.zeros(batch_size, T_max, dtype=torch.long)
    code_token_ids = torch.zeros(batch_size, T_max, MAX_CODE_TOKENS, dtype=torch.long)
    code_token_lens = torch.zeros(batch_size, T_max, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    next_problem_ids = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size)
    mastery_targets = torch.zeros(batch_size, K)
    problem_mask = torch.zeros(batch_size, T_max, dtype=torch.bool)

    # Trainer 需要的额外字段（用零填充）
    score_features = torch.zeros(batch_size, T_max, 3)
    time_features = torch.zeros(batch_size, T_max, 4)
    student_features = torch.zeros(batch_size, T_max, 4)
    problem_difficulty = torch.zeros(batch_size, T_max, 2)
    problem_categories = torch.zeros(batch_size, T_max, dtype=torch.long)
    attempt_counts = torch.zeros(batch_size, T_max, dtype=torch.long)
    next_problem_categories = torch.zeros(batch_size, dtype=torch.long)
    next_problem_difficulty = torch.zeros(batch_size, 2)

    for i, item in enumerate(batch):
        T = item['seq_len']
        seq_lens[i] = T
        next_problem_ids[i] = item['next_problem_id']
        targets[i] = item['target']
        mastery_targets[i] = torch.from_numpy(item['mastery_target'])
        problem_mask[i, :T] = True

        for t in range(T):
            problem_ids[i, t] = item['problem_ids'][t]
            verdict_types[i, t] = item['verdict_types'][t]
            code_token_ids[i, t] = torch.from_numpy(item['code_token_ids'][t])
            code_token_lens[i, t] = item['code_token_lens'][t]

    return {
        'problem_ids': problem_ids,
        'verdict_types': verdict_types,
        'code_token_ids': code_token_ids,
        'code_token_lens': code_token_lens,
        'attempt_counts': attempt_counts,
        'score_features': score_features,
        'time_features': time_features,
        'student_features': student_features,
        'problem_difficulty': problem_difficulty,
        'problem_categories': problem_categories,
        'seq_lens': seq_lens,
        'next_problem_ids': next_problem_ids,
        'next_problem_categories': next_problem_categories,
        'next_problem_difficulty': next_problem_difficulty,
        'targets': targets,
        'mastery_targets': mastery_targets,
        'problem_mask': problem_mask,
    }


def train_one_fold(fold_idx, train_tl, val_tl, preprocessor, config,
                   num_kp, num_problems):
    cfg = copy.deepcopy(config)

    problem_difficulty = preprocessor.compute_problem_difficulty(train_tl)
    train_ds = CodeDKTDataset(train_tl, preprocessor, cfg, problem_difficulty)
    val_ds = CodeDKTDataset(val_tl, preprocessor, cfg, problem_difficulty)

    pos, neg = train_ds.get_class_distribution()
    pos_weight = train_ds.get_pos_weight()
    logger.info(f"    样本: 训练={len(train_ds)}, 验证={len(val_ds)}, "
                f"正/负={pos}/{neg}, pos_weight={pos_weight:.4f}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=code_dkt_collate_fn,
        num_workers=getattr(cfg, 'NUM_WORKERS', 0),
        pin_memory=getattr(cfg, 'PIN_MEMORY', False),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=code_dkt_collate_fn,
        num_workers=getattr(cfg, 'NUM_WORKERS', 0),
        pin_memory=getattr(cfg, 'PIN_MEMORY', False),
    )

    model = CodeDKTBaseline(
        num_problems=num_problems,
        num_kp=num_kp,
        num_token_types=17,
        emb_size=64,
        hidden_size=64,
        code_emb_dim=16,
        code_hidden_dim=32,
        dropout_rate=0.2,
    )

    trainer = Trainer(model, cfg, train_loader, val_loader, pos_weight=pos_weight)
    trainer.train()

    if trainer.best_model_state:
        model.load_state_dict(trainer.best_model_state)

    model.eval()
    model.to(cfg.DEVICE)
    all_ac_logits, all_ac_targets = [], []
    all_mastery_pred, all_mastery_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            device_batch = {
                k: v.to(cfg.DEVICE) if isinstance(v, torch.Tensor) else v
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


def main(args):
    seed = 42
    set_seed(seed)
    k = args.k_folds

    logger.info("=" * 70)
    logger.info(f"Code-DKT CodeWorkout K 折交叉验证 — {k} 折")
    logger.info("=" * 70)

    # 1. 加载数据
    logger.info("[1/3] 加载 CodeWorkout 数据 + 代码特征...")
    preprocessor = CodeDKTPreprocessor(args.data_dir)
    student_timelines, num_problems, num_kp, init_q_matrix = preprocessor.load_and_build()

    config = Config()
    config.NUM_VERDICT_TYPES = 2
    config.NUM_ALGO_CATEGORIES = 1
    config.SEED = seed
    # Code-DKT 使用标准训练策略
    config.USE_SWA = False
    config.USE_COSINE_ANNEALING = False
    config.WARMUP_EPOCHS = 0
    config.USE_FOCAL_LOSS = False
    config.LABEL_SMOOTHING = 0.0
    config.MASTERY_LOSS_WEIGHT = 0.0
    config.QMATRIX_LOSS_WEIGHT = 0.0

    # 2. K-fold split
    logger.info(f"[2/3] 按学生进行 {k} 折划分...")
    folds = k_fold_split(student_timelines, k, seed=seed)

    # 3. Train
    logger.info("[3/3] 开始训练 Code-DKT...")
    fold_metrics_list = []

    for fold_idx, (train_tl, val_tl) in enumerate(folds):
        set_seed(seed + fold_idx)
        logger.info(f"\n  Fold {fold_idx + 1}/{k}")

        config.MODEL_SAVE_PATH = f"checkpoints/codeworkout/Code-DKT/"

        fold_metrics = train_one_fold(
            fold_idx, train_tl, val_tl, preprocessor, config,
            num_kp, num_problems,
        )

        fold_metrics_list.append(fold_metrics)
        logger.info(f"    Fold {fold_idx + 1} AUC={fold_metrics.get('auc', 0):.4f}  "
                     f"BA={fold_metrics.get('balanced_accuracy', 0):.4f}  "
                     f"F1={fold_metrics.get('f1', 0):.4f}")

    # 汇总
    logger.info("\n" + "=" * 70)
    logger.info("Code-DKT 实验汇总")
    logger.info("=" * 70)

    metric_keys = ['auc', 'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']
    for key in metric_keys:
        values = [m[key] for m in fold_metrics_list if key in m and not np.isnan(m[key])]
        if values:
            logger.info(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code-DKT CodeWorkout 实验")
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--data-dir', type=str, default=None)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'data', 'All')
    main(args)
