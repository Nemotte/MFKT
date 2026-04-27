"""
统计显著性检验：MFKT vs baselines
收集每折 AUC，做配对 Wilcoxon signed-rank test + Bootstrap 95% CI
"""
import sys, os, copy, logging, argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from config import Config
from data.preprocessor import OJDataPreprocessor
from data.dataset import StudentTimelineDataset, collate_fn
from models.model import OJKnowledgeTracingModel
from models.registry import STAT_TEST_MODEL_REGISTRY as MODELS
from utils.evaluation import set_seed, k_fold_split
from utils.trainer import Trainer
from utils.metrics import MetricsCalculator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def train_fold(train_tl, val_tl, model_name, model_info, preprocessor, config, num_kp, num_problems, init_q_matrix):
    cfg = copy.deepcopy(config)
    problem_difficulty = preprocessor.compute_problem_difficulty(train_tl)
    train_ds = StudentTimelineDataset(train_tl, preprocessor, cfg, problem_difficulty=problem_difficulty)
    val_ds   = StudentTimelineDataset(val_tl,   preprocessor, cfg, problem_difficulty=problem_difficulty)
    pos_weight = train_ds.get_pos_weight()

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  collate_fn=collate_fn,
                               num_workers=getattr(cfg, 'NUM_WORKERS', 0), pin_memory=getattr(cfg, 'PIN_MEMORY', False))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                               num_workers=getattr(cfg, 'NUM_WORKERS', 0), pin_memory=getattr(cfg, 'PIN_MEMORY', False))

    if model_info.get('type') == 'baseline':
        model = model_info['class'](num_problems=num_problems, num_kp=num_kp, **model_info.get('params', {}))
        use_pw = None
    else:
        model = OJKnowledgeTracingModel(cfg, num_kp, num_problems, init_q_matrix)
        use_pw = pos_weight

    trainer = Trainer(model, cfg, train_loader, val_loader, pos_weight=use_pw)
    trainer.train()
    if trainer.best_model_state:
        model.load_state_dict(trainer.best_model_state)

    model.eval()
    model.to(cfg.DEVICE)
    all_logits, all_targets = [], []
    all_mp, all_mt = [], []
    with torch.no_grad():
        for batch in val_loader:
            db = {k: v.to(cfg.DEVICE) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            ac_logits, mastery_pred = model(db)
            all_logits.append(ac_logits.cpu())
            all_targets.append(db['targets'].cpu())
            all_mp.append(mastery_pred.cpu())
            all_mt.append(db['mastery_targets'].cpu())

    metrics = MetricsCalculator.compute_all(
        torch.cat(all_logits), torch.cat(all_targets),
        torch.cat(all_mp), torch.cat(all_mt),
        threshold=trainer.best_threshold,
    )
    return metrics['auc']

def bootstrap_ci(diff_arr, n_boot=10000, ci=0.95, seed=0):
    """对 per-fold AUC 差值做 bootstrap，返回 (mean, ci_low, ci_high)。"""
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(diff_arr, size=len(diff_arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return boot_means.mean(), np.quantile(boot_means, alpha), np.quantile(boot_means, 1 - alpha)


def main():
    parser = argparse.ArgumentParser(description='MFKT 统计显著性检验')
    parser.add_argument('--k-folds', type=int, default=10,
                        help='交叉验证折数 (默认 10，折数越多 Wilcoxon p 值越小)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='gold',
                        choices=['gold', 'standard'], help='数据集变体')
    parser.add_argument('--n-boot', type=int, default=10000,
                        help='Bootstrap 重采样次数 (默认 10000)')
    args = parser.parse_args()

    K = args.k_folds
    seed = args.seed
    set_seed(seed)

    print(f"加载数据 (dataset={args.dataset}, K={K})...")
    config = Config(DATASET_VARIANT=args.dataset)
    preprocessor = OJDataPreprocessor(config)
    submissions, knowledge_data, problem_data = preprocessor.load_data()
    preprocessor.build_vocabularies(knowledge_data, problem_data, submissions=submissions)
    num_kp = len(preprocessor.knowledge_to_idx)
    num_problems = len(preprocessor.problem_to_idx)
    student_timelines = preprocessor.create_student_timelines(submissions)
    init_q_matrix = preprocessor.build_q_matrix_init()
    folds = k_fold_split(student_timelines, K, seed=seed)

    results = {name: [] for name in MODELS}

    for name, info in MODELS.items():
        print(f"\n=== {name} ===")
        for fold_idx, (train_tl, val_tl) in enumerate(folds):
            set_seed(seed + fold_idx)
            cfg = Config(DATASET_VARIANT=args.dataset)
            if name == 'MFKT':
                cfg.MODEL_TYPE = 'transformer'
            cfg.MODEL_SAVE_PATH = f'checkpoints/stat_test/{name}/'
            auc = train_fold(train_tl, val_tl, name, info, preprocessor, cfg, num_kp, num_problems, init_q_matrix)
            results[name].append(auc)
            print(f"  Fold {fold_idx+1}: AUC={auc:.4f}")
        arr = np.array(results[name])
        print(f"  {name}: {arr.mean():.4f} ± {arr.std():.4f}")

    print("\n" + "="*90)
    print(f"Wilcoxon signed-rank test ({K}-fold): MFKT vs each baseline")
    print(f"多重比较校正：Holm-Bonferroni（对 {len(MODELS)-1} 个对比）")
    print("="*90)
    mfkt = np.array(results['MFKT'])
    print(f"MFKT per-fold AUC: {mfkt}")
    print(f"\n{'Baseline':<14s}  {'W':>6s}  {'p(raw)':>8s}  {'p(Holm)':>9s}  {'sig':>4s}  "
          f"{'mean_diff':>10s}  {'d':>6s}  {'95% CI (boot)':>22s}")
    print("-" * 95)

    # 先收集所有 p 值，再做 Holm 校正
    baseline_names = list(MODELS.keys())[1:]
    raw_results_list = []
    for name in baseline_names:
        baseline = np.array(results[name])
        diff = mfkt - baseline
        stat, p = stats.wilcoxon(diff, alternative='greater')
        cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float('inf')
        bmean, blo, bhi = bootstrap_ci(diff, n_boot=args.n_boot, seed=seed)
        raw_results_list.append((name, stat, p, diff.mean(), cohens_d, blo, bhi))

    # Holm-Bonferroni 校正
    raw_ps = np.array([r[2] for r in raw_results_list])
    m = len(raw_ps)
    order = np.argsort(raw_ps)
    holm_ps = np.zeros(m)
    for rank, idx in enumerate(order):
        holm_ps[idx] = min(1.0, raw_ps[idx] * (m - rank))
    # 单调性保证（Holm 要求 p_corrected 单调不减）
    for i in range(1, m):
        holm_ps[order[i]] = max(holm_ps[order[i]], holm_ps[order[i-1]])

    for i, (name, stat, p, mean_diff, cohens_d, blo, bhi) in enumerate(raw_results_list):
        p_holm = holm_ps[i]
        sig = '***' if p_holm < 0.01 else ('**' if p_holm < 0.05 else ('*' if p_holm < 0.1 else 'ns'))
        print(f"  MFKT vs {name:<12s}  W={stat:>4.0f}  p={p:.4f}  p_H={p_holm:.4f}  {sig:>3s}  "
              f"diff={mean_diff:+.4f}  d={cohens_d:>5.2f}  [{blo:+.4f}, {bhi:+.4f}]")

    print(f"\n注: p(Holm) 为 Holm-Bonferroni 校正后的 p 值；sig 基于 p(Holm)。")

if __name__ == '__main__':
    main()
