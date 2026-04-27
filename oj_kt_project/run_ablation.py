"""
消融实验 — 逐一移除模型组件，验证各模块贡献

消融维度:
  A. 输入特征: time / student / difficulty / category / score / q_matrix_emb
  B. 架构组件: KG注意力 / 可学习Q-Matrix / 掌握度多任务 / Q-Matrix正则
  C. 训练策略: Focal Loss / SWA / Label Smoothing / Warmup+Cosine

用法:
  python run_ablation.py                          # 运行全部消融
  python run_ablation.py --groups A               # 只跑输入特征消融
  python run_ablation.py --groups A B             # 跑 A 和 B
  python run_ablation.py --variants w/o-time w/o-KG-attention  # 指定变体
  python run_ablation.py --k-folds 3              # 3折（加速）
"""
import torch
import numpy as np
import logging
import argparse
import copy
from collections import OrderedDict
from torch.utils.data import DataLoader
from scipy import stats as scipy_stats

from config import Config
from data.preprocessor import OJDataPreprocessor
from data.dataset import StudentTimelineDataset, collate_fn
from models.model import OJKnowledgeTracingModel
from utils.evaluation import set_seed, k_fold_split
from utils.trainer import Trainer
from utils.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================
# 消融变体定义
# ============================================================

# 每个变体: (名称, 组别, 描述, config修改dict, forward_hook名称或None)
# forward_hook 用于在前向传播中置零特定特征

ABLATION_VARIANTS = OrderedDict([
    # ── A. 输入特征消融 ──
    ('w/o-time', {
        'group': 'A', 'desc': '移除时间特征',
        'config': {}, 'hook': 'zero_time',
    }),
    ('w/o-student', {
        'group': 'A', 'desc': '移除学生能力特征',
        'config': {}, 'hook': 'zero_student',
    }),
    ('w/o-difficulty', {
        'group': 'A', 'desc': '移除题目难度特征',
        'config': {}, 'hook': 'zero_difficulty',
    }),
    ('w/o-category', {
        'group': 'A', 'desc': '移除算法类别嵌入',
        'config': {}, 'hook': 'zero_category',
    }),
    ('w/o-score', {
        'group': 'A', 'desc': '移除得分/耗时/内存特征',
        'config': {}, 'hook': 'zero_score',
    }),
    ('binary-verdict', {
        'group': 'A', 'desc': '6类verdict退化为2类(AC/非AC)',
        'config': {}, 'hook': 'binary_verdict',
    }),
    ('MFKT-binary', {
        'group': 'A', 'desc': 'MFKT完整架构 + 纯binary输入(清零所有提交元数据)',
        'config': {}, 'hook': 'binary_all',
    }),

    # ── B. 架构组件消融 ──
    ('w/o-KG-attention', {
        'group': 'B', 'desc': '移除KG引导注意力 → 普通注意力',
        'config': {'USE_KG_ATTENTION': False}, 'hook': None,
    }),
    ('w/o-mastery', {
        'group': 'B', 'desc': '移除掌握度多任务学习',
        'config': {'MASTERY_LOSS_WEIGHT': 0.0}, 'hook': None,
    }),
    ('w/o-Q-reg', {
        'group': 'B', 'desc': '移除Q-Matrix正则化损失',
        'config': {'QMATRIX_LOSS_WEIGHT': 0.0}, 'hook': None,
    }),

    # ── C. 训练策略消融 ──
    ('w/o-focal', {
        'group': 'C', 'desc': '替换Focal Loss为BCE',
        'config': {'USE_FOCAL_LOSS': False}, 'hook': None,
    }),
    ('w/o-SWA', {
        'group': 'C', 'desc': '移除SWA',
        'config': {'USE_SWA': False}, 'hook': None,
    }),
    ('w/o-label-smooth', {
        'group': 'C', 'desc': '移除Label Smoothing',
        'config': {'LABEL_SMOOTHING': 0.0}, 'hook': None,
    }),
    ('w/o-warmup-cosine', {
        'group': 'C', 'desc': '替换Warmup+Cosine为ReduceLROnPlateau',
        'config': {'WARMUP_EPOCHS': 0, 'USE_COSINE_ANNEALING': False}, 'hook': None,
    }),
])


# ============================================================
# 特征置零 Hook — 包装模型 forward，将指定特征置零
# ============================================================

class AblationModelWrapper(torch.nn.Module):
    """包装原始模型，在 forward 中对指定 batch 字段置零"""

    def __init__(self, model, hook_name):
        super().__init__()
        self.model = model
        self.hook_name = hook_name

    # 代理属性，让 Trainer 能正常访问
    @property
    def is_baseline(self):
        return getattr(self.model, 'is_baseline', False)

    @property
    def q_matrix(self):
        return self.model.q_matrix

    def get_q_regularization_loss(self):
        return self.model.get_q_regularization_loss()

    def parameters(self, recurse=True):
        return self.model.parameters(recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.model.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def forward(self, batch):
        batch = self._apply_hook(batch)
        return self.model(batch)

    def _apply_hook(self, batch):
        """根据 hook_name 将对应特征置零"""
        b = dict(batch)  # 浅拷贝，避免污染原始 batch

        if self.hook_name == 'zero_time':
            b['time_features'] = torch.zeros_like(b['time_features'])

        elif self.hook_name == 'zero_student':
            b['student_features'] = torch.zeros_like(b['student_features'])

        elif self.hook_name == 'zero_difficulty':
            b['problem_difficulty'] = torch.zeros_like(b['problem_difficulty'])
            b['next_problem_difficulty'] = torch.zeros_like(b['next_problem_difficulty'])

        elif self.hook_name == 'zero_category':
            # 置为 0 号类别（相当于统一类别，消除区分度）
            b['problem_categories'] = torch.zeros_like(b['problem_categories'])
            b['next_problem_categories'] = torch.zeros_like(b['next_problem_categories'])

        elif self.hook_name == 'zero_score':
            b['score_features'] = torch.zeros_like(b['score_features'])

        elif self.hook_name == 'binary_verdict':
            # 6类verdict → 2类: AC(0)保持0, 非AC(1-5)全部映射为1
            vt = b['verdict_types']
            b['verdict_types'] = (vt > 0).long()
            # 也将 verdict_dist 退化为 binary: [AC_freq, non-AC_freq, 0, 0, 0, 0]
            if 'verdict_dist' in b:
                vd = b['verdict_dist']
                binary_vd = torch.zeros_like(vd)
                binary_vd[..., 0] = vd[..., 0]           # AC 频率保持
                binary_vd[..., 1] = vd[..., 1:].sum(-1)  # 非AC 频率合并
                b['verdict_dist'] = binary_vd

        elif self.hook_name == 'binary_all':
            # MFKT 完整架构 + 纯 binary 输入：
            # 只保留 binary correct/incorrect 信号，清零所有提交元数据和统计特征
            # 保留的架构组件: KG-attention, Q-matrix embed, mastery head, focal loss
            vt = b['verdict_types']
            b['verdict_types'] = (vt > 0).long()
            b['score_features'] = torch.zeros_like(b['score_features'])
            b['time_features'] = torch.zeros_like(b['time_features'])
            b['student_features'] = torch.zeros_like(b['student_features'])
            b['problem_difficulty'] = torch.zeros_like(b['problem_difficulty'])
            b['next_problem_difficulty'] = torch.zeros_like(b['next_problem_difficulty'])
            b['problem_categories'] = torch.zeros_like(b['problem_categories'])
            b['next_problem_categories'] = torch.zeros_like(b['next_problem_categories'])
            b['attempt_counts'] = torch.zeros_like(b['attempt_counts'])
            if 'verdict_dist' in b:
                b['verdict_dist'] = torch.zeros_like(b['verdict_dist'])

        return b


# ============================================================
# 训练单折
# ============================================================

def train_one_fold(fold_idx, train_tl, val_tl, variant_name, variant_info,
                   preprocessor, config, num_kp, num_problems, init_q_matrix):
    """训练单折并返回验证指标"""
    set_seed(config.SEED + fold_idx)

    cfg = copy.deepcopy(config)
    for k, v in variant_info.get('config', {}).items():
        setattr(cfg, k, v)
    cfg.validate()

    problem_difficulty = preprocessor.compute_problem_difficulty(train_tl)
    train_ds = StudentTimelineDataset(train_tl, preprocessor, cfg, problem_difficulty=problem_difficulty)
    val_ds = StudentTimelineDataset(val_tl, preprocessor, cfg, problem_difficulty=problem_difficulty)

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.warning(f"  Fold {fold_idx + 1}: 数据集为空，跳过")
        return {}

    pos_weight = train_ds.get_pos_weight()
    logger.info(f"    样本: 训练={len(train_ds)}, 验证={len(val_ds)}, pos_weight={pos_weight:.4f}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=getattr(cfg, 'NUM_WORKERS', 0),
                              pin_memory=getattr(cfg, 'PIN_MEMORY', False))
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=getattr(cfg, 'NUM_WORKERS', 0),
                            pin_memory=getattr(cfg, 'PIN_MEMORY', False))

    # 构建模型
    prerequisite_adj = None
    if hasattr(preprocessor, 'knowledge_structure') and preprocessor.knowledge_structure:
        ks = preprocessor.knowledge_structure
        if hasattr(ks, 'get_prerequisite_adjacency'):
            prerequisite_adj = ks.get_prerequisite_adjacency(num_kp)

    model = OJKnowledgeTracingModel(
        cfg, num_kp, num_problems, init_q_matrix, prerequisite_adj,
    )

    # 如果有 forward hook，用 wrapper 包装
    hook_name = variant_info.get('hook')
    if hook_name:
        model = AblationModelWrapper(model, hook_name)

    trainer = Trainer(model, cfg, train_loader, val_loader, pos_weight)
    trainer.train(cfg.NUM_EPOCHS)

    # 评估最佳模型
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


# ============================================================
# 结果输出
# ============================================================

METRIC_KEYS = ['auc', 'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'mastery_mae']


def print_ablation_table(all_results):
    """打印消融实验对比表"""
    header = f"{'Variant':<25s}"
    for key in METRIC_KEYS:
        header += f"{key:<24s}"
    logger.info(header)
    logger.info("=" * (25 + 24 * len(METRIC_KEYS)))

    for name, fold_metrics_list in all_results.items():
        row = f"{name:<25s}"
        for key in METRIC_KEYS:
            values = [m[key] for m in fold_metrics_list if key in m and not np.isnan(m.get(key, float('nan')))]
            if values:
                mean_v = np.mean(values)
                std_v = np.std(values)
                row += f"{mean_v:.4f}\u00b1{std_v:.4f}{'':>8s}"
            else:
                row += f"{'N/A':<24s}"
        logger.info(row)


def tost_equivalence(full_aucs, variant_aucs, delta=0.005, alpha=0.05):
    """
    TOST 等效性检验（双单侧 t 检验）
    H0: |mean(full - variant)| >= delta
    H1: |mean(full - variant)| < delta (等效)

    返回 (p_lower, p_upper, equiv_decision)
    p_lower: full - variant > -delta (下界单侧检验 p 值)
    p_upper: full - variant < +delta (上界单侧检验 p 值)
    等效当且仅当 max(p_lower, p_upper) < alpha
    """
    diff = np.array(full_aucs) - np.array(variant_aucs)
    n = len(diff)
    mean_d = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 1.0, 1.0, False
    # 下界检验: t = (mean_d - (-delta)) / se, 单侧右尾
    t_lower = (mean_d + delta) / se
    p_lower = 1 - scipy_stats.t.cdf(t_lower, df=n - 1)
    # 上界检验: t = (delta - mean_d) / se, 单侧右尾
    t_upper = (delta - mean_d) / se
    p_upper = 1 - scipy_stats.t.cdf(t_upper, df=n - 1)
    equiv = max(p_lower, p_upper) < alpha
    return p_lower, p_upper, equiv


def print_delta_table(all_results, equiv_delta=0.005):
    """打印相对于 Full Model 的 delta 表，附配对 t 检验显著性、TOST 等效性检验和 95% CI"""
    if 'Full-Model' not in all_results:
        return

    full_fold_aucs = [m['auc'] for m in all_results['Full-Model']
                      if 'auc' in m and not np.isnan(m.get('auc', float('nan')))]

    full_metrics = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in all_results['Full-Model'] if key in m and not np.isnan(m.get(key, float('nan')))]
        full_metrics[key] = np.mean(values) if values else None

    logger.info("")
    logger.info("Delta vs Full-Model (负值 = 性能下降 = 该组件有正贡献)")
    logger.info(f"TOST 等效界: δ={equiv_delta} AUC | EQUIV 表示检验结论'可等效'（注意：不等于证明无差异）")
    logger.info("=" * 115)
    header = (f"{'Variant':<25s}{'Δ AUC':<10s}{'95% CI':<22s}{'p(t-test)':<12s}"
              f"{'TOST p_low':<12s}{'TOST p_up':<12s}{'Equiv?':<8s}")
    logger.info(header)
    logger.info("-" * 101)

    for name, fold_metrics_list in all_results.items():
        if name == 'Full-Model':
            continue
        row = f"{name:<25s}"
        # Δ AUC + 95% Bootstrap CI
        variant_fold_aucs = [m['auc'] for m in fold_metrics_list
                             if 'auc' in m and not np.isnan(m.get('auc', float('nan')))]
        auc_values = [m['auc'] for m in fold_metrics_list if 'auc' in m and not np.isnan(m.get('auc', float('nan')))]
        if auc_values and full_metrics.get('auc') is not None:
            delta_auc = np.mean(auc_values) - full_metrics['auc']
            sign = "+" if delta_auc >= 0 else ""
            row += f"{sign}{delta_auc:.4f}{'':>3s}"
        else:
            row += f"{'N/A':<10s}"

        # 95% CI via bootstrap
        if len(variant_fold_aucs) == len(full_fold_aucs) and len(full_fold_aucs) > 1:
            diff_arr = np.array(variant_fold_aucs) - np.array(full_fold_aucs)
            rng = np.random.default_rng(42)
            boot_means = np.array([
                rng.choice(diff_arr, size=len(diff_arr), replace=True).mean()
                for _ in range(5000)
            ])
            ci_lo, ci_hi = np.quantile(boot_means, 0.025), np.quantile(boot_means, 0.975)
            row += f"[{ci_lo:+.4f},{ci_hi:+.4f}]{'':>2s}"
        else:
            row += f"{'N/A':<22s}"

        # 配对 t 检验
        if len(variant_fold_aucs) == len(full_fold_aucs) and len(full_fold_aucs) > 1:
            _, p_val = scipy_stats.ttest_rel(full_fold_aucs, variant_fold_aucs)
            row += f"p={p_val:.4f}{'':>4s}"
        else:
            row += f"{'N/A':<12s}"

        # TOST 等效性检验
        if len(variant_fold_aucs) == len(full_fold_aucs) and len(full_fold_aucs) > 1:
            p_lo, p_up, is_equiv = tost_equivalence(full_fold_aucs, variant_fold_aucs, delta=equiv_delta)
            equiv_str = "EQUIV" if is_equiv else "n/a"
            row += f"p={p_lo:.4f}{'':>4s}p={p_up:.4f}{'':>4s}{equiv_str:<8s}"
        else:
            row += f"{'N/A':<12s}{'N/A':<12s}{'N/A':<8s}"

        logger.info(row)


# ============================================================
# Main
# ============================================================

def main(args):
    base_config = Config(DATASET_VARIANT=args.dataset)

    logger.info(f"数据集: {args.dataset} | K折: {args.k_folds}")
    logger.info(f"设备: {base_config.DEVICE}")

    # 加载数据
    preprocessor = OJDataPreprocessor(base_config)
    submissions, knowledge_data, problem_data = preprocessor.load_data()

    preprocessor.build_vocabularies(knowledge_data, problem_data, submissions=submissions)
    num_kp = len(preprocessor.knowledge_to_idx)
    num_problems = len(preprocessor.problem_to_idx)
    student_timelines = preprocessor.create_student_timelines(submissions)
    init_q_matrix = preprocessor.build_q_matrix_init()

    logger.info(f"学生数: {len(student_timelines)} | 题目数: {num_problems} | 知识点数: {num_kp}")

    folds = k_fold_split(student_timelines, args.k_folds, seed=base_config.SEED)

    # 确定要运行的变体
    variants_to_run = OrderedDict()

    # 始终先跑 Full Model 作为基准
    variants_to_run['Full-Model'] = {
        'group': '-', 'desc': '完整模型（基准）',
        'config': {}, 'hook': None,
    }

    if args.variants:
        # 指定了具体变体
        for v in args.variants:
            if v in ABLATION_VARIANTS:
                variants_to_run[v] = ABLATION_VARIANTS[v]
            else:
                logger.warning(f"未知变体: {v}，跳过。可选: {list(ABLATION_VARIANTS.keys())}")
    elif args.groups:
        # 指定了组别
        for name, info in ABLATION_VARIANTS.items():
            if info['group'] in args.groups:
                variants_to_run[name] = info
    else:
        # 全部
        variants_to_run.update(ABLATION_VARIANTS)

    logger.info(f"\n将运行 {len(variants_to_run)} 个变体:")
    for name, info in variants_to_run.items():
        logger.info(f"  [{info.get('group', '-')}] {name}: {info['desc']}")

    # 运行消融实验
    all_results = OrderedDict()

    for variant_name, variant_info in variants_to_run.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"消融变体: {variant_name} — {variant_info['desc']}")
        logger.info(f"{'='*60}")

        fold_metrics_list = []
        for fold_idx, (train_tl, val_tl) in enumerate(folds):
            logger.info(f"  Fold {fold_idx + 1}/{args.k_folds}")

            # config 深拷贝和修改在 train_one_fold 内部完成
            config = copy.deepcopy(base_config)
            config.MODEL_SAVE_PATH = f"checkpoints/ablation/{variant_name}/"

            fold_metrics = train_one_fold(
                fold_idx, train_tl, val_tl, variant_name, variant_info,
                preprocessor, config, num_kp, num_problems, init_q_matrix,
            )
            fold_metrics_list.append(fold_metrics)

            auc_val = fold_metrics.get('auc', 0)
            ba_val = fold_metrics.get('balanced_accuracy', 0)
            f1_val = fold_metrics.get('f1', 0)
            logger.info(f"    Fold {fold_idx + 1} AUC={auc_val:.4f}  BA={ba_val:.4f}  F1={f1_val:.4f}")

        all_results[variant_name] = fold_metrics_list

        # 打印该变体汇总
        for key in ['auc', 'balanced_accuracy', 'f1']:
            values = [m[key] for m in fold_metrics_list if key in m and not np.isnan(m.get(key, float('nan')))]
            if values:
                logger.info(f"  {variant_name} {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # 最终汇总
    logger.info(f"\n{'='*70}")
    logger.info("消融实验汇总")
    logger.info(f"{'='*70}")
    print_ablation_table(all_results)
    print_delta_table(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument('--k-folds', type=int, default=5, help='折数 (默认 5)')
    parser.add_argument('--dataset', type=str, default='gold',
                        choices=['gold', 'standard', 'raw'], help='数据集变体')
    parser.add_argument('--groups', type=str, nargs='+', default=None,
                        choices=['A', 'B', 'C'],
                        help='消融组别: A=输入特征, B=架构组件, C=训练策略')
    parser.add_argument('--variants', type=str, nargs='+', default=None,
                        help='指定变体名称，逗号分隔')
    args = parser.parse_args()
    main(args)
