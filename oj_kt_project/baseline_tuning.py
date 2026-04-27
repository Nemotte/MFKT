"""
Baseline 独立超参调优实验 — 回应评审 2 的 Priority Revision 2

流程:
  1) 对每个 baseline，在 2-fold CV 上跑精简网格 (容量 × dropout × lr)
  2) 选择 2-fold 平均 AUC 最高的组合作为该 baseline 的 tuned 配置
  3) 用 tuned 配置在完整 5-fold CV 上重新训练，记录 mean ± std
  4) 与 unified-protocol 5-fold 结果对比，给出 ΔAUC 与配对 Wilcoxon p

输出:
  - logs/baseline_tuning.log               搜索过程详细日志
  - baseline_tuning_grid.csv               每个 baseline × grid 组合的 2-fold AUC
  - baseline_tuning_results.csv            tuned vs unified 5-fold 对比 (即 Table A1)

用法:
  python baseline_tuning.py                        # 全部 8 个 baseline
  python baseline_tuning.py --models DKT,AKT       # 仅指定模型
  python baseline_tuning.py --grid quick           # 仅 2 组（冒烟测试）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import copy
import csv
import itertools
import logging
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.preprocessor import OJDataPreprocessor
from models.registry import OJ_MODEL_REGISTRY
from utils.evaluation import set_seed, k_fold_split, train_one_fold


# ----------------------------------------------------------------------
# 网格定义 — 每个 baseline 把 (capacity, dropout, lr) 映射到自己的参数名
# ----------------------------------------------------------------------
def _grid_full():
    return list(itertools.product([64, 128], [0.1, 0.3], [1e-3, 3e-3]))


def _grid_quick():
    return [(64, 0.2, 1e-3), (128, 0.2, 3e-3)]


def _build_param_dict(model_name, capacity, dropout, lr):
    """把 (capacity, dropout, lr) 翻译成 OJ_MODEL_REGISTRY 条目能消化的 params + config_override

    返回 (params_override, config_override)
    """
    cfg_ovr = {'LEARNING_RATE': lr}

    # 各 baseline 的构造器参数名不同，按类分发
    if model_name in ('DKT', 'DKT-Vanilla'):
        params = {'emb_size': capacity, 'hidden_size': capacity, 'dropout_rate': dropout}
    elif model_name == 'DKT+':
        # DKT+ 多了几个正则系数，保持原值
        params = {'emb_size': capacity, 'hidden_size': capacity,
                  'lambda_r': 0.01, 'lambda_w1': 0.03, 'lambda_w2': 0.3}
    elif model_name == 'SAKT':
        # SAKT 头数随 d 走，d=64→4 头, d=128→8 头
        heads = 4 if capacity == 64 else 8
        params = {'n': 50, 'd': capacity, 'num_attn_heads': heads, 'dropout': dropout}
    elif model_name == 'AKT':
        heads = 4 if capacity == 64 else 8
        params = {'d': capacity, 'num_heads': heads, 'dropout': dropout}
    elif model_name == 'simpleKT':
        heads = 4 if capacity == 64 else 8
        params = {'d': capacity, 'num_heads': heads, 'num_layers': 2, 'dropout': dropout}
    elif model_name == 'CMKT':
        heads = 4 if capacity == 64 else 8
        params = {'d': capacity, 'num_heads': heads, 'num_layers': 2, 'dropout': dropout}
    elif model_name == 'ATKT':
        params = {'emb_size': capacity, 'hidden_size': capacity, 'dropout_rate': dropout}
    elif model_name == 'CL4KT':
        params = {'emb_size': capacity, 'hidden_size': capacity, 'dropout_rate': dropout}
    elif model_name == 'DKVMN':
        # DKVMN 用 dim_s 不是 emb_size; size_m 固定
        params = {'dim_s': capacity, 'size_m': 20}
    elif model_name == 'SAINT':
        heads = 4 if capacity == 64 else 8
        params = {'n': 50, 'd': capacity, 'num_attn_heads': heads,
                  'dropout': dropout, 'num_tr_layers': 1}
    else:
        raise ValueError(f'未支持的 baseline: {model_name}')

    return params, cfg_ovr


def _make_model_info(model_name, capacity, dropout, lr):
    """从 OJ_MODEL_REGISTRY 复制基础条目，覆盖 params/config_override"""
    base = copy.deepcopy(OJ_MODEL_REGISTRY[model_name])
    params, cfg_ovr = _build_param_dict(model_name, capacity, dropout, lr)
    base['params'] = params
    base.setdefault('config_override', {}).update(cfg_ovr)
    return base


# ----------------------------------------------------------------------
# 评估单组 (baseline, params)
# ----------------------------------------------------------------------
def evaluate_one_combo(model_name, model_info, folds, preprocessor,
                       config, num_kp, num_problems, init_q_matrix, seed=42):
    """在给定 folds 上评估一组超参，返回每折 AUC 列表"""
    aucs = []
    for fold_idx, (train_tl, val_tl) in enumerate(folds):
        set_seed(seed + fold_idx)
        m = train_one_fold(fold_idx, train_tl, val_tl, model_name, model_info,
                           preprocessor, config, num_kp, num_problems, init_q_matrix)
        aucs.append(m['auc'])
    return aucs


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
DEFAULT_MODELS = ['DKT', 'DKT+', 'SAKT', 'AKT', 'simpleKT', 'CMKT', 'ATKT', 'CL4KT']


def main(args):
    os.makedirs('logs', exist_ok=True)
    log_path = 'logs/baseline_tuning.log'
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [file_handler, stream_handler]
    # 抑制下游 trainer 的 INFO（每 epoch 一行太吵）
    logging.getLogger('utils.trainer').setLevel(logging.WARNING)
    logging.getLogger('data.preprocessor').setLevel(logging.WARNING)
    log = logging.getLogger('baseline_tuning')

    # ── 加载数据（全程只做一次） ──
    log.info('=' * 70)
    log.info('Baseline 超参调优实验 (gold dataset)')
    log.info('=' * 70)

    config = Config(DATASET_VARIANT=args.dataset)
    preprocessor = OJDataPreprocessor(config)
    subs, kd, pd_ = preprocessor.load_data()
    preprocessor.build_vocabularies(kd, pd_, subs)
    timelines = preprocessor.create_student_timelines(subs)
    init_q_matrix = preprocessor.build_q_matrix_init()
    num_kp = len(preprocessor.knowledge_to_idx)
    num_problems = len(preprocessor.problem_to_idx)
    log.info(f'数据: {len(timelines)} 学生, {num_problems} 题, {num_kp} KP')

    folds_search = k_fold_split(timelines, k=2, seed=42)
    folds_eval = k_fold_split(timelines, k=5, seed=42)

    grid = _grid_quick() if args.grid == 'quick' else _grid_full()
    models = args.models.split(',') if args.models else DEFAULT_MODELS
    log.info(f'网格 {len(grid)} 组 × {len(models)} 模型 = {len(grid)*len(models)} 组合 (2-fold)')
    log.info(f'之后 {len(models)} 模型 × 5-fold = {len(models)*5} 训练 (eval)')

    grid_rows = []     # 写 baseline_tuning_grid.csv
    final_rows = []    # 写 baseline_tuning_results.csv

    for model_name in models:
        log.info(f'\n{"="*70}')
        log.info(f'>> {model_name}')
        log.info(f'{"="*70}')

        # ── Step 1: unified protocol (注册表默认) 在 5-fold 上的 AUC ──
        log.info(f'[1/3] Unified protocol 5-fold ...')
        t0 = time.time()
        unified_aucs = evaluate_one_combo(
            model_name, OJ_MODEL_REGISTRY[model_name], folds_eval,
            preprocessor, config, num_kp, num_problems, init_q_matrix,
        )
        log.info(f'    Unified 5-fold AUC = {np.mean(unified_aucs):.4f} ± {np.std(unified_aucs):.4f}'
                 f'  ({time.time()-t0:.0f}s)')

        # ── Step 2: 在 2-fold 上跑网格 ──
        log.info(f'[2/3] Grid search ({len(grid)} 组) on 2-fold ...')
        best_auc = -1
        best_combo = None
        for ci, (cap, dr, lr) in enumerate(grid, 1):
            t1 = time.time()
            try:
                info = _make_model_info(model_name, cap, dr, lr)
                aucs = evaluate_one_combo(
                    model_name, info, folds_search,
                    preprocessor, config, num_kp, num_problems, init_q_matrix,
                )
                mean_auc = float(np.mean(aucs))
                grid_rows.append({
                    'model': model_name, 'capacity': cap, 'dropout': dr, 'lr': lr,
                    'fold0_auc': aucs[0], 'fold1_auc': aucs[1], 'mean_auc': mean_auc,
                })
                log.info(f'    [{ci}/{len(grid)}] cap={cap} dr={dr} lr={lr:.0e} '
                         f'-> AUC={mean_auc:.4f}  ({time.time()-t1:.0f}s)')
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_combo = (cap, dr, lr)
            except Exception as e:
                log.error(f'    [{ci}/{len(grid)}] cap={cap} dr={dr} lr={lr:.0e} FAILED: {e}')
                grid_rows.append({
                    'model': model_name, 'capacity': cap, 'dropout': dr, 'lr': lr,
                    'fold0_auc': float('nan'), 'fold1_auc': float('nan'),
                    'mean_auc': float('nan'),
                })

        if best_combo is None:
            log.warning(f'    {model_name} 所有组合失败，跳过 5-fold eval')
            continue
        log.info(f'    Best: cap={best_combo[0]} dr={best_combo[1]} lr={best_combo[2]:.0e}'
                 f'  (2-fold AUC={best_auc:.4f})')

        # ── Step 3: tuned 配置在 5-fold 上重训 ──
        log.info(f'[3/3] Tuned 5-fold eval ...')
        t2 = time.time()
        tuned_info = _make_model_info(model_name, *best_combo)
        tuned_aucs = evaluate_one_combo(
            model_name, tuned_info, folds_eval,
            preprocessor, config, num_kp, num_problems, init_q_matrix,
        )
        log.info(f'    Tuned 5-fold AUC = {np.mean(tuned_aucs):.4f} ± {np.std(tuned_aucs):.4f}'
                 f'  ({time.time()-t2:.0f}s)')

        # ── 配对 Wilcoxon (n=5 折，per-fold 配对) ──
        try:
            stat, pval = wilcoxon(tuned_aucs, unified_aucs)
            wilcoxon_p = float(pval)
        except Exception as e:
            log.warning(f'    Wilcoxon 失败: {e}')
            wilcoxon_p = float('nan')

        delta = float(np.mean(tuned_aucs) - np.mean(unified_aucs))
        final_rows.append({
            'model': model_name,
            'unified_auc_mean': float(np.mean(unified_aucs)),
            'unified_auc_std': float(np.std(unified_aucs)),
            'unified_aucs': ';'.join(f'{x:.4f}' for x in unified_aucs),
            'tuned_capacity': best_combo[0],
            'tuned_dropout': best_combo[1],
            'tuned_lr': best_combo[2],
            'tuned_auc_mean': float(np.mean(tuned_aucs)),
            'tuned_auc_std': float(np.std(tuned_aucs)),
            'tuned_aucs': ';'.join(f'{x:.4f}' for x in tuned_aucs),
            'delta_auc': delta,
            'wilcoxon_p': wilcoxon_p,
        })
        log.info(f'    Δ AUC = {delta:+.4f}  Wilcoxon p = {wilcoxon_p:.3f}')

        # 每完成一个 baseline 就 flush 一次 CSV，避免中途崩溃丢数据
        _write_csv('baseline_tuning_grid.csv', grid_rows,
                   ['model', 'capacity', 'dropout', 'lr',
                    'fold0_auc', 'fold1_auc', 'mean_auc'])
        _write_csv('baseline_tuning_results.csv', final_rows,
                   ['model',
                    'unified_auc_mean', 'unified_auc_std', 'unified_aucs',
                    'tuned_capacity', 'tuned_dropout', 'tuned_lr',
                    'tuned_auc_mean', 'tuned_auc_std', 'tuned_aucs',
                    'delta_auc', 'wilcoxon_p'])

    log.info(f'\n{"="*70}')
    log.info('完成')
    log.info(f'{"="*70}')
    log.info(f'  logs/baseline_tuning.log')
    log.info(f'  baseline_tuning_grid.csv      ({len(grid_rows)} 行)')
    log.info(f'  baseline_tuning_results.csv   ({len(final_rows)} 行)')


def _write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='gold', choices=['gold', 'standard', 'raw'])
    p.add_argument('--models', default=None,
                   help='逗号分隔，默认 ' + ','.join(DEFAULT_MODELS))
    p.add_argument('--grid', default='full', choices=['full', 'quick'],
                   help='quick=2 组（冒烟），full=8 组')
    args = p.parse_args()
    main(args)
