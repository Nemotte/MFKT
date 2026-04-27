"""
统一的模型注册表

将 train_baselines_cv.py 和 train_codeworkout_cv.py 中分散的 MODEL_REGISTRY 合并到此处。
通过 get_registry() 按数据集类型获取对应注册表。
"""
from models.baselines import (
    VanillaDKTBaseline, DKTBaseline, DKTPlusBaseline, DKVMNBaseline,
    SAKTBaseline, GKTBaseline, SAINTBaseline, DKTWithFeaturesBaseline,
    AKTBaseline, SimpleKTBaseline,
    qDKTBaseline, LPKTBaseline, CL4KTBaseline, ATKTBaseline,
    CMKTBaseline, MambaKTBaseline,
)

# ── Vanilla DKT 的训练策略覆盖 ──
_VANILLA_CONFIG = {
    'USE_SWA': False,
    'USE_COSINE_ANNEALING': False,
    'WARMUP_EPOCHS': 0,
    'WEIGHT_DECAY': 0.0,
    'SEARCH_BEST_THRESHOLD': False,
    'USE_FOCAL_LOSS': False,
    'LABEL_SMOOTHING': 0.0,
    'LEARNING_RATE': 0.001,
    'NUM_EPOCHS': 100,
    'EARLY_STOP_PATIENCE': 10,
}

# ── V3 基础 config (禁用所有额外特征/损失) ──
_V3_BASE = {
    'MODEL_TYPE': 'transformer',
    'USE_STUDENT_FEATURES': False,
    'USE_CATEGORY_FEATURES': False,
    'USE_SCORE_FEATURES': False,
    'USE_TIME_FEATURES': False,
    'USE_QMATRIX_EMBED': False,
    'USE_ATTEMPT_EMBED': False,
    'USE_KG_ATTENTION': False,
    'USE_FOCAL_LOSS': False,
    'QMATRIX_LOSS_WEIGHT': 0.0,
    'MASTERY_LOSS_WEIGHT': 0.0,
    'LABEL_SMOOTHING': 0.0,
}

# ========================================================================
# OJ 数据集模型注册表
# ========================================================================
OJ_MODEL_REGISTRY = {
    # ── Ours 模型变体 ──
    'Ours-Transformer': {'config_override': {'MODEL_TYPE': 'transformer'}},
    'Ours-Transformer-V1': {'config_override': {'MODEL_TYPE': 'transformer', 'STUDENT_FEATURE_INPUT_DIM': 2}},
    'Ours-Transformer-V3': {'config_override': {**_V3_BASE}},
    'Ours-Transformer-V3+score': {
        'config_override': {**_V3_BASE, 'USE_SCORE_FEATURES': True},
    },
    'Ours-Transformer-V3+score+focal': {
        'config_override': {**_V3_BASE, 'USE_SCORE_FEATURES': True, 'USE_FOCAL_LOSS': True},
    },
    'Ours-Transformer-V3+score+tuned': {
        'config_override': {
            **_V3_BASE, 'USE_SCORE_FEATURES': True,
            'DROPOUT': 0.2, 'HIDDEN_DIM': 64, 'FF_DIM': 128,
        },
    },
    'Ours-LSTM': {'config_override': {'MODEL_TYPE': 'lstm'}},
    'Ours-LSTM-V3': {
        'config_override': {**_V3_BASE, 'MODEL_TYPE': 'lstm'},
    },
    'Ours-LSTM-V3+score': {
        'config_override': {**_V3_BASE, 'MODEL_TYPE': 'lstm', 'USE_SCORE_FEATURES': True},
    },
    # ── Baselines ──
    'DKT-Vanilla': {
        'type': 'baseline',
        'class': VanillaDKTBaseline,
        'params': {'emb_size': 64, 'hidden_size': 64, 'dropout_rate': 0.2},
        'vanilla': True,
        'config_override': _VANILLA_CONFIG,
    },
    'DKT': {
        'type': 'baseline',
        'class': DKTBaseline,
        'params': {'emb_size': 64, 'hidden_size': 64, 'dropout_rate': 0.2},
    },
    'DKT-Large': {
        'type': 'baseline',
        'class': DKTBaseline,
        'params': {'emb_size': 128, 'hidden_size': 128, 'dropout_rate': 0.2},
    },
    'DKT+': {
        'type': 'baseline',
        'class': DKTPlusBaseline,
        'params': {'emb_size': 64, 'hidden_size': 64,
                   'lambda_r': 0.01, 'lambda_w1': 0.03, 'lambda_w2': 0.3},
    },
    'DKT+Features': {
        'type': 'baseline',
        'class': DKTWithFeaturesBaseline,
        'params': {'hidden_size': 64, 'dropout_rate': 0.2,
                   'num_verdict_types': 6, 'verdict_embed_dim': 16,
                   'score_input_dim': 3, 'score_dim': 8,
                   'time_input_dim': 4, 'time_dim': 16,
                   'difficulty_input_dim': 2, 'difficulty_dim': 8,
                   'num_categories': 8, 'category_dim': 8,
                   'verdict_dist_dim': 8},
    },
    'DKVMN': {
        'type': 'baseline',
        'class': DKVMNBaseline,
        'params': {'dim_s': 50, 'size_m': 20},
    },
    'SAKT': {
        'type': 'baseline',
        'class': SAKTBaseline,
        'params': {'n': 50, 'd': 64, 'num_attn_heads': 4, 'dropout': 0.2},
    },
    'GKT': {
        'type': 'baseline',
        'class': GKTBaseline,
        'params': {'hidden_size': 30, 'num_attn_heads': 2},
    },
    'SAINT': {
        'type': 'baseline',
        'class': SAINTBaseline,
        'params': {'n': 50, 'd': 64, 'num_attn_heads': 4, 'dropout': 0.2, 'num_tr_layers': 1},
    },
    'AKT': {
        'type': 'baseline',
        'class': AKTBaseline,
        'params': {'d': 64, 'num_heads': 4, 'dropout': 0.2},
    },
    'simpleKT': {
        'type': 'baseline',
        'class': SimpleKTBaseline,
        'params': {'d': 64, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.2},
    },
    # ── Recent Baselines (2022-2024) ──
    'qDKT': {
        'type': 'baseline',
        'class': qDKTBaseline,
        'params': {'emb_size': 64, 'hidden_size': 64, 'dropout_rate': 0.2},
    },
    'LPKT': {
        'type': 'baseline',
        'class': LPKTBaseline,
        'params': {'d': 64, 'dropout': 0.2},
    },
    'CL4KT': {
        'type': 'baseline',
        'class': CL4KTBaseline,
        'params': {'emb_size': 64, 'hidden_size': 64, 'dropout_rate': 0.2},
    },
    'ATKT': {
        'type': 'baseline',
        'class': ATKTBaseline,
        'params': {'emb_size': 64, 'hidden_size': 64, 'dropout_rate': 0.2},
    },
    # ── Latest Baselines (2024-2025) ──
    'CMKT': {
        'type': 'baseline',
        'class': CMKTBaseline,
        'params': {'d': 64, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.2},
    },
    'MambaKT': {
        'type': 'baseline',
        'class': MambaKTBaseline,
        'params': {'emb_size': 64, 'd_model': 64, 'd_state': 16, 'num_layers': 2, 'dropout': 0.2},
    },
    # ── Tuned Baselines (d=128，验证排名稳定性，回应 W5) ──
    # 数据集 158 学生，d=128 已是 unified protocol 的 2 倍，再大会过拟合
    'SAINT-Tuned': {
        'type': 'baseline',
        'class': SAINTBaseline,
        'params': {'n': 50, 'd': 128, 'num_attn_heads': 4, 'dropout': 0.2, 'num_tr_layers': 2},
    },
    'AKT-Tuned': {
        'type': 'baseline',
        'class': AKTBaseline,
        'params': {'d': 128, 'num_heads': 4, 'dropout': 0.2},
    },
    'simpleKT-Tuned': {
        'type': 'baseline',
        'class': SimpleKTBaseline,
        'params': {'d': 128, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.2},
    },
    'SAKT-Tuned': {
        'type': 'baseline',
        'class': SAKTBaseline,
        'params': {'n': 50, 'd': 128, 'num_attn_heads': 8, 'dropout': 0.2},
    },
    'CMKT-Tuned': {
        'type': 'baseline',
        'class': CMKTBaseline,
        'params': {'d': 128, 'num_heads': 8, 'num_layers': 2, 'dropout': 0.2},
    },
    # ── BCE Baselines（回应 M2：Focal Loss 偏差质疑）──
    'DKT-BCE': {
        'type': 'baseline',
        'class': DKTBaseline,
        'params': {'emb_size': 64, 'hidden_size': 64, 'dropout_rate': 0.2},
        'config_override': {'USE_FOCAL_LOSS': False, 'LABEL_SMOOTHING': 0.0},
    },
    'SAINT-BCE': {
        'type': 'baseline',
        'class': SAINTBaseline,
        'params': {'n': 50, 'd': 64, 'num_attn_heads': 4, 'dropout': 0.2, 'num_tr_layers': 1},
        'config_override': {'USE_FOCAL_LOSS': False, 'LABEL_SMOOTHING': 0.0},
    },
    'simpleKT-BCE': {
        'type': 'baseline',
        'class': SimpleKTBaseline,
        'params': {'d': 64, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.2},
        'config_override': {'USE_FOCAL_LOSS': False, 'LABEL_SMOOTHING': 0.0},
    },
}

# ========================================================================
# CodeWorkout 数据集模型注册表
# ========================================================================
CODEWORKOUT_MODEL_REGISTRY = {
    'DKT-Vanilla': OJ_MODEL_REGISTRY['DKT-Vanilla'],
    'DKT': OJ_MODEL_REGISTRY['DKT'],
    'DKT+': OJ_MODEL_REGISTRY['DKT+'],
    'DKT+Features': OJ_MODEL_REGISTRY['DKT+Features'],
    'DKVMN': OJ_MODEL_REGISTRY['DKVMN'],
    'SAKT': OJ_MODEL_REGISTRY['SAKT'],
    'GKT': OJ_MODEL_REGISTRY['GKT'],
    'SAINT': OJ_MODEL_REGISTRY['SAINT'],
    'AKT': OJ_MODEL_REGISTRY['AKT'],
    'simpleKT': OJ_MODEL_REGISTRY['simpleKT'],
    'qDKT': OJ_MODEL_REGISTRY['qDKT'],
    'LPKT': OJ_MODEL_REGISTRY['LPKT'],
    'CL4KT': OJ_MODEL_REGISTRY['CL4KT'],
    'ATKT': OJ_MODEL_REGISTRY['ATKT'],
    'CMKT': OJ_MODEL_REGISTRY['CMKT'],
    'MambaKT': OJ_MODEL_REGISTRY['MambaKT'],
    'Ours-Transformer': {'config_override': {'MODEL_TYPE': 'transformer'}},
    'Ours-Transformer-V1': {'config_override': {'MODEL_TYPE': 'transformer', 'STUDENT_FEATURE_INPUT_DIM': 2}},
    'Ours-Transformer-V3': {'config_override': {**_V3_BASE}},
    'Ours-Transformer-V3+score': {
        'config_override': {**_V3_BASE, 'USE_SCORE_FEATURES': True},
    },
    'Ours-LSTM-V3+score': {
        'config_override': {**_V3_BASE, 'MODEL_TYPE': 'lstm', 'USE_SCORE_FEATURES': True},
    },
}

# ========================================================================
# 统计检验使用的模型注册表 (run_stat_test.py)
# ========================================================================
STAT_TEST_MODEL_REGISTRY = {
    'MFKT': {},
    'DKT': OJ_MODEL_REGISTRY['DKT'],
    'DKT+': OJ_MODEL_REGISTRY['DKT+'],
    'DKVMN': OJ_MODEL_REGISTRY['DKVMN'],
    'SAKT': OJ_MODEL_REGISTRY['SAKT'],
    'SAINT': OJ_MODEL_REGISTRY['SAINT'],
    'AKT': OJ_MODEL_REGISTRY['AKT'],
    'simpleKT': OJ_MODEL_REGISTRY['simpleKT'],
    'ATKT': OJ_MODEL_REGISTRY['ATKT'],
    'CL4KT': OJ_MODEL_REGISTRY['CL4KT'],
    'CMKT': OJ_MODEL_REGISTRY['CMKT'],
    # Tuned baselines (W5: 验证排名稳定性)
    'SAINT-Tuned': OJ_MODEL_REGISTRY['SAINT-Tuned'],
    'AKT-Tuned': OJ_MODEL_REGISTRY['AKT-Tuned'],
    'simpleKT-Tuned': OJ_MODEL_REGISTRY['simpleKT-Tuned'],
    'SAKT-Tuned': OJ_MODEL_REGISTRY['SAKT-Tuned'],
    'CMKT-Tuned': OJ_MODEL_REGISTRY['CMKT-Tuned'],
    # BCE baselines (M2: 验证 Focal Loss 不造成人为偏差)
    'DKT-BCE': OJ_MODEL_REGISTRY['DKT-BCE'],
    'SAINT-BCE': OJ_MODEL_REGISTRY['SAINT-BCE'],
    'simpleKT-BCE': OJ_MODEL_REGISTRY['simpleKT-BCE'],
}


def get_registry(dataset_type='oj'):
    """按数据集类型获取模型注册表"""
    if dataset_type == 'codeworkout':
        return CODEWORKOUT_MODEL_REGISTRY
    elif dataset_type == 'stat_test':
        return STAT_TEST_MODEL_REGISTRY
    return OJ_MODEL_REGISTRY
