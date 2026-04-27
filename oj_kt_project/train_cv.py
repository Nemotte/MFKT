"""
K 折交叉验证训练入口（统一 OJ / CodeWorkout）

用法:
  python train_cv.py --dataset-type oj --models DKT,SAKT
  python train_cv.py --dataset-type codeworkout --data-dir ../data/All
"""
import numpy as np
import os
import logging
import argparse

from config import Config
from models.registry import get_registry
from utils.evaluation import set_seed, k_fold_split
from utils.evaluation import train_one_fold, print_comparison_table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def _load_oj(args):
    """加载 OJ 数据集"""
    from data.preprocessor import OJDataPreprocessor

    config = Config(DATASET_VARIANT=args.dataset)
    preprocessor = OJDataPreprocessor(config)

    submissions, knowledge_data, problem_data = preprocessor.load_data()
    logger.info(f"  加载了 {len(submissions)} 条提交记录, {len(problem_data)} 个题目")

    preprocessor.build_vocabularies(knowledge_data, problem_data, submissions=submissions)
    num_kp = len(preprocessor.knowledge_to_idx)
    num_problems = len(preprocessor.problem_to_idx)
    student_timelines = preprocessor.create_student_timelines(submissions)
    init_q_matrix = preprocessor.build_q_matrix_init()

    return preprocessor, config, student_timelines, num_problems, num_kp, init_q_matrix


def _load_codeworkout(args):
    """加载 CodeWorkout 数据集"""
    from data.codeworkout_preprocessor import CodeWorkoutPreprocessor

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'All')

    preprocessor = CodeWorkoutPreprocessor(data_dir)
    student_timelines, num_problems, num_kp, init_q_matrix = preprocessor.load_and_build()

    config = Config()
    config.NUM_VERDICT_TYPES = 2   # CodeWorkout 只有 AC/WA
    config.NUM_ALGO_CATEGORIES = 1  # 无类别

    return preprocessor, config, student_timelines, num_problems, num_kp, init_q_matrix


def main(args):
    seed = 42
    set_seed(seed)
    k = args.k_folds
    dataset_type = args.dataset_type

    # 加载数据
    logger.info("=" * 70)
    logger.info(f"K 折交叉验证 — {k} 折 | 数据集类型: {dataset_type}")
    logger.info("=" * 70)

    logger.info("[1/3] 加载和预处理数据...")
    if dataset_type == 'codeworkout':
        preprocessor, config, student_timelines, num_problems, num_kp, init_q_matrix = _load_codeworkout(args)
    else:
        preprocessor, config, student_timelines, num_problems, num_kp, init_q_matrix = _load_oj(args)

    config.SEED = seed
    logger.info(f"  学生数: {len(student_timelines)}, 题目数: {num_problems}, 知识点数: {num_kp}")

    # 模型注册表
    MODEL_REGISTRY = get_registry(dataset_type)

    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
    else:
        model_names = list(MODEL_REGISTRY.keys())

    logger.info(f"模型: {', '.join(model_names)}")

    # K-fold split
    logger.info(f"[2/3] 按学生进行 {k} 折划分...")
    folds = k_fold_split(student_timelines, k, seed=config.SEED)

    # Train
    logger.info("[3/3] 开始训练...")
    all_results = {}

    checkpoint_prefix = f"checkpoints/{dataset_type}" if dataset_type == 'codeworkout' else "checkpoints"

    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"  未知模型: {model_name}, 跳过")
            continue

        model_info = MODEL_REGISTRY[model_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"模型: {model_name}")
        logger.info(f"{'='*60}")

        fold_metrics_list = []
        for fold_idx, (train_tl, val_tl) in enumerate(folds):
            set_seed(config.SEED + fold_idx)
            logger.info(f"\n  Fold {fold_idx + 1}/{k}")

            config.MODEL_SAVE_PATH = f"{checkpoint_prefix}/{model_name}/"

            fold_metrics = train_one_fold(
                fold_idx, train_tl, val_tl, model_name, model_info,
                preprocessor, config, num_kp, num_problems, init_q_matrix,
            )

            fold_metrics_list.append(fold_metrics)
            logger.info(f"    Fold {fold_idx + 1} AUC={fold_metrics.get('auc', 0):.4f}  "
                         f"BA={fold_metrics.get('balanced_accuracy', 0):.4f}  "
                         f"F1={fold_metrics.get('f1', 0):.4f}")

        all_results[model_name] = fold_metrics_list

        for key in ['auc', 'balanced_accuracy', 'f1']:
            values = [m[key] for m in fold_metrics_list
                      if key in m and not np.isnan(m[key])]
            if values:
                logger.info(f"  {model_name} {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Comparison table
    logger.info("\n" + "=" * 70)
    logger.info(f"{dataset_type.upper()} 对比实验汇总")
    logger.info("=" * 70)
    print_comparison_table(all_results, model_col_width=20, save_csv=args.save_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K 折交叉验证训练（OJ / CodeWorkout）")
    parser.add_argument('--dataset-type', type=str, default='oj',
                        choices=['oj', 'codeworkout'],
                        help='数据集类型 (默认 oj)')
    parser.add_argument('--k-folds', type=int, default=5, help='折数 (默认 5)')
    parser.add_argument('--models', type=str, default=None,
                        help='模型，逗号分隔 (默认全部)')
    # OJ-specific
    parser.add_argument('--dataset', type=str, default='gold',
                        choices=['gold', 'standard', 'raw'],
                        help='OJ 数据集变体 (默认 gold)')
    # CodeWorkout-specific
    parser.add_argument('--data-dir', type=str, default=None,
                        help='CodeWorkout 数据目录 (默认 data/All/)')
    parser.add_argument('--save-csv', type=str, default=None,
                        help='将汇总结果保存为 CSV 文件（可选）')
    args = parser.parse_args()
    main(args)
