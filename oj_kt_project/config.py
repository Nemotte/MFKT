"""
配置文件 - 使用 dataclass 实现类型安全和自动验证
"""
import os
import torch
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # 数据集变体: "gold" / "standard" / "raw"
    DATASET_VARIANT: str = "gold"

    # 数据路径（由 __post_init__ 根据 DATASET_VARIANT 自动设置）
    SUBMISSION_DATA_PATH: str = "data/gold/submissions.csv"
    KNOWLEDGE_DATA_PATH: str = "data/knowledge_points.json"
    PROBLEM_DATA_PATH: str = "data/gold/problems.json"

    # 模型类型: "transformer" 或 "lstm"
    MODEL_TYPE: str = "transformer"

    # Verdict 嵌入（AC/WA/TLE/MLE/RE/CE → 6类）
    NUM_VERDICT_TYPES: int = 6
    VERDICT_EMBED_DIM: int = 16

    # 得分 + 耗时 + 内存特征编码
    SCORE_FEATURE_INPUT_DIM: int = 3       # [normalized_score, log_time, log_memory]
    SCORE_FEATURE_DIM: int = 8             # 编码后维度

    # 尝试次数嵌入
    MAX_ATTEMPTS: int = 10
    ATTEMPT_EMBED_DIM: int = 8

    # 序列模型参数
    HIDDEN_DIM: int = 32
    NUM_LAYERS: int = 1
    DROPOUT: float = 0.35

    # Transformer 参数
    NUM_HEADS: int = 2
    NUM_TRANSFORMER_LAYERS: int = 1
    FF_DIM: int = 64

    # 训练参数
    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 50
    GRAD_CLIP_NORM: float = 1.0
    WEIGHT_DECAY: float = 5e-4

    # Early Stopping
    EARLY_STOP_PATIENCE: int = 15

    # 保存路径
    MODEL_SAVE_PATH: str = "checkpoints/"
    LOG_DIR: str = "logs/"

    # 其他
    SEED: int = 42
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1

    # 序列长度
    MAX_PROBLEM_SEQ_LEN: int = 50

    # Q-Matrix
    QMATRIX_EMBED_DIM: int = 16
    QMATRIX_INIT_WEIGHT: float = 0.5
    QMATRIX_LOSS_WEIGHT: float = 0.1

    # 多任务权重
    AC_LOSS_WEIGHT: float = 1.0
    MASTERY_LOSS_WEIGHT: float = 0.2

    # 掌握度伪标签
    MASTERY_WINDOW_SIZE: int = 5

    # 时间窗口 / Session 切割
    SESSION_GAP_HOURS: float = 2.0
    SESSION_MAX_HOURS: float = 4.0
    TIME_FEATURE_INPUT_DIM: int = 4
    TIME_FEATURE_DIM: int = 16

    # 分类阈值
    AC_THRESHOLD: float = 0.5

    # Focal Loss
    USE_FOCAL_LOSS: bool = True
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTHING: float = 0.05

    # 最优阈值搜索
    SEARCH_BEST_THRESHOLD: bool = True

    # 学习率调度器
    LR_SCHEDULER_FACTOR: float = 0.5
    LR_SCHEDULER_PATIENCE: int = 7

    # 学习率 Warmup
    WARMUP_EPOCHS: int = 5
    USE_COSINE_ANNEALING: bool = True

    # 题目难度特征（从训练集统计）
    PROBLEM_DIFFICULTY_INPUT_DIM: int = 2   # [ac_rate, log_avg_attempts]
    PROBLEM_DIFFICULTY_DIM: int = 8

    # 学生能力特征（动态，序列中每步变化）
    STUDENT_FEATURE_INPUT_DIM: int = 4      # [running_ac_rate, recent_ac_rate, kp_hist_ac_delta, inter_problem_gap]
    STUDENT_FEATURE_DIM: int = 8

    # Attempt Encoder（内层提交序列编码）
    USE_ATTEMPT_ENCODER: bool = False
    ATTEMPT_ENCODER_HIDDEN: int = 16
    ATTEMPT_ENCODER_OUTPUT_DIM: int = 24  # 替换 VERDICT_EMBED_DIM(16) + ATTEMPT_EMBED_DIM(8)

    # SWA（Stochastic Weight Averaging）
    USE_SWA: bool = True
    SWA_START_FRAC: float = 0.5

    # CLRS 领域特化
    NUM_ALGO_CATEGORIES: int = 8
    ALGO_CATEGORY_DIM: int = 8
    USE_KG_ATTENTION: bool = True
    KG_RELEVANCE_TEMPERATURE: float = 1.0

    # Verdict 分布特征（session 内各 verdict 频率）
    USE_VERDICT_DIST: bool = True
    VERDICT_DIST_DIM: int = 8  # 编码后维度

    # Verdict-conditioned 特征调制（让 verdict 类型影响 score/time 的解读）
    USE_VERDICT_MODULATION: bool = True

    # V3 特征开关
    USE_STUDENT_FEATURES: bool = True
    USE_CATEGORY_FEATURES: bool = True
    USE_SCORE_FEATURES: bool = True
    USE_TIME_FEATURES: bool = True
    USE_QMATRIX_EMBED: bool = True
    USE_ATTEMPT_EMBED: bool = True

    # 设备
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def validate(self):
        """验证配置参数的一致性"""
        assert self.MODEL_TYPE in ("transformer", "lstm"), \
            f"MODEL_TYPE 必须是 'transformer' 或 'lstm'，实际: {self.MODEL_TYPE}"
        assert 0 < self.TRAIN_RATIO + self.VAL_RATIO + self.TEST_RATIO <= 1.0 + 1e-6
        assert self.HIDDEN_DIM % self.NUM_HEADS == 0, \
            f"HIDDEN_DIM ({self.HIDDEN_DIM}) 必须能被 NUM_HEADS ({self.NUM_HEADS}) 整除"
        assert 0 < self.DROPOUT < 1
        logger.info("配置验证通过")

    def __post_init__(self):
        # 根据 DATASET_VARIANT 自动设置数据路径
        base = os.path.dirname(os.path.abspath(__file__))
        if self.DATASET_VARIANT == "raw":
            self.SUBMISSION_DATA_PATH = os.path.join(base, '..', 'data', 'submissions.csv')
            self.PROBLEM_DATA_PATH = os.path.join(base, '..', 'data', 'problems.json')
        elif self.DATASET_VARIANT in ("gold", "standard"):
            self.SUBMISSION_DATA_PATH = os.path.join(base, '..', 'data', self.DATASET_VARIANT, 'submissions.csv')
            self.PROBLEM_DATA_PATH = os.path.join(base, '..', 'data', self.DATASET_VARIANT, 'problems.json')
        else:
            raise ValueError(f"未知 DATASET_VARIANT: {self.DATASET_VARIANT}，可选: gold/standard/raw")
        self.KNOWLEDGE_DATA_PATH = os.path.join(base, '..', 'data', 'knowledge_points.json')
        self.validate()
