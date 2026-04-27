# MFKT — OJ 编程教育知识追踪

基于多特征融合的 Transformer 架构，利用 OJ（Online Judge）提交元数据（评判结果、得分、时间/内存用量）预测学生知识掌握状态。目标会议：ICIC2026。

## 目录结构

```
oj_kt/
├── oj_kt_project/              # 主项目代码
│   ├── config.py               # 全局超参数配置（Config dataclass）
│   ├── train_cv.py             # 主入口：K折交叉验证训练
│   ├── run_ablation.py         # 消融实验
│   ├── run_stat_test.py        # 统计显著性检验（Wilcoxon）
│   ├── hyperparam_search.py    # 超参搜索
│   ├── experiment_mastery_validation.py  # 掌握度验证实验
│   ├── train_code_dkt.py       # Code-DKT baseline（独立流程）
│   ├── data/                   # 数据预处理
│   │   ├── preprocessor.py     # OJ 数据集预处理
│   │   ├── codeworkout_preprocessor.py
│   │   ├── dataset.py          # StudentTimelineDataset
│   │   └── knowledge_structure.py
│   ├── models/                 # 模型定义
│   │   ├── model.py            # MFKT 主模型
│   │   ├── sequence_model.py   # LSTM / Transformer 序列编码器
│   │   ├── attempt_encoder.py  # 单题提交序列 GRU 编码器
│   │   ├── q_matrix.py         # 可学习 Q 矩阵
│   │   ├── baselines.py        # 16 个 baseline 模型 wrapper
│   │   └── registry.py         # 模型注册表
│   └── utils/
│       ├── trainer.py          # 多任务训练循环（含 SWA、早停）
│       ├── evaluation.py       # K折划分、训练一折
│       └── metrics.py          # AUC、balanced acc、F1、mastery MAE
├── data/                       # 数据文件
│   ├── gold/problems.json      # 主数据集题目信息
│   ├── standard/problems.json
│   ├── knowledge_points.json   # 知识点定义及先修关系
│   └── problems.json
└── knowledge-tracing-collection-pytorch/  # Baseline 模型库（本地依赖）
    └── models/                 # DKT、DKVMN、SAKT、AKT 等模型实现
```

## 环境依赖

```bash
conda activate dkt
# 或按 requirements.txt 安装
pip install -r oj_kt_project/requirements.txt
```

## 运行

所有命令从 `oj_kt_project/` 目录执行：

```bash
cd oj_kt_project

# K 折交叉验证（主要入口）
python train_cv.py --dataset-type oj --dataset gold --k-folds 5

# 只跑指定模型
python train_cv.py --dataset-type oj --models DKT,SAKT,Ours-Transformer

# CodeWorkout 数据集
python train_cv.py --dataset-type codeworkout

# 消融实验
python run_ablation.py --dataset gold --k-folds 5      # 全部消融组
python run_ablation.py --groups A                       # 仅输入特征消融
python run_ablation.py --groups B C                     # 架构 + 训练策略
python run_ablation.py --variants w/o-time w/o-KG-attention

# 统计显著性检验（MFKT vs baselines，Wilcoxon signed-rank）
python run_stat_test.py

# 超参搜索（2折网格搜索）
python hyperparam_search.py

# 掌握度验证实验
python experiment_mastery_validation.py --dataset gold
```

## 数据说明

| 数据集 | 路径 | 说明 |
|--------|------|------|
| OJ gold | `data/gold/` + `data/submissions.csv` | 主数据集，6种评判结果 |
| OJ standard | `data/standard/` | 标准划分版本 |
| CodeWorkout | `data/All/` | CSEDM 数据集，二值评判 |

> 大型 CSV（`submissions.csv`、`data/All/`）不纳入版本控制，需单独获取。

## 模型架构（MFKT）

输入编码 → 特征融合投影 → 序列模型（LSTM/Transformer）→ KG 引导注意力 → 双头输出

- **AC 预测头**：下一题是否答对（二分类，BCE + Focal Loss）
- **掌握度头**：每个知识点掌握程度（多标签回归，MSE）
- 损失：`L = λ_ac * FocalLoss + λ_mastery * MSE + λ_q * Q_reg`
