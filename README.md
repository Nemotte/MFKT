# MFKT — Multi-Feature Knowledge Tracing for Online Judge Systems

A Transformer-based knowledge tracing model that leverages OJ submission metadata (verdict types, scores, time/memory usage) to predict student knowledge states. Target venue: ICIC 2026.

## Repository Structure

```
MFKT/
├── oj_kt_project/
│   ├── config.py                        # Global hyperparameters (Config dataclass)
│   ├── train_cv.py                      # Main entry: K-fold cross-validation training
│   ├── run_ablation.py                  # Ablation study
│   ├── run_stat_test.py                 # Statistical significance test (Wilcoxon)
│   ├── hyperparam_search.py             # Hyperparameter grid search
│   ├── experiment_mastery_validation.py # Mastery validation experiment
│   ├── baseline_tuning.py               # Baseline hyperparameter tuning
│   ├── train_code_dkt.py                # Code-DKT baseline (standalone pipeline)
│   ├── data/
│   │   ├── preprocessor.py              # OJ dataset preprocessor
│   │   ├── codeworkout_preprocessor.py  # CodeWorkout/CSEDM preprocessor
│   │   ├── dataset.py                   # StudentTimelineDataset
│   │   └── knowledge_structure.py       # Prerequisite knowledge graph
│   ├── models/
│   │   ├── model.py                     # MFKT main model
│   │   ├── sequence_model.py            # LSTM / Transformer encoder
│   │   ├── attempt_encoder.py           # Per-problem submission GRU encoder
│   │   ├── q_matrix.py                  # Learnable Q-matrix
│   │   ├── baselines.py                 # 16 baseline model wrappers
│   │   └── registry.py                  # Model registry
│   └── utils/
│       ├── trainer.py                   # Multi-task training loop (SWA, early stopping)
│       ├── evaluation.py                # K-fold split, train one fold
│       └── metrics.py                   # AUC, balanced acc, F1, mastery MAE
├── data/
│   ├── gold/
│   │   ├── submissions.csv              # OJ gold dataset (primary)
│   │   └── problems.json
│   ├── standard/problems.json
│   ├── knowledge_points.json            # Knowledge point definitions and prerequisites
│   └── problems.json
└── knowledge-tracing-collection-pytorch/
    └── models/                          # Baseline implementations (DKT, DKVMN, SAKT, AKT, ...)
```

## Setup

```bash
conda activate dkt
# or
pip install -r oj_kt_project/requirements.txt
```

## Usage

All commands are run from the **repository root**.

```bash
# K-fold cross-validation (primary entry point)
python oj_kt_project/train_cv.py --dataset-type oj --dataset gold --k-folds 5

# Run specific models only
python oj_kt_project/train_cv.py --dataset-type oj --models DKT,SAKT,Ours-Transformer

# CodeWorkout dataset
python oj_kt_project/train_cv.py --dataset-type codeworkout

# Ablation study
python oj_kt_project/run_ablation.py --dataset gold --k-folds 5      # all groups
python oj_kt_project/run_ablation.py --groups A                       # input feature ablations
python oj_kt_project/run_ablation.py --groups B C                     # architecture + training
python oj_kt_project/run_ablation.py --variants w/o-time w/o-KG-attention

# Statistical significance test (MFKT vs baselines, Wilcoxon signed-rank)
python oj_kt_project/run_stat_test.py

# Hyperparameter search (2-fold grid search)
python oj_kt_project/hyperparam_search.py

# Mastery validation experiment
python oj_kt_project/experiment_mastery_validation.py --dataset gold
```

## Datasets

| Dataset | Path | Description |
|---------|------|-------------|
| OJ gold | `data/gold/` | Primary dataset, 6 verdict types (AC/WA/TLE/MLE/RE/CE) |
| OJ standard | `data/standard/` | Standard split variant |
| CodeWorkout | `data/All/` | CSEDM dataset, binary verdicts — not included, provide separately |

## Model Architecture

Input encoding → Feature fusion projection → Sequence model (LSTM/Transformer) → KG-guided attention → Dual output heads

- **AC head**: next-problem correctness prediction (binary, BCE + Focal Loss)
- **Mastery head**: per-knowledge-point mastery (multi-label regression, MSE)
- Loss: `L = λ_ac · FocalLoss + λ_mastery · MSE + λ_q · Q_reg`
