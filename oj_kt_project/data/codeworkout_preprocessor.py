"""
CodeWorkout (CSEDM) 数据集预处理器

将 early.csv + late.csv 转换为与 StudentTimelineDataset 兼容的 student_data 格式。
"""
import pandas as pd
import numpy as np
import os
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CodeWorkoutPreprocessor:
    """
    CodeWorkout 数据适配器

    提供与 OJDataPreprocessor 兼容的接口，使 StudentTimelineDataset 可以直接使用。
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.problem_to_idx = {}
        self.knowledge_to_idx = {}  # StudentTimelineDataset 通过此属性获取 num_kp

    def load_and_build(self):
        """加载 CSV 并构建 student_timelines + 元数据"""
        # 加载 early + late
        dfs = []
        for fname in ['early.csv', 'late.csv']:
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info(f"  加载 {fname}: {len(df)} 条记录")
                dfs.append(df)
        if not dfs:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到 early.csv 或 late.csv")

        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"  合并后: {len(df)} 条记录, "
                     f"{df['SubjectID'].nunique()} 学生, "
                     f"{df['ProblemID'].nunique()} 题")

        # 构建题目索引
        unique_problems = sorted(df['ProblemID'].unique())
        self.problem_to_idx = {pid: idx for idx, pid in enumerate(unique_problems)}
        num_problems = len(unique_problems)

        # CodeWorkout 无知识点，用题目本身作为知识点（1:1 映射）
        self.knowledge_to_idx = {i: i for i in range(num_problems)}
        num_kp = num_problems

        # Q-Matrix: 单位矩阵
        init_q_matrix = np.eye(num_kp, dtype=np.float32)

        # 构建学生时间线
        student_timelines = self._build_timelines(df, num_kp)

        logger.info(f"  学生数: {len(student_timelines)}, 题目数: {num_problems}, 知识点数: {num_kp}")

        return student_timelines, num_problems, num_kp, init_q_matrix

    def _build_timelines(self, df, num_kp):
        """按学生构建时间线，按 AssignmentID + ProblemID 排序"""
        student_timelines = {}

        for subject_id, group in df.groupby('SubjectID'):
            # 按 AssignmentID 排序（作为时间代理），同 Assignment 内按 ProblemID 排序
            group = group.sort_values(['AssignmentID', 'ProblemID'])

            timeline = []
            for _, row in group.iterrows():
                pid_idx = self.problem_to_idx[row['ProblemID']]
                is_ac = bool(row['Label'])
                attempts = int(row['Attempts'])

                # knowledge_vec: one-hot
                kv = np.zeros(num_kp, dtype=np.float32)
                kv[pid_idx] = 1.0

                # verdict_dist: CodeWorkout 只有 binary (AC/WA)
                verdict_dist = np.zeros(2, dtype=np.float32)
                verdict_dist[0 if is_ac else 1] = 1.0

                timeline.append({
                    'problem_idx': pid_idx,
                    'verdict_type': 0 if is_ac else 1,  # 0=AC, 1=WA
                    'attempt_count': min(attempts, 10),
                    'score_features': np.array([
                        float(is_ac),           # normalized_score 代理
                        np.log1p(attempts),     # log_time 代理
                        0.0,                    # log_memory 占位
                    ], dtype=np.float32),
                    'time_features': np.zeros(4, dtype=np.float32),  # 无时间信息
                    'knowledge_vec': kv,
                    'session_ac': is_ac,
                    'first_ac': is_ac and attempts == 1,
                    'total_submissions': attempts,
                    'problem_category': 0,  # 无类别
                    'verdict_dist': verdict_dist,
                })

            if len(timeline) >= 2:
                student_timelines[subject_id] = timeline

        return student_timelines

    def compute_problem_difficulty(self, student_timelines):
        """从训练集统计题目难度 [ac_rate, log1p(avg_attempts)]"""
        problem_stats = defaultdict(lambda: {'ac': 0, 'total': 0, 'attempts_sum': 0})

        for timeline in student_timelines.values():
            for attempt in timeline:
                pid = attempt['problem_idx']
                problem_stats[pid]['total'] += 1
                problem_stats[pid]['attempts_sum'] += attempt['attempt_count']
                if attempt['session_ac']:
                    problem_stats[pid]['ac'] += 1

        difficulty = {}
        for pid, stats in problem_stats.items():
            ac_rate = stats['ac'] / stats['total'] if stats['total'] > 0 else 0.5
            avg_attempts = stats['attempts_sum'] / stats['total'] if stats['total'] > 0 else 1.0
            difficulty[pid] = np.array([ac_rate, np.log1p(avg_attempts)], dtype=np.float32)

        return difficulty
