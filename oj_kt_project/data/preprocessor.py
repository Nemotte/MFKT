"""
数据预处理模块
"""
import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .knowledge_structure import (
    build_knowledge_structure,
    build_learning_relevance_matrix,
)

logger = logging.getLogger(__name__)


class OJDataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.problem_to_knowledge = {}
        self.knowledge_to_idx = {}
        self.problem_to_idx = {}

    # ========== 数据加载 ==========

    def load_data(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """加载所有数据"""
        submissions = pd.read_csv(self.config.SUBMISSION_DATA_PATH, encoding='utf-8-sig')
        with open(self.config.KNOWLEDGE_DATA_PATH, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
        with open(self.config.PROBLEM_DATA_PATH, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)

        # 列名兼容映射
        col_rename = {
            '时间': '提交时间',
            '题目': '题目名',
            '评测结果': '提交状态',
            '评测分数': '得分',
        }
        for old_col, new_col in col_rename.items():
            if old_col in submissions.columns and new_col not in submissions.columns:
                submissions.rename(columns={old_col: new_col}, inplace=True)

        # 过滤掉时间格式异常的行
        time_valid = pd.to_datetime(submissions['提交时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        n_bad = time_valid.isna().sum()
        if n_bad > 0:
            logger.warning(f"过滤掉 {n_bad} 条时间格式异常的提交记录")
            submissions = submissions[time_valid.notna()].reset_index(drop=True)

        return submissions, knowledge_data, problem_data

    # ========== 词汇表与 Q-Matrix ==========

    def build_vocabularies(self, knowledge_data, problem_data: Dict,
                           submissions: Optional[pd.DataFrame] = None):
        """构建知识点和题目词汇表"""
        all_knowledge = set()
        for problem in problem_data:
            if '相关知识点' in problem:
                for kp in problem['相关知识点']:
                    all_knowledge.add(kp['知识点'])

        self.knowledge_to_idx = {kp: idx for idx, kp in enumerate(sorted(all_knowledge))}

        all_problems = set()
        for problem in problem_data:
            all_problems.add(problem['题目'])
        if submissions is not None and '题目名' in submissions.columns:
            all_problems.update(submissions['题目名'].dropna().unique())
        all_problems.discard(np.nan)

        self.problem_to_idx = {prob: idx for idx, prob in enumerate(sorted(all_problems))}

        for problem in problem_data:
            problem_name = problem['题目']
            knowledge_points = []
            if '相关知识点' in problem:
                for kp in problem['相关知识点']:
                    knowledge_points.append({
                        'name': kp['知识点'],
                        'relevance': kp.get('关联度', kp.get('相关度', 0.5)),
                    })
            self.problem_to_knowledge[problem_name] = knowledge_points

        logger.info(f"知识点数: {len(self.knowledge_to_idx)}, 题目数: {len(self.problem_to_idx)}")

        # 构建 CLRS 知识结构
        ks = build_knowledge_structure(
            knowledge_data, self.knowledge_to_idx,
            self.problem_to_knowledge, self.problem_to_idx,
        )
        self.prerequisite_adj = ks['prerequisite_adj']
        self.problem_category = ks['problem_category']
        self._kp_to_chapter = ks['kp_to_chapter']
        self._kp_idx_to_chapter = ks['kp_idx_to_chapter']

    def build_q_matrix_init(self) -> np.ndarray:
        """构建初始 Q-matrix [num_problems, K]"""
        num_problems = len(self.problem_to_idx)
        num_kp = len(self.knowledge_to_idx)
        q_matrix = np.zeros((num_problems, num_kp), dtype=np.float32)

        for problem_name, kp_list in self.problem_to_knowledge.items():
            if problem_name not in self.problem_to_idx:
                continue
            prob_idx = self.problem_to_idx[problem_name]
            for kp in kp_list:
                kp_name = kp['name']
                if kp_name in self.knowledge_to_idx:
                    q_matrix[prob_idx, self.knowledge_to_idx[kp_name]] = kp['relevance']

        return q_matrix

    def build_learning_relevance_matrix(self) -> np.ndarray:
        """构建学习关联矩阵 F [num_problems, num_problems]"""
        F = build_learning_relevance_matrix(
            self.problem_to_knowledge, self.problem_to_idx,
            self.knowledge_to_idx, self._kp_idx_to_chapter,
        )
        self.learning_relevance_matrix = F
        return F

    # ========== 错误类型 ==========

    @staticmethod
    def get_verdict_type(status) -> int:
        """将提交状态映射为 verdict 类型 ID（0=AC,1=WA,2=TLE,3=MLE,4=RE,5=CE/Other）"""
        if pd.isna(status):
            return 5
        s = str(status).lower()
        if 'accepted' in s:
            return 0
        elif 'wrong' in s:
            return 1
        elif 'time' in s:
            return 2
        elif 'memory' in s:
            return 3
        elif 'runtime' in s or 'error' in s:
            return 4
        elif 'compil' in s:
            return 5
        return 5

    # ========== Session 切割 ==========

    @staticmethod
    def _split_sessions(prob_group: pd.DataFrame, gap_hours: float,
                        max_hours: float) -> List[pd.DataFrame]:
        """将同一题的提交按时间间隔切割为 session"""
        times = pd.to_datetime(prob_group['提交时间'], format='%Y-%m-%d %H:%M:%S')
        gaps = times.diff()
        gap_threshold = pd.Timedelta(hours=gap_hours)
        max_duration = pd.Timedelta(hours=max_hours)

        sessions = []
        session_start = 0
        session_start_time = times.iloc[0]

        for i in range(1, len(prob_group)):
            if gaps.iloc[i] > gap_threshold or (times.iloc[i] - session_start_time) > max_duration:
                sessions.append(prob_group.iloc[session_start:i])
                session_start = i
                session_start_time = times.iloc[i]

        sessions.append(prob_group.iloc[session_start:])
        return sessions

    @staticmethod
    def _compute_time_features(session_rows: pd.DataFrame) -> np.ndarray:
        """从 session 提取时间特征 [4维]"""
        times = pd.to_datetime(session_rows['提交时间'], format='%Y-%m-%d %H:%M:%S')
        if len(times) < 2:
            return np.array([0.0, np.log1p(1), 0.0, 0.0], dtype=np.float32)

        duration = (times.iloc[-1] - times.iloc[0]).total_seconds() / 60.0
        intervals = times.diff().dropna().dt.total_seconds().values

        return np.array([
            np.log1p(duration),
            np.log1p(len(times)),
            np.log1p(intervals.mean()) if len(intervals) > 0 else 0.0,
            np.log1p(intervals.std()) if len(intervals) > 1 else 0.0,
        ], dtype=np.float32)

    # ========== 学生时间线 ==========

    def create_student_timelines(self, submissions: pd.DataFrame) -> Dict[str, List]:
        """
        为每个学生创建做题时间线

        每个题目记录：verdict类型、得分、耗时、内存、尝试次数
        防数据泄露：模型输入仅包含 session 内 AC 之前的提交信息
        """
        submissions = submissions.sort_values('提交时间')
        num_kp = len(self.knowledge_to_idx)
        gap_hours = self.config.SESSION_GAP_HOURS
        max_hours = self.config.SESSION_MAX_HOURS
        student_timelines = {}

        for student_id, student_group in submissions.groupby('送交者'):
            timeline = []

            for problem_name, prob_group in student_group.groupby('题目名'):
                prob_group = prob_group.sort_values('提交时间')
                sessions = self._split_sessions(prob_group, gap_hours, max_hours)
                first_session = sessions[0]

                # 收集首个 session 的提交信息
                verdict_list = []
                score_list = []
                first_ac_idx = None

                for sub_idx, (_, row) in enumerate(first_session.iterrows()):
                    verdict = self.get_verdict_type(row['提交状态'])
                    verdict_list.append(verdict)

                    score = float(row.get('得分', 0)) / 100.0 if pd.notna(row.get('得分')) else 0.0
                    time_ms = float(row.get('耗时', 0)) if pd.notna(row.get('耗时')) else 0.0
                    memory_kb = float(row.get('内存', 0)) if pd.notna(row.get('内存')) else 0.0
                    score_list.append(np.array([
                        score, np.log1p(time_ms), np.log1p(memory_kb)
                    ], dtype=np.float32))

                    if first_ac_idx is None and verdict == 0:
                        first_ac_idx = sub_idx

                session_ac = first_ac_idx is not None
                num_before_ac = first_ac_idx if session_ac else len(verdict_list)

                # 题目级聚合：AC前的最终verdict、最高得分、尝试次数
                verdicts_before_ac = verdict_list[:num_before_ac]
                scores_before_ac = score_list[:num_before_ac]

                # 最终verdict：如果AC了就是AC(0)，否则取最后一次提交的verdict
                if session_ac:
                    final_verdict = 0
                elif verdicts_before_ac:
                    final_verdict = verdicts_before_ac[-1]
                else:
                    final_verdict = 5

                # 最终得分/耗时/内存：取最后一次提交（或AC提交）
                if session_ac and first_ac_idx < len(score_list):
                    final_score_features = score_list[first_ac_idx]
                elif scores_before_ac:
                    final_score_features = scores_before_ac[-1]
                else:
                    final_score_features = np.zeros(3, dtype=np.float32)

                attempt_count = min(len(verdict_list), self.config.MAX_ATTEMPTS) if session_ac \
                    else min(num_before_ac, self.config.MAX_ATTEMPTS)

                is_ac_final = any(
                    row['提交状态'] == 'Accepted'
                    for _, row in prob_group.iterrows()
                )

                time_features = self._compute_time_features(first_session)

                knowledge_vec = np.zeros(num_kp, dtype=np.float32)
                for kp in self.problem_to_knowledge.get(problem_name, []):
                    kp_name = kp['name']
                    if kp_name in self.knowledge_to_idx:
                        knowledge_vec[self.knowledge_to_idx[kp_name]] = kp['relevance']

                # 保留完整提交序列供 AttemptEncoder 使用
                cutoff = (first_ac_idx + 1) if session_ac else len(verdict_list)
                cutoff = min(cutoff, self.config.MAX_ATTEMPTS)

                # Verdict 分布特征: session 内各 verdict 类型的频率 [6维]
                verdict_dist = np.zeros(6, dtype=np.float32)
                for v in verdict_list[:cutoff]:
                    if 0 <= v < 6:
                        verdict_dist[v] += 1
                if verdict_dist.sum() > 0:
                    verdict_dist = verdict_dist / verdict_dist.sum()

                timeline.append({
                    'problem_name': problem_name,
                    'problem_idx': self.problem_to_idx.get(problem_name, 0),
                    'problem_category': getattr(self, 'problem_category', {}).get(
                        self.problem_to_idx.get(problem_name, 0), 7),
                    'first_submit_time': str(first_session.iloc[0]['提交时间']),
                    'is_ac': is_ac_final,
                    'session_ac': session_ac,
                    'first_ac': (first_ac_idx == 0) if session_ac else False,
                    'verdict_type': final_verdict,
                    'verdict_dist': verdict_dist,
                    'score_features': final_score_features,
                    'attempt_count': attempt_count,
                    'total_submissions': len(verdict_list),
                    'time_features': time_features,
                    'knowledge_vec': knowledge_vec,
                    'attempt_verdicts': verdict_list[:cutoff],
                    'attempt_scores': score_list[:cutoff],
                })

            timeline.sort(key=lambda x: x['first_submit_time'])
            student_timelines[str(student_id)] = timeline

        logger.info(f"创建了 {len(student_timelines)} 个学生时间线")
        return student_timelines

    # ========== 统计量 ==========

    def compute_problem_difficulty(self, train_timelines):
        """从训练集统计题目难度特征"""
        problem_stats = {}

        for student_id, timeline in train_timelines.items():
            for attempt in timeline:
                pid = attempt['problem_idx']
                if pid not in problem_stats:
                    problem_stats[pid] = {'ac': 0, 'total': 0, 'subs': 0}
                problem_stats[pid]['total'] += 1
                problem_stats[pid]['subs'] += attempt['total_submissions']
                if attempt['session_ac']:
                    problem_stats[pid]['ac'] += 1

        difficulty = {}
        for pid, stats in problem_stats.items():
            ac_rate = stats['ac'] / max(stats['total'], 1)
            avg_attempts = stats['subs'] / max(stats['total'], 1)
            difficulty[pid] = np.array([ac_rate, np.log1p(avg_attempts)], dtype=np.float32)

        logger.info(f"计算了 {len(difficulty)} 道题目的难度特征")
        return difficulty
