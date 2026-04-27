"""
PyTorch Dataset 和 Collate 函数 — 题目级序列（无内层提交聚合）
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def compute_mastery_labels(timeline: List[Dict], num_kp: int,
                           window_size: int = 5) -> List[np.ndarray]:
    """
    滑动窗口策略生成掌握度伪标签

    返回 List[ndarray [K]]（1D 标量 AC 掌握度）
    """
    kp_scalar_windows: Dict[int, List[float]] = {
        k: [] for k in range(num_kp)
    }

    mastery_labels = []

    for attempt in timeline:
        kv = attempt['knowledge_vec']

        # 标量 AC 掌握度
        if attempt.get('first_ac', False):
            ac_signal = 1.0
        elif attempt.get('is_ac', False):
            total = attempt.get('total_submissions', 1)
            ac_signal = max(0.5, 1.0 - 0.1 * (total - 1))
        else:
            ac_signal = 0.0

        for kp_idx in range(num_kp):
            if kv[kp_idx] > 0:
                w = kp_scalar_windows[kp_idx]
                w.append(ac_signal)
                if len(w) > window_size:
                    w.pop(0)

        mastery = np.zeros(num_kp, dtype=np.float32)
        for kp_idx in range(num_kp):
            w = kp_scalar_windows[kp_idx]
            if len(w) > 0:
                mastery[kp_idx] = sum(w) / len(w)

        mastery_labels.append(mastery)

    return mastery_labels


class StudentTimelineDataset(Dataset):
    """
    学生做题时间线数据集 — 题目级序列

    每个样本 = 一个学生的前 i 道题历史 → 预测第 i+1 道题是否 AC
    每个时间步 = (problem_idx, verdict_type, attempt_count, score_features,
                  time_features, student_features, problem_difficulty, problem_category)
    """
    def __init__(self, student_timelines, preprocessor, config, problem_difficulty=None):
        self.config = config
        self.preprocessor = preprocessor
        self.samples = []
        self.num_pos = 0
        self.num_neg = 0

        self.problem_difficulty = problem_difficulty or {}
        if self.problem_difficulty:
            all_diffs = np.stack(list(self.problem_difficulty.values()))
            self.default_difficulty = np.mean(all_diffs, axis=0)
        else:
            self.default_difficulty = np.array([0.5, 1.0], dtype=np.float32)

        num_kp = len(preprocessor.knowledge_to_idx)
        max_T = config.MAX_PROBLEM_SEQ_LEN
        window_size = config.MASTERY_WINDOW_SIZE

        for student_id, timeline in student_timelines.items():
            if len(timeline) < 2:
                continue

            mastery_labels = compute_mastery_labels(
                timeline, num_kp, window_size,
            )

            for i in range(1, len(timeline)):
                history = timeline[:i]
                if len(history) > max_T:
                    history = history[-max_T:]

                problem_ids = []
                verdict_types = []
                attempt_counts = []
                score_features_list = []
                time_features_list = []
                student_features_list = []
                problem_diff_list = []
                problem_categories = []
                attempt_verdicts_list = []
                attempt_scores_list = []
                attempt_lens_list = []
                verdict_dist_list = []

                ac_count_so_far = 0
                # V2 增量计算：预分配 kp 级别的 AC 计数
                if config.STUDENT_FEATURE_INPUT_DIM != 2:
                    kp_ac_count = np.zeros(len(preprocessor.knowledge_to_idx), dtype=np.int32)
                    kp_total_count = np.zeros(len(preprocessor.knowledge_to_idx), dtype=np.int32)

                for t_idx, attempt in enumerate(history):
                    problem_ids.append(attempt['problem_idx'])
                    verdict_types.append(attempt['verdict_type'])
                    attempt_counts.append(min(attempt['attempt_count'], config.MAX_ATTEMPTS - 1))
                    score_features_list.append(attempt['score_features'])
                    time_features_list.append(attempt['time_features'])
                    problem_categories.append(attempt.get('problem_category', 7))

                    # 学生能力特征
                    if attempt['session_ac']:
                        ac_count_so_far += 1
                    running_ac_rate = ac_count_so_far / (t_idx + 1)
                    recent_window = min(5, t_idx + 1)
                    recent_ac = sum(1 for j in range(t_idx + 1 - recent_window, t_idx + 1)
                                    if history[j]['session_ac'])
                    recent_ac_rate = recent_ac / recent_window

                    if config.STUDENT_FEATURE_INPUT_DIM == 2:
                        # V1: 原始 2 维特征
                        student_features_list.append(
                            np.array([running_ac_rate, recent_ac_rate], dtype=np.float32)
                        )
                    else:
                        # V2: 4 维特征（+知识点AC率差值 +题目间时间间隔）
                        # O(T) 增量计算：用 kp 向量做点积判断重叠
                        curr_kv = attempt['knowledge_vec']
                        curr_active = curr_kv > 0
                        kp_overlap_total = kp_total_count[curr_active].sum()
                        kp_overlap_ac = kp_ac_count[curr_active].sum()
                        kp_hist_ac_delta = (kp_overlap_ac / kp_overlap_total - running_ac_rate) if kp_overlap_total > 0 else 0.0

                        if t_idx > 0:
                            try:
                                t_prev = datetime.strptime(history[t_idx-1]['first_submit_time'], '%Y-%m-%d %H:%M:%S')
                                t_curr = datetime.strptime(attempt['first_submit_time'], '%Y-%m-%d %H:%M:%S')
                                gap_hours = max((t_curr - t_prev).total_seconds() / 3600, 0)
                            except (ValueError, KeyError):
                                gap_hours = 0.0
                        else:
                            gap_hours = 0.0

                        student_features_list.append(np.array([
                            running_ac_rate,
                            recent_ac_rate,
                            kp_hist_ac_delta,
                            np.log1p(gap_hours),
                        ], dtype=np.float32))

                        # 增量更新：当前 attempt 的 kp 贡献加入历史
                        kp_total_count[curr_active] += 1
                        if attempt['session_ac']:
                            kp_ac_count[curr_active] += 1

                    # 题目难度特征
                    pid = attempt['problem_idx']
                    diff = self.problem_difficulty.get(pid, self.default_difficulty)
                    problem_diff_list.append(diff)

                    # Attempt 序列（供 AttemptEncoder 使用）
                    av = attempt.get('attempt_verdicts', [attempt['verdict_type']])
                    asc = attempt.get('attempt_scores', [attempt['score_features']])
                    attempt_verdicts_list.append(av)
                    attempt_scores_list.append(asc)
                    attempt_lens_list.append(len(av))

                    # Verdict 分布特征
                    vd = attempt.get('verdict_dist', None)
                    if vd is None:
                        vd = np.zeros(6, dtype=np.float32)
                        vt_val = attempt.get('verdict_type', 0)
                        if 0 <= vt_val < 6:
                            vd[vt_val] = 1.0
                    verdict_dist_list.append(vd)

                next_attempt = timeline[i]
                target = 1.0 if next_attempt['session_ac'] else 0.0
                next_problem_id = next_attempt['problem_idx']
                next_problem_category = next_attempt.get('problem_category', 7)

                if target > 0.5:
                    self.num_pos += 1
                else:
                    self.num_neg += 1

                mastery_target = mastery_labels[i - 1]

                next_problem_diff = self.problem_difficulty.get(
                    next_problem_id, self.default_difficulty
                )

                self.samples.append({
                    'problem_ids': problem_ids,
                    'verdict_types': verdict_types,
                    'attempt_counts': attempt_counts,
                    'score_features': score_features_list,
                    'time_features': time_features_list,
                    'student_features': student_features_list,
                    'problem_difficulty': problem_diff_list,
                    'problem_categories': problem_categories,
                    'verdict_dist': verdict_dist_list,
                    'seq_len': len(history),
                    'next_problem_id': next_problem_id,
                    'next_problem_category': next_problem_category,
                    'target': target,
                    'mastery_target': mastery_target,
                    'next_problem_difficulty': next_problem_diff,
                    'attempt_verdicts': attempt_verdicts_list,
                    'attempt_scores': attempt_scores_list,
                    'attempt_lens': attempt_lens_list,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_pos_weight(self):
        if self.num_pos == 0:
            return 1.0
        return self.num_neg / self.num_pos

    def get_class_distribution(self):
        return self.num_pos, self.num_neg


def collate_fn(batch):
    """
    题目级 padding collate 函数（向量化版本）

    用 numpy 批量填充后一次性转 tensor，避免 Python for 循环逐元素赋值。
    """
    batch_size = len(batch)
    T_max = max(item['seq_len'] for item in batch)
    K = batch[0]['mastery_target'].shape[0]

    time_dim = len(batch[0]['time_features'][0])
    score_dim = len(batch[0]['score_features'][0])
    student_feat_dim = len(batch[0]['student_features'][0])
    diff_dim = len(batch[0]['problem_difficulty'][0])

    has_verdict_dist = 'verdict_dist' in batch[0]
    has_attempts = 'attempt_verdicts' in batch[0]

    if has_verdict_dist:
        verdict_dist_dim = len(batch[0]['verdict_dist'][0])

    if has_attempts:
        A_max = max(
            a_len
            for item in batch
            for a_len in item['attempt_lens']
        )
        A_max = max(A_max, 1)

    # 用 numpy 预分配
    np_problem_ids = np.zeros((batch_size, T_max), dtype=np.int64)
    np_verdict_types = np.zeros((batch_size, T_max), dtype=np.int64)
    np_attempt_counts = np.zeros((batch_size, T_max), dtype=np.int64)
    np_problem_categories = np.zeros((batch_size, T_max), dtype=np.int64)
    np_score_features = np.zeros((batch_size, T_max, score_dim), dtype=np.float32)
    np_time_features = np.zeros((batch_size, T_max, time_dim), dtype=np.float32)
    np_student_features = np.zeros((batch_size, T_max, student_feat_dim), dtype=np.float32)
    np_problem_difficulty = np.zeros((batch_size, T_max, diff_dim), dtype=np.float32)
    np_seq_lens = np.zeros(batch_size, dtype=np.int64)
    np_next_problem_ids = np.zeros(batch_size, dtype=np.int64)
    np_next_problem_categories = np.zeros(batch_size, dtype=np.int64)
    np_next_problem_difficulty = np.zeros((batch_size, diff_dim), dtype=np.float32)
    np_targets = np.zeros(batch_size, dtype=np.float32)
    mastery_shape = batch[0]['mastery_target'].shape
    np_mastery_targets = np.zeros((batch_size, *mastery_shape), dtype=np.float32)
    np_problem_mask = np.zeros((batch_size, T_max), dtype=np.bool_)

    if has_verdict_dist:
        np_verdict_dist = np.zeros((batch_size, T_max, verdict_dist_dim), dtype=np.float32)
    if has_attempts:
        np_attempt_verdicts = np.zeros((batch_size, T_max, A_max), dtype=np.int64)
        np_attempt_scores = np.zeros((batch_size, T_max, A_max, 3), dtype=np.float32)
        np_attempt_lens = np.zeros((batch_size, T_max), dtype=np.int64)

    for i, item in enumerate(batch):
        T = item['seq_len']
        np_seq_lens[i] = T
        np_next_problem_ids[i] = item['next_problem_id']
        np_next_problem_categories[i] = item['next_problem_category']
        np_targets[i] = item['target']
        np_mastery_targets[i] = item['mastery_target']
        np_next_problem_difficulty[i] = item['next_problem_difficulty']
        np_problem_mask[i, :T] = True

        # 向量化填充：一次性赋值整个序列
        np_problem_ids[i, :T] = item['problem_ids']
        np_verdict_types[i, :T] = item['verdict_types']
        np_attempt_counts[i, :T] = item['attempt_counts']
        np_problem_categories[i, :T] = item['problem_categories']
        np_score_features[i, :T] = np.stack(item['score_features'][:T])
        np_time_features[i, :T] = np.stack(item['time_features'][:T])
        np_student_features[i, :T] = np.stack(item['student_features'][:T])
        np_problem_difficulty[i, :T] = np.stack(item['problem_difficulty'][:T])

        if has_verdict_dist:
            np_verdict_dist[i, :T] = np.stack(item['verdict_dist'][:T])

        if has_attempts:
            np_attempt_lens[i, :T] = item['attempt_lens']
            for t in range(T):
                a_len = item['attempt_lens'][t]
                if a_len > 0:
                    np_attempt_verdicts[i, t, :a_len] = item['attempt_verdicts'][t][:a_len]
                    np_attempt_scores[i, t, :a_len] = np.stack(item['attempt_scores'][t][:a_len])

    # 一次性转 tensor
    result = {
        'problem_ids': torch.from_numpy(np_problem_ids),
        'verdict_types': torch.from_numpy(np_verdict_types),
        'attempt_counts': torch.from_numpy(np_attempt_counts),
        'score_features': torch.from_numpy(np_score_features),
        'time_features': torch.from_numpy(np_time_features),
        'student_features': torch.from_numpy(np_student_features),
        'problem_difficulty': torch.from_numpy(np_problem_difficulty),
        'problem_categories': torch.from_numpy(np_problem_categories),
        'seq_lens': torch.from_numpy(np_seq_lens),
        'next_problem_ids': torch.from_numpy(np_next_problem_ids),
        'next_problem_categories': torch.from_numpy(np_next_problem_categories),
        'next_problem_difficulty': torch.from_numpy(np_next_problem_difficulty),
        'targets': torch.from_numpy(np_targets),
        'mastery_targets': torch.from_numpy(np_mastery_targets),
        'problem_mask': torch.from_numpy(np_problem_mask),
    }

    if has_verdict_dist:
        result['verdict_dist'] = torch.from_numpy(np_verdict_dist)

    if has_attempts:
        result['attempt_verdicts'] = torch.from_numpy(np_attempt_verdicts)
        result['attempt_scores'] = torch.from_numpy(np_attempt_scores)
        result['attempt_lens'] = torch.from_numpy(np_attempt_lens)

    return result
