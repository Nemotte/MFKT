"""
Code-DKT 专用预处理器

从 MainTable + CodeStates 构建带代码 token 特征的学生时间线。
兼容 CodeWorkoutPreprocessor 的接口。
"""
import pandas as pd
import numpy as np
import os
import logging
import javalang
from collections import defaultdict

logger = logging.getLogger(__name__)

# javalang tokenizer 产生的 token 类型 → 整数索引 (0 保留给 padding)
TOKEN_TYPE_MAP = {
    'Keyword': 1,
    'Identifier': 2,
    'Separator': 3,
    'Operator': 4,
    'BasicType': 5,
    'Modifier': 6,
    'DecimalInteger': 7,
    'String': 8,
    'Boolean': 9,
    'Null': 10,
    'OctalInteger': 11,
    'DecimalFloatingPoint': 12,
    'HexInteger': 13,
    'BinaryInteger': 14,
    'Annotation': 15,
    'Character': 16,
    'HexFloatingPoint': 17,
}
NUM_TOKEN_TYPES = len(TOKEN_TYPE_MAP)
MAX_CODE_TOKENS = 50  # 截断/padding 到固定长度


def tokenize_java_code(code_str, max_len=MAX_CODE_TOKENS):
    """将 Java 代码转为 token 类型 ID 序列"""
    try:
        tokens = list(javalang.tokenizer.tokenize(str(code_str)))
        ids = []
        for t in tokens[:max_len]:
            tname = type(t).__name__
            ids.append(TOKEN_TYPE_MAP.get(tname, 0))
        return ids
    except Exception:
        return []


class CodeDKTPreprocessor:
    """
    从 MainTable + CodeStates 构建带代码特征的学生时间线

    对每个 (student, problem) 对，取最后一次 Run.Program 提交的代码，
    用 javalang tokenizer 提取 token 类型序列。
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.problem_to_idx = {}
        self.knowledge_to_idx = {}

    def load_and_build(self):
        """加载数据并构建 student_timelines + 元数据"""
        # 1. 加载 early + late (聚合数据)
        dfs = []
        for fname in ['early.csv', 'late.csv']:
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info(f"  加载 {fname}: {len(df)} 条记录")
                dfs.append(df)
        if not dfs:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到 early.csv 或 late.csv")
        combined = pd.concat(dfs, ignore_index=True)

        # 2. 加载 MainTable (Run.Program 事件)
        mt_path = os.path.join(self.data_dir, 'Data', 'MainTable.csv')
        mt = pd.read_csv(mt_path)
        runs = mt[mt['EventType'] == 'Run.Program'].copy()
        logger.info(f"  MainTable Run.Program: {len(runs)} 条")

        # 3. 加载 CodeStates (只加载需要的)
        needed_ids = set(runs['CodeStateID'].unique())
        cs_path = os.path.join(self.data_dir, 'Data', 'CodeStates', 'CodeStates.csv')
        logger.info(f"  加载 CodeStates (需要 {len(needed_ids)} 个)...")
        cs_dict = {}
        for chunk in pd.read_csv(cs_path, chunksize=50000):
            mask = chunk['CodeStateID'].isin(needed_ids)
            for _, row in chunk[mask].iterrows():
                cs_dict[row['CodeStateID']] = row['Code']
            if len(cs_dict) >= len(needed_ids):
                break
        logger.info(f"  匹配到 {len(cs_dict)} 个代码")

        # 4. 对每个 (student, problem) 取最后一次提交的代码并 tokenize
        logger.info("  提取代码 token 特征...")
        runs = runs.sort_values('Order')
        last_code = runs.groupby(['SubjectID', 'ProblemID']).last().reset_index()

        code_tokens_map = {}  # (SubjectID, ProblemID) → token_ids list
        tokenize_success = 0
        tokenize_fail = 0
        for _, row in last_code.iterrows():
            key = (row['SubjectID'], row['ProblemID'])
            csid = row['CodeStateID']
            code = cs_dict.get(csid, '')
            token_ids = tokenize_java_code(code)
            if token_ids:
                tokenize_success += 1
            else:
                tokenize_fail += 1
            code_tokens_map[key] = token_ids

        logger.info(f"  Tokenize 成功: {tokenize_success}, 失败: {tokenize_fail}")

        # 5. 构建题目索引
        unique_problems = sorted(combined['ProblemID'].unique())
        self.problem_to_idx = {pid: idx for idx, pid in enumerate(unique_problems)}
        num_problems = len(unique_problems)
        self.knowledge_to_idx = {i: i for i in range(num_problems)}
        num_kp = num_problems
        init_q_matrix = np.eye(num_kp, dtype=np.float32)

        # 6. 构建学生时间线
        student_timelines = self._build_timelines(combined, code_tokens_map, num_kp)
        logger.info(f"  学生数: {len(student_timelines)}, 题目数: {num_problems}")

        return student_timelines, num_problems, num_kp, init_q_matrix

    def _build_timelines(self, df, code_tokens_map, num_kp):
        """构建带代码特征的学生时间线"""
        student_timelines = {}

        for subject_id, group in df.groupby('SubjectID'):
            group = group.sort_values(['AssignmentID', 'ProblemID'])
            timeline = []

            for _, row in group.iterrows():
                pid_idx = self.problem_to_idx[row['ProblemID']]
                is_ac = bool(row['Label'])
                attempts = int(row['Attempts'])

                kv = np.zeros(num_kp, dtype=np.float32)
                kv[pid_idx] = 1.0

                # 获取代码 token 特征
                key = (subject_id, row['ProblemID'])
                token_ids = code_tokens_map.get(key, [])

                # padding/truncation 到 MAX_CODE_TOKENS
                if len(token_ids) >= MAX_CODE_TOKENS:
                    padded_tokens = token_ids[:MAX_CODE_TOKENS]
                    token_len = MAX_CODE_TOKENS
                else:
                    token_len = len(token_ids)
                    padded_tokens = token_ids + [0] * (MAX_CODE_TOKENS - len(token_ids))

                timeline.append({
                    'problem_idx': pid_idx,
                    'verdict_type': 0 if is_ac else 1,
                    'attempt_count': min(attempts, 10),
                    'score_features': np.array([
                        float(is_ac), np.log1p(attempts), 0.0,
                    ], dtype=np.float32),
                    'time_features': np.zeros(4, dtype=np.float32),
                    'knowledge_vec': kv,
                    'session_ac': is_ac,
                    'first_ac': is_ac and attempts == 1,
                    'total_submissions': attempts,
                    'problem_category': 0,
                    # Code-DKT 专用字段
                    'code_token_ids': np.array(padded_tokens, dtype=np.int64),
                    'code_token_len': token_len,
                })

            if len(timeline) >= 2:
                student_timelines[subject_id] = timeline

        return student_timelines

    def compute_problem_difficulty(self, student_timelines):
        """从训练集统计题目难度"""
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
