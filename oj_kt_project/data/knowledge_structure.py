"""
CLRS 知识结构 — 章节先修关系、知识概念路径、学习关联矩阵
"""
import re
import numpy as np
import logging
from typing import Dict, List
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# CLRS 章节先修关系（基于《算法导论》教材结构）
CHAPTER_PREREQUISITES = {
    3: [2], 4: [2, 3], 7: [2], 8: [7], 12: [6],
    15: [2, 4], 16: [15], 19: [6], 22: [11],
    23: [6, 22], 24: [22], 25: [15, 24],
    26: [22, 24], 29: [26], 34: [15],
}

# CLRS 章节 → 算法范式类别 (8 类)
# 0=Sorting, 1=D&C, 2=DataStructure, 3=DP, 4=Greedy,
# 5=GraphBasics, 6=GraphAlgo, 7=Advanced
CHAPTER_TO_CATEGORY = {
    2: 0, 3: 0, 4: 1, 6: 2, 7: 0, 8: 0, 11: 2, 12: 2,
    15: 3, 16: 4, 19: 2, 22: 5, 23: 6, 24: 6, 25: 6,
    26: 7, 29: 7, 34: 7,
}


def _compute_chapter_ancestors() -> Dict[int, set]:
    """
    计算每个 CLRS 章节的传递先修闭包（含自身）

    例：Ch25 的先修链 = {25} ∪ {15,24} ∪ {2,4,22} ∪ {3,11} = {25,15,24,2,4,3,22,11}
    """
    cache: Dict[int, set] = {}

    def _get(ch: int) -> set:
        if ch in cache:
            return cache[ch]
        result = {ch}
        for prereq in CHAPTER_PREREQUISITES.get(ch, []):
            result |= _get(prereq)
        cache[ch] = result
        return result

    all_chapters = set()
    for ch in CHAPTER_PREREQUISITES:
        all_chapters.add(ch)
        all_chapters.update(CHAPTER_PREREQUISITES[ch])

    for ch in all_chapters:
        _get(ch)

    return cache


def build_knowledge_structure(
    knowledge_data: List[Dict],
    knowledge_to_idx: Dict[str, int],
    problem_to_knowledge: Dict[str, List],
    problem_to_idx: Dict[str, int],
) -> dict:
    """
    从 knowledge_points.json 构建 CLRS 知识结构

    Returns:
        dict with keys:
            prerequisite_adj: np.ndarray [K, K]
            problem_category: Dict[int, int]
            kp_to_chapter: Dict[str, int]
            kp_idx_to_chapter: Dict[int, int]
    """
    # 1. 解析知识点所属章节
    kp_to_chapter = {}
    chapter_pattern = re.compile(r'第(\d+)章')
    for item in knowledge_data:
        chapter_str = item.get('章节', '')
        m = chapter_pattern.search(chapter_str)
        if not m:
            continue
        chapter_num = int(m.group(1))
        kp_full_name = item['知识点']
        kp_short = re.sub(r'^\d+\.\d+\s*', '', kp_full_name).strip()
        kp_to_chapter[kp_short] = chapter_num

    # 2. 将 knowledge_to_idx 中的 KP 名映射到章节
    kp_idx_to_chapter = {}
    for kp_name, kp_idx in knowledge_to_idx.items():
        matched_chapter = None
        for full_name, chapter in kp_to_chapter.items():
            if kp_name in full_name or full_name in kp_name:
                matched_chapter = chapter
                break
        if matched_chapter is None:
            best_ratio, best_ch = 0.0, None
            for full_name, chapter in kp_to_chapter.items():
                ratio = SequenceMatcher(None, kp_name, full_name).ratio()
                if ratio > best_ratio:
                    best_ratio, best_ch = ratio, chapter
            if best_ratio > 0.5:
                matched_chapter = best_ch

        if matched_chapter is not None:
            kp_idx_to_chapter[kp_idx] = matched_chapter

    logger.info(f"KP→章节映射: {len(kp_idx_to_chapter)}/{len(knowledge_to_idx)} 成功")

    # 3. 构建先修关系邻接矩阵 [K, K]
    num_kp = len(knowledge_to_idx)
    prerequisite_adj = np.zeros((num_kp, num_kp), dtype=np.float32)

    for kp_idx, chapter in kp_idx_to_chapter.items():
        prereq_chapters = CHAPTER_PREREQUISITES.get(chapter, [])
        for other_idx, other_chapter in kp_idx_to_chapter.items():
            if other_chapter in prereq_chapters:
                prerequisite_adj[kp_idx, other_idx] = 1.0

    n_edges = int(prerequisite_adj.sum())
    logger.info(f"先修关系邻接矩阵: {num_kp}x{num_kp}, {n_edges} 条边")

    # 4. 构建题目 → 算法范式类别
    problem_category = {}
    for problem_name, kp_list in problem_to_knowledge.items():
        if problem_name not in problem_to_idx:
            continue
        prob_idx = problem_to_idx[problem_name]
        best_relevance, best_chapter = 0.0, None
        for kp in kp_list:
            kp_name = kp['name']
            for full_name, chapter in kp_to_chapter.items():
                if kp_name in full_name or full_name in kp_name:
                    if kp['relevance'] > best_relevance:
                        best_relevance = kp['relevance']
                        best_chapter = chapter
                    break
        if best_chapter is not None:
            problem_category[prob_idx] = CHAPTER_TO_CATEGORY.get(best_chapter, 7)
        else:
            problem_category[prob_idx] = 7

    logger.info(f"题目→算法类别: {len(problem_category)} 道题已分类")

    return {
        'prerequisite_adj': prerequisite_adj,
        'problem_category': problem_category,
        'kp_to_chapter': kp_to_chapter,
        'kp_idx_to_chapter': kp_idx_to_chapter,
    }


def build_learning_relevance_matrix(
    problem_to_knowledge: Dict[str, List],
    problem_to_idx: Dict[str, int],
    knowledge_to_idx: Dict[str, int],
    kp_idx_to_chapter: Dict[int, int],
) -> np.ndarray:
    """
    构建学习关联矩阵 F [num_problems, num_problems]

    基于论文"领域知识引导注意力知识追踪"的方法论：
    1. 知识概念路径 = 自身所属章节 + 沿先修关系回溯到根的所有章节
    2. 题目路径 = 其所有关联 KP 的路径并集
    3. F[i,j] = 1 当且仅当题目 i 和题目 j 的知识概念路径有交集
    """
    num_problems = len(problem_to_idx)
    chapter_ancestors = _compute_chapter_ancestors()

    # 为每道题计算知识概念路径（章节集合）
    problem_routes: Dict[int, set] = {}
    for problem_name, kp_list in problem_to_knowledge.items():
        if problem_name not in problem_to_idx:
            continue
        prob_idx = problem_to_idx[problem_name]
        route = set()
        for kp in kp_list:
            kp_name = kp['name']
            if kp_name in knowledge_to_idx:
                kp_idx = knowledge_to_idx[kp_name]
                if kp_idx in kp_idx_to_chapter:
                    ch = kp_idx_to_chapter[kp_idx]
                    route |= chapter_ancestors.get(ch, {ch})
        problem_routes[prob_idx] = route

    # 构建 F 矩阵
    F = np.zeros((num_problems, num_problems), dtype=np.float32)
    np.fill_diagonal(F, 1.0)

    for i in range(num_problems):
        route_i = problem_routes.get(i, set())
        if not route_i:
            continue
        for j in range(i + 1, num_problems):
            route_j = problem_routes.get(j, set())
            if route_i & route_j:
                F[i, j] = 1.0
                F[j, i] = 1.0

    n_ones = int(F.sum())
    n_with_route = sum(1 for r in problem_routes.values() if r)
    density = n_ones / (num_problems * num_problems) if num_problems > 0 else 0
    logger.info(
        f"学习关联矩阵 F: {num_problems}×{num_problems}, "
        f"{n_with_route}/{num_problems} 道题有知识路径, "
        f"{n_ones} 个关联对 (密度 {density:.3f})"
    )

    return F
