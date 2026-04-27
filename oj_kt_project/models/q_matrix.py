"""
可学习 Q-Matrix — 题目与知识点的关联矩阵
使用题目 embedding 和知识点 embedding 点积 + sigmoid 生成 Q 权重
支持 SVD 初始化 + BCE 正则化
"""
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LearnableQMatrix(nn.Module):
    """
    可学习的题目-知识点关联矩阵

    Q_learned[p, k] = sigmoid(prob_emb[p] @ kp_emb[k])
    使用人工标注 Q-matrix 的 SVD 分解来初始化两组 embedding
    """
    def __init__(self, num_problems, num_kp, config, init_q_matrix=None):
        super().__init__()
        self.num_problems = num_problems
        self.num_kp = num_kp
        self.embed_dim = config.QMATRIX_EMBED_DIM

        self.problem_embedding = nn.Embedding(num_problems, self.embed_dim)
        self.kp_embedding = nn.Embedding(num_kp, self.embed_dim)

        if init_q_matrix is not None:
            self.register_buffer(
                'annotated_q',
                torch.from_numpy(init_q_matrix).float()
            )
            self._svd_init(init_q_matrix)
        else:
            self.register_buffer(
                'annotated_q',
                torch.full((num_problems, num_kp), config.QMATRIX_INIT_WEIGHT)
            )
            nn.init.xavier_uniform_(self.problem_embedding.weight)
            nn.init.xavier_uniform_(self.kp_embedding.weight)

    def _svd_init(self, q_matrix):
        """用 SVD 分解初始化 embedding，数值安全处理"""
        eps = 1e-4
        q_clamped = np.clip(q_matrix, eps, 1.0 - eps)
        logit_q = np.log(q_clamped / (1.0 - q_clamped))

        # 将 NaN/Inf 替换为 0（防御性处理）
        logit_q = np.nan_to_num(logit_q, nan=0.0, posinf=5.0, neginf=-5.0)

        try:
            U, S, Vt = np.linalg.svd(logit_q, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.warning("SVD 分解失败，使用 xavier 初始化")
            nn.init.xavier_uniform_(self.problem_embedding.weight)
            nn.init.xavier_uniform_(self.kp_embedding.weight)
            return

        k = min(self.embed_dim, len(S))
        sqrt_S = np.sqrt(np.maximum(S[:k], 0.0))  # 确保非负

        prob_init = U[:, :k] * sqrt_S[np.newaxis, :]
        kp_init = Vt[:k, :].T * sqrt_S[np.newaxis, :]

        if k < self.embed_dim:
            prob_init = np.pad(prob_init, ((0, 0), (0, self.embed_dim - k)))
            kp_init = np.pad(kp_init, ((0, 0), (0, self.embed_dim - k)))

        with torch.no_grad():
            self.problem_embedding.weight.copy_(torch.from_numpy(prob_init).float())
            self.kp_embedding.weight.copy_(torch.from_numpy(kp_init).float())

        logger.info(f"Q-Matrix SVD 初始化完成，使用 {k}/{self.embed_dim} 个奇异值")

    def get_q_weights(self, problem_ids):
        """
        获取 Q 权重

        Args:
            problem_ids: [B, T] 或 [B]
        Returns:
            q_weights: [..., K] 范围 [0, 1]
        """
        prob_emb = self.problem_embedding(problem_ids)
        kp_emb = self.kp_embedding.weight
        logits = torch.matmul(prob_emb, kp_emb.T)
        return torch.sigmoid(logits)

    def get_problem_embedding(self, problem_ids):
        """获取题目 embedding"""
        return self.problem_embedding(problem_ids)

    def compute_regularization_loss(self):
        """BCE(Q_learned, Q_annotated) 作为正则化损失"""
        all_ids = torch.arange(self.num_problems, device=self.annotated_q.device)
        q_learned = self.get_q_weights(all_ids)
        return nn.functional.binary_cross_entropy(
            q_learned, self.annotated_q, reduction='mean'
        )
