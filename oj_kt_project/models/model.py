"""
重构后的知识追踪模型 — 题目级序列，无内层聚合

输入编码：verdict_embedding + attempt_embedding + score_feature_encoder
          + Q-Matrix题目嵌入 + 时间特征 + 学生能力 + 题目难度 + 算法类别
序列模型：LSTM / Transformer
注意力：KG引导注意力（可选）
输出：AC预测头 + 掌握度头 [B,K]
返回 2-tuple: (ac_logits, mastery)
"""
import torch
import torch.nn as nn
import logging

from .q_matrix import LearnableQMatrix
from .sequence_model import (
    build_sequence_model,
    KnowledgeGuidedAttention,
    AttentionLayer,
)
from .attempt_encoder import AttemptEncoder

logger = logging.getLogger(__name__)


class OJKnowledgeTracingModel(nn.Module):
    def __init__(self, config, num_kp, num_problems, init_q_matrix,
                 prerequisite_adj=None):
        super().__init__()
        self.config = config
        self.num_kp = num_kp
        self.num_problems = num_problems

        # ── Attempt Encoder（可选）──
        self.use_attempt_encoder = getattr(config, 'USE_ATTEMPT_ENCODER', False)
        if self.use_attempt_encoder:
            self.attempt_encoder = AttemptEncoder(config)

        # ── 输入编码器 ──
        self.verdict_embedding = nn.Embedding(config.NUM_VERDICT_TYPES, config.VERDICT_EMBED_DIM)
        self.use_attempt_embed = getattr(config, 'USE_ATTEMPT_EMBED', True)
        if self.use_attempt_embed:
            self.attempt_embedding = nn.Embedding(config.MAX_ATTEMPTS, config.ATTEMPT_EMBED_DIM)

        self.use_score_features = getattr(config, 'USE_SCORE_FEATURES', True)
        self.use_time_features = getattr(config, 'USE_TIME_FEATURES', True)
        self.use_student_features = getattr(config, 'USE_STUDENT_FEATURES', True)
        self.use_category_features = getattr(config, 'USE_CATEGORY_FEATURES', True)

        if self.use_score_features:
            self.score_encoder = nn.Sequential(
                nn.Linear(config.SCORE_FEATURE_INPUT_DIM, config.SCORE_FEATURE_DIM),
                nn.ReLU(),
            )
        if self.use_time_features:
            self.time_encoder = nn.Sequential(
                nn.Linear(config.TIME_FEATURE_INPUT_DIM, config.TIME_FEATURE_DIM),
                nn.ReLU(),
            )

        # ── Verdict 分布特征编码器 ──
        self.use_verdict_dist = getattr(config, 'USE_VERDICT_DIST', False)
        if self.use_verdict_dist:
            self.verdict_dist_encoder = nn.Sequential(
                nn.Linear(config.NUM_VERDICT_TYPES, getattr(config, 'VERDICT_DIST_DIM', 8)),
                nn.ReLU(),
            )

        # ── Verdict-conditioned 特征调制 ──
        # Verdict embedding 生成门控信号，调制 score/time 特征的解读
        self.use_verdict_modulation = getattr(config, 'USE_VERDICT_MODULATION', False)
        if self.use_verdict_modulation:
            modulated_dim = (
                (config.SCORE_FEATURE_DIM if self.use_score_features else 0) +
                (config.TIME_FEATURE_DIM if self.use_time_features else 0)
            )
            if modulated_dim > 0:
                self.verdict_gate = nn.Sequential(
                    nn.Linear(config.VERDICT_EMBED_DIM, modulated_dim),
                    nn.Sigmoid(),
                )
            else:
                self.use_verdict_modulation = False

        if self.use_student_features:
            self.student_encoder = nn.Sequential(
                nn.Linear(config.STUDENT_FEATURE_INPUT_DIM, config.STUDENT_FEATURE_DIM),
                nn.ReLU(),
            )
        self.difficulty_encoder = nn.Sequential(
            nn.Linear(config.PROBLEM_DIFFICULTY_INPUT_DIM, config.PROBLEM_DIFFICULTY_DIM),
            nn.ReLU(),
        )
        if self.use_category_features:
            self.category_embedding = nn.Embedding(
                config.NUM_ALGO_CATEGORIES, config.ALGO_CATEGORY_DIM,
            )

        # Q-Matrix（始终创建，用于掌握度头和 KG 注意力；嵌入拼接可选）
        self.use_qmatrix_embed = getattr(config, 'USE_QMATRIX_EMBED', True)
        self.q_matrix = LearnableQMatrix(num_problems, num_kp, config, init_q_matrix)

        # 融合维度
        if self.use_attempt_encoder:
            interaction_dim = config.ATTEMPT_ENCODER_OUTPUT_DIM
        else:
            interaction_dim = config.VERDICT_EMBED_DIM + (config.ATTEMPT_EMBED_DIM if self.use_attempt_embed else 0)

        self.fusion_input_dim = (
            interaction_dim +
            config.PROBLEM_DIFFICULTY_DIM +
            (config.QMATRIX_EMBED_DIM if self.use_qmatrix_embed else 0) +
            (config.SCORE_FEATURE_DIM if self.use_score_features else 0) +
            (config.TIME_FEATURE_DIM if self.use_time_features else 0) +
            (config.STUDENT_FEATURE_DIM if self.use_student_features else 0) +
            (config.ALGO_CATEGORY_DIM if self.use_category_features else 0) +
            (getattr(config, 'VERDICT_DIST_DIM', 8) if self.use_verdict_dist else 0)
        )

        # 融合投影 + 残差
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.fusion_input_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )

        # 序列模型
        self.sequence_model = build_sequence_model(config, config.HIDDEN_DIM)

        # 注意力层
        self.use_kg_attention = config.USE_KG_ATTENTION
        if self.use_kg_attention:
            self.attention = KnowledgeGuidedAttention(
                config.HIDDEN_DIM, config.KG_RELEVANCE_TEMPERATURE,
            )
        else:
            self.attention = AttentionLayer(config.HIDDEN_DIM)

        # ── 输出头 ──
        # AC 预测：融合 attended_output + next_problem 信息
        next_info_dim = (
            config.PROBLEM_DIFFICULTY_DIM +
            (config.QMATRIX_EMBED_DIM if self.use_qmatrix_embed else 0) +
            (config.ALGO_CATEGORY_DIM if self.use_category_features else 0)
        )
        self.ac_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM + next_info_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1),
        )

        # 掌握度预测头
        self.mastery_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, num_kp),
            nn.Sigmoid(),
        )

        self._log_param_count()

    def _log_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"模型参数: {total:,} (可训练: {trainable:,})")

    def get_q_regularization_loss(self):
        return self.q_matrix.compute_regularization_loss()

    def _compute_kg_relevance(self, history_problem_ids, next_problem_ids):
        """
        计算历史题目与下一题的知识相关性 [B, T]
        基于 Q-Matrix 余弦相似度
        """
        # history_q: [B, T, K], next_q: [B, K]
        history_q = self.q_matrix.get_q_weights(history_problem_ids)
        next_q = self.q_matrix.get_q_weights(next_problem_ids)

        # 余弦相似度
        history_norm = history_q / (history_q.norm(dim=-1, keepdim=True) + 1e-8)
        next_norm = next_q / (next_q.norm(dim=-1, keepdim=True) + 1e-8)

        # [B, T, K] @ [B, K, 1] → [B, T]
        relevance = torch.bmm(history_norm, next_norm.unsqueeze(-1)).squeeze(-1)
        return relevance

    def forward(self, batch):
        device = next(self.parameters()).device

        problem_ids = batch['problem_ids']           # [B, T]
        verdict_types = batch['verdict_types']       # [B, T]
        attempt_counts = batch['attempt_counts']     # [B, T]
        score_features = batch['score_features']     # [B, T, 3]
        time_features = batch['time_features']       # [B, T, 4]
        student_features = batch['student_features'] # [B, T, 2]
        problem_diff = batch['problem_difficulty']   # [B, T, 2]
        problem_cats = batch['problem_categories']   # [B, T]
        seq_lens = batch['seq_lens']                 # [B]
        next_problem_ids = batch['next_problem_ids'] # [B]
        next_problem_diff = batch['next_problem_difficulty']  # [B, 2]
        next_problem_cats = batch['next_problem_categories']  # [B]

        # ── 编码各特征 ──
        diff_enc = self.difficulty_encoder(problem_diff)          # [B, T, diff_dim]

        parts_common = [diff_enc]
        if self.use_qmatrix_embed:
            prob_emb = self.q_matrix.get_problem_embedding(problem_ids)  # [B, T, qmatrix_dim]
            parts_common.append(prob_emb)

        # Verdict embedding (用于 modulation 和 interaction)
        verdict_emb = self.verdict_embedding(verdict_types)       # [B, T, verdict_dim]

        if self.use_score_features:
            score_enc = self.score_encoder(score_features)        # [B, T, score_dim]
        if self.use_time_features:
            time_enc = self.time_encoder(time_features)           # [B, T, time_dim]

        # Verdict-conditioned 特征调制: verdict 类型影响 score/time 的解读
        if self.use_verdict_modulation:
            modulated_parts = []
            if self.use_score_features:
                modulated_parts.append(score_enc)
            if self.use_time_features:
                modulated_parts.append(time_enc)
            if modulated_parts:
                combined = torch.cat(modulated_parts, dim=-1)     # [B, T, score_dim + time_dim]
                gate = self.verdict_gate(verdict_emb)              # [B, T, score_dim + time_dim]
                combined = combined * gate                         # element-wise gating
                # Split back
                offset = 0
                if self.use_score_features:
                    score_enc = combined[..., offset:offset + score_enc.size(-1)]
                    offset += score_enc.size(-1)
                if self.use_time_features:
                    time_enc = combined[..., offset:offset + time_enc.size(-1)]

        if self.use_score_features:
            parts_common.append(score_enc)
        if self.use_time_features:
            parts_common.append(time_enc)
        if self.use_student_features:
            student_enc = self.student_encoder(student_features)  # [B, T, student_dim]
            parts_common.append(student_enc)
        if self.use_category_features:
            cat_emb = self.category_embedding(problem_cats)       # [B, T, cat_dim]
            parts_common.append(cat_emb)

        # Verdict 分布特征
        if self.use_verdict_dist and 'verdict_dist' in batch:
            verdict_dist_enc = self.verdict_dist_encoder(batch['verdict_dist'])  # [B, T, vd_dim]
            parts_common.append(verdict_dist_enc)

        # ── 融合 ──
        if self.use_attempt_encoder and 'attempt_verdicts' in batch:
            attempt_out = self.attempt_encoder(
                batch['attempt_verdicts'], batch['attempt_scores'], batch['attempt_lens'],
            )  # [B, T, attempt_encoder_output_dim]
            fused = torch.cat([attempt_out] + parts_common, dim=-1)
        else:
            interaction_parts = [verdict_emb]
            if self.use_attempt_embed:
                attempt_emb = self.attempt_embedding(attempt_counts)  # [B, T, attempt_dim]
                interaction_parts.append(attempt_emb)
            fused = torch.cat(interaction_parts + parts_common, dim=-1)

        fused = self.fusion_proj(fused)  # [B, T, hidden_dim]

        # ── 序列模型 ──
        seq_output, final_hidden = self.sequence_model(fused, seq_lens)

        # ── 注意力聚合 ──
        if self.use_kg_attention:
            kg_relevance = self._compute_kg_relevance(problem_ids, next_problem_ids)
            attended, _ = self.attention(seq_output, seq_lens, kg_relevance)
        else:
            attended, _ = self.attention(seq_output, seq_lens)

        # ── AC 预测 ──
        next_diff_enc = self.difficulty_encoder(next_problem_diff)              # [B, diff_dim]

        ac_parts = [attended, next_diff_enc]
        if self.use_qmatrix_embed:
            next_prob_emb = self.q_matrix.get_problem_embedding(next_problem_ids)  # [B, qmatrix_dim]
            ac_parts.append(next_prob_emb)
        if self.use_category_features:
            next_cat_emb = self.category_embedding(next_problem_cats)           # [B, cat_dim]
            ac_parts.append(next_cat_emb)

        ac_input = torch.cat(ac_parts, dim=-1)
        ac_logits = self.ac_head(ac_input)  # [B, 1]

        # ── 掌握度预测 ──
        mastery = self.mastery_head(attended)  # [B, K]

        return ac_logits, mastery
