"""
Baseline KT 模型 Wrapper

所有 baseline 模型适配到 oj_kt 的 Trainer 接口:
- forward(batch_dict) → (ac_logits [B,1], mastery_pred [B,K])
- get_q_regularization_loss() → 0.0

包含:
  经典 (2015-2020): DKT, DKT+, DKVMN, SAKT, GKT, SAINT
  注意力 (2020-2023): AKT, simpleKT, qDKT
  前沿 (2021-2024): LPKT, CL4KT, ATKT, CMKT, MambaKT
  公平对比: DKT+Features (与 MFKT 相同输入)
"""
import os
import math
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

# 通过 importlib 从绝对路径导入 baseline 模型，避免与本地 models/ 包冲突
_BASELINE_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..',
                 'knowledge-tracing-collection-pytorch', 'models')
)


def _import_from_file(module_name, filename):
    """从指定文件路径导入模块"""
    filepath = os.path.join(_BASELINE_MODELS_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dkt_mod = _import_from_file('baseline_dkt', 'dkt.py')
_dkt_plus_mod = _import_from_file('baseline_dkt_plus', 'dkt_plus.py')
_dkvmn_mod = _import_from_file('baseline_dkvmn', 'dkvmn.py')
_sakt_mod = _import_from_file('baseline_sakt', 'sakt.py')
_gkt_mod = _import_from_file('baseline_gkt', 'gkt.py')
_saint_mod = _import_from_file('baseline_saint', 'saint.py')

DKT = _dkt_mod.DKT
DKTPlus = _dkt_plus_mod.DKTPlus
DKVMN = _dkvmn_mod.DKVMN
SAKT = _sakt_mod.SAKT
GKTPAM = _gkt_mod.PAM
SAINT = _saint_mod.SAINT


class _SAKTFixed(SAKT):
    """修复 SAKT 的 causal_mask 设备问题 (原始代码在 CPU 上创建 mask)"""

    def forward(self, q, r, qry):
        x = q + self.num_q * r

        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]], device=q.device), diagonal=1
        ).bool()

        M = M + P

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze(-1)

        return p, attn_weights


class _SAINTFixed(SAINT):
    """修复 SAINT 的 mask 设备问题"""

    def forward(self, q, r):
        batch_size = r.shape[0]

        E = self.E(q).permute(1, 0, 2)

        R = self.R(r[:, :-1]).permute(1, 0, 2)
        S = self.S.repeat(batch_size, 1).unsqueeze(0)
        R = torch.cat([S, R], dim=0)

        P = self.P.unsqueeze(1)

        mask = self.transformer.generate_square_subsequent_mask(
            E.shape[0], device=q.device
        )
        R = self.transformer(
            E + P, R + P, mask, mask, mask
        )
        R = R.permute(1, 0, 2)

        p = torch.sigmoid(self.pred(R)).squeeze(-1)  # [B, T]

        return p


class BaselineWrapper(nn.Module):
    """
    基类: 从 oj_kt batch dict 提取 (q, r) 序列，调用内部 baseline 模型，
    将输出转换为 (ac_logits [B,1], mastery_pred [B, num_kp]) 格式。
    """
    is_baseline = True

    def __init__(self, inner_model, num_problems, num_kp):
        super().__init__()
        self.inner = inner_model
        self.num_problems = num_problems
        self.num_kp = num_kp

    def _extract_sequences(self, batch):
        """从 batch dict 提取 problem_ids, correctness, seq_lens, next_problem_ids"""
        problem_ids = batch['problem_ids']        # [B, T]
        verdict_types = batch['verdict_types']    # [B, T]
        seq_lens = batch['seq_lens']              # [B]
        next_problem_ids = batch['next_problem_ids']  # [B]

        # AC = verdict_type 0
        correctness = (verdict_types == 0).long()
        return problem_ids, correctness, seq_lens, next_problem_ids

    @staticmethod
    def _prob_to_logit(p):
        """prob → logit, 因为 Trainer 用 BCEWithLogitsLoss"""
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    def get_q_regularization_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, batch):
        raise NotImplementedError


class DKTBaseline(BaselineWrapper):
    """DKT: forward(q, r) → predictions [B, T, num_q] (sigmoid'd)"""

    def __init__(self, num_problems, num_kp, emb_size=64, hidden_size=64, dropout_rate=0.2):
        inner = DKT(num_q=num_problems, emb_size=emb_size,
                     hidden_size=hidden_size, dropout_rate=dropout_rate)
        super().__init__(inner, num_problems, num_kp)

    def forward(self, batch):
        q, r, seq_lens, next_q = self._extract_sequences(batch)
        B = q.size(0)

        predictions = self.inner(q, r)  # [B, T, num_q]

        # 取每个样本最后有效步的预测，再用 next_problem_id 索引
        last_idx = (seq_lens - 1).long()  # [B]
        # [B, num_q]
        last_step_pred = predictions[torch.arange(B, device=q.device), last_idx]
        # [B] — 对 next_problem_id 的预测概率
        prob = last_step_pred[torch.arange(B, device=q.device), next_q.long()]

        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)  # [B, 1]
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


class DKTPlusBaseline(BaselineWrapper):
    """DKT+: forward(q, r) → y [B, T, num_q] (sigmoid'd)"""

    def __init__(self, num_problems, num_kp, emb_size=64, hidden_size=64,
                 lambda_r=0.01, lambda_w1=0.03, lambda_w2=0.3):
        inner = DKTPlus(num_q=num_problems, emb_size=emb_size,
                        hidden_size=hidden_size, lambda_r=lambda_r,
                        lambda_w1=lambda_w1, lambda_w2=lambda_w2)
        super().__init__(inner, num_problems, num_kp)

    def forward(self, batch):
        q, r, seq_lens, next_q = self._extract_sequences(batch)
        B = q.size(0)

        y = self.inner(q, r)  # [B, T, num_q]

        last_idx = (seq_lens - 1).long()
        last_step_pred = y[torch.arange(B, device=q.device), last_idx]
        prob = last_step_pred[torch.arange(B, device=q.device), next_q.long()]

        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


class DKVMNBaseline(BaselineWrapper):
    """DKVMN: forward(q, r) → (p [B, T], Mv)"""

    def __init__(self, num_problems, num_kp, dim_s=50, size_m=20):
        inner = DKVMN(num_q=num_problems, dim_s=dim_s, size_m=size_m)
        super().__init__(inner, num_problems, num_kp)

    def forward(self, batch):
        q, r, seq_lens, next_q = self._extract_sequences(batch)
        B = q.size(0)

        p, _ = self.inner(q, r)  # p: [B, T], [T] when B=1, or scalar when B=1&T=1
        while p.dim() < 2:
            p = p.unsqueeze(0)

        last_idx = (seq_lens - 1).long()
        prob = p[torch.arange(B, device=q.device), last_idx]  # [B]

        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


class SAKTBaseline(BaselineWrapper):
    """SAKT: forward(q, r, qry) → (p [B, T], attn_weights)"""

    def __init__(self, num_problems, num_kp, n=50, d=64, num_attn_heads=4, dropout=0.2):
        inner = _SAKTFixed(num_q=num_problems, n=n, d=d,
                           num_attn_heads=num_attn_heads, dropout=dropout)
        super().__init__(inner, num_problems, num_kp)
        self._n = n

    def forward(self, batch):
        q, r, seq_lens, next_q = self._extract_sequences(batch)
        B, T = q.shape

        # SAKT 需要 qry 序列: 将序列右移一位，最后一位填 next_problem_id
        qry = torch.cat([q[:, 1:], next_q.unsqueeze(-1)], dim=1)  # [B, T]

        # SAKT 的位置编码 self.P 固定为 [n, d]，需要将序列 pad 到长度 n
        if T < self._n:
            pad_len = self._n - T
            q = torch.nn.functional.pad(q, (0, pad_len), value=0)
            r = torch.nn.functional.pad(r, (0, pad_len), value=0)
            qry = torch.nn.functional.pad(qry, (0, pad_len), value=0)

        p, _ = self.inner(q, r, qry)  # p: [B, n], [n] when B=1, scalar when B=1&n=1
        while p.dim() < 2:
            p = p.unsqueeze(0)

        last_idx = (seq_lens - 1).long()
        prob = p[torch.arange(B, device=q.device), last_idx]  # [B]

        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


class GKTBaseline(BaselineWrapper):
    """GKT (PAM): forward(q, r) → (y [B, T, num_q], h)"""

    def __init__(self, num_problems, num_kp, hidden_size=30, num_attn_heads=2):
        inner = GKTPAM(num_q=num_problems, hidden_size=hidden_size,
                       num_attn_heads=num_attn_heads, method="PAM")
        super().__init__(inner, num_problems, num_kp)

    def forward(self, batch):
        q, r, seq_lens, next_q = self._extract_sequences(batch)
        B = q.size(0)

        y, _ = self.inner(q, r)
        # GKT predict() uses squeeze() which mangles shapes when B=1:
        #   Normal: [B, T, num_q]
        #   B=1:    [num_q, T]  (from stacking T x [num_q] along dim=1)
        if B == 1 and y.dim() == 2:
            # [num_q, T] → [1, T, num_q]
            y = y.T.unsqueeze(0)

        last_idx = (seq_lens - 1).long()
        last_step_pred = y[torch.arange(B, device=q.device), last_idx]
        prob = last_step_pred[torch.arange(B, device=q.device), next_q.long()]

        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


class SAINTBaseline(BaselineWrapper):
    """SAINT: Encoder-Decoder Transformer for KT"""

    def __init__(self, num_problems, num_kp, n=50, d=64,
                 num_attn_heads=4, dropout=0.2, num_tr_layers=1):
        inner = _SAINTFixed(num_q=num_problems, n=n, d=d,
                            num_attn_heads=num_attn_heads, dropout=dropout,
                            num_tr_layers=num_tr_layers)
        super().__init__(inner, num_problems, num_kp)
        self._n = n

    def forward(self, batch):
        q, r, seq_lens, next_q = self._extract_sequences(batch)
        B, T = q.shape

        if T < self._n:
            pad_len = self._n - T
            q = torch.nn.functional.pad(q, (0, pad_len), value=0)
            r = torch.nn.functional.pad(r, (0, pad_len), value=0)

        p = self.inner(q, r)  # [B, n]
        while p.dim() < 2:
            p = p.unsqueeze(0)

        last_idx = (seq_lens - 1).long()
        prob = p[torch.arange(B, device=q.device), last_idx]

        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


class DKTWithFeaturesBaseline(nn.Module):
    """
    DKT + 多粒度特征 — 公平对比实验

    与 MFKT 接收相同的输入特征（verdict, score, time, difficulty, category, verdict_dist），
    但使用标准 DKT 的 LSTM + per-problem 预测头架构。
    用于验证 MFKT 的性能提升来自架构还是特征工程。
    """
    is_baseline = True

    def __init__(self, num_problems, num_kp,
                 hidden_size=64, dropout_rate=0.2,
                 num_verdict_types=6, verdict_embed_dim=16,
                 score_input_dim=3, score_dim=8,
                 time_input_dim=4, time_dim=16,
                 difficulty_input_dim=2, difficulty_dim=8,
                 num_categories=8, category_dim=8,
                 verdict_dist_dim=8):
        super().__init__()
        self.num_problems = num_problems
        self.num_kp = num_kp
        self.hidden_size = hidden_size

        # 与 MFKT 相同的特征编码器
        self.verdict_embedding = nn.Embedding(num_verdict_types, verdict_embed_dim)
        self.score_encoder = nn.Sequential(
            nn.Linear(score_input_dim, score_dim), nn.ReLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(time_input_dim, time_dim), nn.ReLU(),
        )
        self.difficulty_encoder = nn.Sequential(
            nn.Linear(difficulty_input_dim, difficulty_dim), nn.ReLU(),
        )
        self.category_embedding = nn.Embedding(num_categories, category_dim)
        self.verdict_dist_encoder = nn.Sequential(
            nn.Linear(num_verdict_types, verdict_dist_dim), nn.ReLU(),
        )

        # 融合维度 = 所有特征拼接
        fusion_dim = (
            verdict_embed_dim + score_dim + time_dim +
            difficulty_dim + category_dim + verdict_dist_dim
        )

        # 融合投影 → hidden_size
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 标准 DKT LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        # Per-problem 预测头（与 DKT 一致）
        self.out_layer = nn.Linear(hidden_size, num_problems)

    def get_q_regularization_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, batch):
        verdict_types = batch['verdict_types']       # [B, T]
        score_features = batch['score_features']     # [B, T, 3]
        time_features = batch['time_features']       # [B, T, 4]
        problem_diff = batch['problem_difficulty']   # [B, T, 2]
        problem_cats = batch['problem_categories']   # [B, T]
        seq_lens = batch['seq_lens']                 # [B]
        next_problem_ids = batch['next_problem_ids'] # [B]

        B, T = verdict_types.shape
        device = verdict_types.device

        # 编码各特征
        verdict_emb = self.verdict_embedding(verdict_types)    # [B, T, v_dim]
        score_enc = self.score_encoder(score_features)         # [B, T, s_dim]
        time_enc = self.time_encoder(time_features)            # [B, T, t_dim]
        diff_enc = self.difficulty_encoder(problem_diff)       # [B, T, d_dim]
        cat_emb = self.category_embedding(problem_cats)        # [B, T, c_dim]

        # Verdict 分布特征
        if 'verdict_dist' in batch:
            vd_enc = self.verdict_dist_encoder(batch['verdict_dist'])  # [B, T, vd_dim]
        else:
            vd_enc = torch.zeros(B, T, 8, device=device)

        # 融合
        fused = torch.cat([
            verdict_emb, score_enc, time_enc, diff_enc, cat_emb, vd_enc
        ], dim=-1)
        fused = self.fusion_proj(fused)  # [B, T, hidden]

        # LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            fused, seq_lens.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T,
        )
        lstm_out = self.dropout(lstm_out)  # [B, T, hidden]

        # Per-problem 预测 (DKT style)
        y = torch.sigmoid(self.out_layer(lstm_out))  # [B, T, num_problems]

        # 取最后有效步，索引 next_problem_id
        last_idx = (seq_lens - 1).long()
        last_pred = y[torch.arange(B, device=device), last_idx]  # [B, num_problems]
        prob = last_pred[torch.arange(B, device=device), next_problem_ids.long()]  # [B]

        # prob → logit
        prob = prob.clamp(1e-6, 1 - 1e-6)
        ac_logits = torch.log(prob / (1 - prob)).unsqueeze(-1)  # [B, 1]
        mastery = torch.zeros(B, self.num_kp, device=device)
        return ac_logits, mastery


class VanillaDKT(nn.Module):
    """标准 DKT (Piech 2015) — 不做任何自定义权重初始化"""

    def __init__(self, num_q, emb_size, hidden_size, dropout_rate=0.2):
        super().__init__()
        self.num_q = num_q
        self.interaction_emb = nn.Embedding(num_q * 2, emb_size, padding_idx=0)
        self.lstm_layer = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.out_layer = nn.Linear(hidden_size, num_q)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, q, r):
        x = q + self.num_q * r
        h = self.interaction_emb(x)
        h = self.dropout_layer(h)
        h, _ = self.lstm_layer(h)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y


class VanillaDKTBaseline(DKTBaseline):
    """Vanilla DKT Wrapper — 用 VanillaDKT 替换增强版 DKT，forward 直接继承 DKTBaseline"""

    def __init__(self, num_problems, num_kp, emb_size=64, hidden_size=64, dropout_rate=0.2):
        nn.Module.__init__(self)
        self.inner = VanillaDKT(num_q=num_problems, emb_size=emb_size,
                                hidden_size=hidden_size, dropout_rate=dropout_rate)
        self.num_problems = num_problems
        self.num_kp = num_kp


# ============================================================
# AKT (Context-Aware Attentive Knowledge Tracing, KDD 2020)
# ============================================================

class MonotonicAttentionBlock(nn.Module):
    """Attention block with exponential decay for monotonic attention"""

    def __init__(self, d, num_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d * 4, d), nn.Dropout(dropout),
        )
        self.decay = nn.Parameter(torch.tensor(0.1))

    def forward(self, query, key_value, causal_mask):
        B, T, d = query.shape
        positions = torch.arange(T, device=query.device).float()
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)
        decay_bias = -torch.abs(self.decay) * dist
        decay_bias = decay_bias.masked_fill(causal_mask, float('-inf'))
        attn_out, _ = self.attn(query, key_value, key_value, attn_mask=decay_bias)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class AKTModel(nn.Module):
    """AKT: Rasch-style embeddings + monotonic attention + knowledge retriever"""

    def __init__(self, num_q, d=64, num_heads=4, dropout=0.2, max_len=200):
        super().__init__()
        self.num_q = num_q
        self.d = d
        self.q_embed = nn.Embedding(num_q + 1, d, padding_idx=0)
        self.q_diff_embed = nn.Embedding(num_q + 1, d, padding_idx=0)
        self.interaction_embed = nn.Embedding(2 * num_q + 1, d, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d)
        self.knowledge_encoder = MonotonicAttentionBlock(d, num_heads, dropout)
        self.knowledge_retriever = MonotonicAttentionBlock(d, num_heads, dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, num_q),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, r):
        B, T = q.shape
        device = q.device
        question_emb = self.q_embed(q) + self.q_diff_embed(q)
        inter_emb = self.interaction_embed(q + self.num_q * r)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos = self.pos_embed(positions)
        question_emb = self.dropout(question_emb + pos)
        inter_emb = self.dropout(inter_emb + pos)
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        q_encoded = self.knowledge_encoder(question_emb, question_emb, causal_mask)
        retrieved = self.knowledge_retriever(q_encoded, inter_emb, causal_mask)
        combined = torch.cat([q_encoded, retrieved], dim=-1)
        return torch.sigmoid(self.out_proj(combined))


class _AKTSimpleKTBaseWrapper(nn.Module):
    """Shared wrapper base for AKT/simpleKT"""
    is_baseline = True

    def __init__(self, inner, num_problems, num_kp):
        super().__init__()
        self.inner = inner
        self.num_problems = num_problems
        self.num_kp = num_kp

    @staticmethod
    def _prob_to_logit(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    def get_q_regularization_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, batch):
        q = batch['problem_ids']
        vt = batch['verdict_types']
        seq_lens = batch['seq_lens']
        next_q = batch['next_problem_ids']
        r = (vt == 0).long()
        B = q.size(0)
        predictions = self.inner(q, r)
        last_idx = (seq_lens - 1).long()
        last_step_pred = predictions[torch.arange(B, device=q.device), last_idx]
        prob = last_step_pred[torch.arange(B, device=q.device), next_q.long()]
        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


class AKTBaseline(_AKTSimpleKTBaseWrapper):
    def __init__(self, num_problems, num_kp, d=64, num_heads=4, dropout=0.2):
        inner = AKTModel(num_q=num_problems, d=d, num_heads=num_heads, dropout=dropout)
        super().__init__(inner, num_problems, num_kp)


# ============================================================
# simpleKT (Liu et al., ICLR 2023)
# ============================================================

class SimpleKTModel(nn.Module):
    """simpleKT: Rasch embeddings + simple Transformer encoder"""

    def __init__(self, num_q, d=64, num_heads=4, num_layers=2, dropout=0.2, max_len=200):
        super().__init__()
        self.num_q = num_q
        self.q_embed = nn.Embedding(num_q + 1, d, padding_idx=0)
        self.q_diff_embed = nn.Embedding(num_q + 1, d, padding_idx=0)
        self.r_embed = nn.Embedding(2, d)
        self.interaction_proj = nn.Linear(2 * d, d)
        self.pos_embed = nn.Embedding(max_len, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation='relu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d, num_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, r):
        B, T = q.shape
        device = q.device
        q_emb = self.q_embed(q) + self.q_diff_embed(q)
        r_emb = self.r_embed(r)
        inter = self.interaction_proj(torch.cat([q_emb, r_emb], dim=-1))
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        inter = self.dropout(inter + self.pos_embed(positions))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        h = self.transformer(inter, mask=causal_mask, is_causal=True)
        return torch.sigmoid(self.out_proj(h))


class SimpleKTBaseline(_AKTSimpleKTBaseWrapper):
    def __init__(self, num_problems, num_kp, d=64, num_heads=4, num_layers=2, dropout=0.2):
        inner = SimpleKTModel(num_q=num_problems, d=d, num_heads=num_heads,
                              num_layers=num_layers, dropout=dropout)
        super().__init__(inner, num_problems, num_kp)


# ============================================================
# qDKT (Question-centric DKT, EDM 2023)
# ============================================================

class qDKTModel(nn.Module):
    """IRT-style: P(correct) = sigmoid(ability * discrimination - difficulty)"""

    def __init__(self, num_q, emb_size=64, hidden_size=64, dropout_rate=0.2):
        super().__init__()
        self.num_q = num_q
        self.interaction_emb = nn.Embedding(num_q * 2 + 1, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.question_difficulty = nn.Embedding(num_q + 1, 1, padding_idx=0)
        self.question_discrimination = nn.Embedding(num_q + 1, 1, padding_idx=0)
        nn.init.uniform_(self.question_discrimination.weight, 0.5, 2.0)
        self.ability_proj = nn.Linear(hidden_size, 1)

    def forward(self, q, r):
        B, T = q.shape
        device = q.device
        x = self.interaction_emb(q + self.num_q * r)
        h, _ = self.lstm(self.dropout(x))
        h = self.dropout(h)
        ability = self.ability_proj(h)
        all_q_ids = torch.arange(1, self.num_q + 1, device=device)
        difficulty = self.question_difficulty(all_q_ids).squeeze(-1)
        discrimination = self.question_discrimination(all_q_ids).squeeze(-1)
        logits = ability * discrimination.unsqueeze(0).unsqueeze(0) - difficulty.unsqueeze(0).unsqueeze(0)
        return torch.sigmoid(logits)


class qDKTBaseline(_AKTSimpleKTBaseWrapper):
    def __init__(self, num_problems, num_kp, emb_size=64, hidden_size=64, dropout_rate=0.2):
        inner = qDKTModel(num_q=num_problems, emb_size=emb_size,
                          hidden_size=hidden_size, dropout_rate=dropout_rate)
        super().__init__(inner, num_problems, num_kp)


# ============================================================
# LPKT (Learning Process-consistent KT, KDD 2021)
# ============================================================

class LPKTModel(nn.Module):
    """Explicit learning/forgetting gates with time-dependent decay"""

    def __init__(self, num_q, num_kc, d=64, dropout=0.2):
        super().__init__()
        self.num_q = num_q
        self.num_kc = num_kc
        self.d = d
        self.q_embed = nn.Embedding(num_q + 1, d, padding_idx=0)
        self.kc_embed = nn.Embedding(num_kc + 1, d, padding_idx=0)
        self.r_embed = nn.Embedding(2, d)
        self.learning_gate = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, 1), nn.Sigmoid())
        self.forgetting_gate = nn.Sequential(nn.Linear(d + 1, d), nn.Tanh(), nn.Linear(d, 1), nn.Sigmoid())
        self.pred_layer = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, num_q))

    def forward(self, q, r, time_gap=None):
        B, T = q.shape
        device = q.device
        if time_gap is None:
            time_gap = torch.full((B, T), 0.1, device=device)
        q_emb = self.q_embed(q)
        kc_ids = (q % self.num_kc) + 1
        kc_emb = self.kc_embed(kc_ids)
        r_emb = self.r_embed(r)
        h_kc = torch.zeros(B, self.num_kc, self.d, device=device)
        predictions_list = []
        for t in range(T):
            q_t, kc_t, r_t = q_emb[:, t], kc_emb[:, t], r_emb[:, t]
            gap_t = time_gap[:, t:t+1]
            h_kc_t = h_kc.mean(dim=1)
            forget_rate = self.forgetting_gate(torch.cat([kc_t, gap_t], dim=-1))
            h_kc_t = h_kc_t * forget_rate
            learn_rate = self.learning_gate(torch.cat([q_t, r_t, h_kc_t], dim=-1))
            delta_h = learn_rate * (r_t - h_kc_t)
            h_kc_t_new = h_kc_t + delta_h
            h_kc = h_kc + delta_h.unsqueeze(1) * 0.1
            predictions_list.append(torch.sigmoid(self.pred_layer(torch.cat([q_t, h_kc_t_new], dim=-1))))
        return torch.stack(predictions_list, dim=1)


class LPKTBaseline(nn.Module):
    is_baseline = True

    def __init__(self, num_problems, num_kp, d=64, dropout=0.2):
        super().__init__()
        self.inner = LPKTModel(num_q=num_problems, num_kc=num_kp, d=d, dropout=dropout)
        self.num_problems = num_problems
        self.num_kp = num_kp

    def get_q_regularization_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    @staticmethod
    def _prob_to_logit(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    def forward(self, batch):
        q, vt, seq_lens, next_q = batch['problem_ids'], batch['verdict_types'], batch['seq_lens'], batch['next_problem_ids']
        r = (vt == 0).long()
        B = q.size(0)
        time_gap = batch['time_features'][:, :, 0] if 'time_features' in batch else None
        predictions = self.inner(q, r, time_gap)
        last_idx = (seq_lens - 1).long()
        last_step_pred = predictions[torch.arange(B, device=q.device), last_idx]
        prob = last_step_pred[torch.arange(B, device=q.device), next_q.long()]
        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


# ============================================================
# CL4KT (Contrastive Learning for KT, WWW 2022)
# ============================================================

class CL4KTModel(nn.Module):
    """DKT backbone + contrastive projection head"""

    def __init__(self, num_q, emb_size=64, hidden_size=64, dropout_rate=0.2):
        super().__init__()
        self.num_q = num_q
        self.interaction_emb = nn.Embedding(num_q * 2 + 1, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.pred_head = nn.Linear(hidden_size, num_q)
        self.proj_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 128))
        self.temperature = 0.05

    def encode(self, q, r):
        x = self.interaction_emb(q + self.num_q * r)
        h, _ = self.lstm(self.dropout(x))
        return self.dropout(h)

    def forward(self, q, r):
        return torch.sigmoid(self.pred_head(self.encode(q, r)))

    def contrastive_forward(self, q1, r1, q2, r2):
        h1, h2 = self.encode(q1, r1), self.encode(q2, r2)
        z1 = F.normalize(self.proj_head(h1[:, -1]), dim=-1)
        z2 = F.normalize(self.proj_head(h2[:, -1]), dim=-1)
        logits = torch.mm(z1, z2.t()) / self.temperature
        contrastive_loss = F.cross_entropy(logits, torch.arange(z1.size(0), device=z1.device))
        return torch.sigmoid(self.pred_head(h1)), contrastive_loss


class CL4KTBaseline(nn.Module):
    is_baseline = True

    def __init__(self, num_problems, num_kp, emb_size=64, hidden_size=64, dropout_rate=0.2):
        super().__init__()
        self.inner = CL4KTModel(num_q=num_problems, emb_size=emb_size,
                                hidden_size=hidden_size, dropout_rate=dropout_rate)
        self.num_problems = num_problems
        self.num_kp = num_kp

    def get_q_regularization_loss(self):
        return getattr(self, '_contrastive_loss', torch.tensor(0.0, device=next(self.parameters()).device))

    @staticmethod
    def _prob_to_logit(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    def forward(self, batch):
        q, vt, seq_lens, next_q = batch['problem_ids'], batch['verdict_types'], batch['seq_lens'], batch['next_problem_ids']
        r = (vt == 0).long()
        B = q.size(0)
        if self.training:
            # Simple crop augmentation
            T = q.size(1)
            crop_len = max(3, int(T * np.random.uniform(0.7, 1.0))) if T > 3 else T
            start = np.random.randint(0, T - crop_len + 1) if T > crop_len else 0
            q_aug, r_aug = q[:, start:start+crop_len], r[:, start:start+crop_len]
            predictions, cl_loss = self.inner.contrastive_forward(q, r, q_aug, r_aug)
            self._contrastive_loss = cl_loss * 0.1
        else:
            predictions = self.inner(q, r)
            self._contrastive_loss = torch.tensor(0.0, device=q.device)
        last_idx = (seq_lens - 1).long()
        last_step_pred = predictions[torch.arange(B, device=q.device), last_idx]
        prob = last_step_pred[torch.arange(B, device=q.device), next_q.long()]
        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


# ============================================================
# ATKT (Adversarial Training for KT, IJCAI 2021)
# ============================================================

class ATKTModel(nn.Module):
    """DKT + adversarial perturbation on embeddings"""

    def __init__(self, num_q, emb_size=64, hidden_size=64, dropout_rate=0.2):
        super().__init__()
        self.num_q = num_q
        self.interaction_emb = nn.Embedding(num_q * 2 + 1, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_layer = nn.Linear(hidden_size, num_q)
        self.epsilon = 0.01

    def forward(self, q, r, adversarial=False):
        emb = self.interaction_emb(q + self.num_q * r)
        if adversarial and self.training:
            emb = emb + torch.randn_like(emb) * self.epsilon
        h, _ = self.lstm(self.dropout(emb))
        return torch.sigmoid(self.out_layer(self.dropout(h)))


class ATKTBaseline(_AKTSimpleKTBaseWrapper):
    def __init__(self, num_problems, num_kp, emb_size=64, hidden_size=64, dropout_rate=0.2):
        inner = ATKTModel(num_q=num_problems, emb_size=emb_size,
                          hidden_size=hidden_size, dropout_rate=dropout_rate)
        super().__init__(inner, num_problems, num_kp)

    def forward(self, batch):
        q, vt, seq_lens, next_q = batch['problem_ids'], batch['verdict_types'], batch['seq_lens'], batch['next_problem_ids']
        r = (vt == 0).long()
        B = q.size(0)
        predictions = self.inner(q, r, adversarial=self.training)
        last_idx = (seq_lens - 1).long()
        last_step_pred = predictions[torch.arange(B, device=q.device), last_idx]
        prob = last_step_pred[torch.arange(B, device=q.device), next_q.long()]
        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


# ============================================================
# CMKT (Causal Mask KT, AAAI 2024)
# ============================================================

class CMKTModel(nn.Module):
    """Transformer with causal + knowledge dependency masks"""

    def __init__(self, num_q, num_kc, d=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_q = num_q
        self.num_kc = num_kc
        self.d = d
        self.q_embed = nn.Embedding(num_q + 1, d, padding_idx=0)
        self.kc_embed = nn.Embedding(num_kc + 1, d, padding_idx=0)
        self.r_embed = nn.Embedding(2, d)
        self.interaction_proj = nn.Linear(3 * d, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True, activation='relu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pred_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, num_q))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, r, kc_ids=None):
        B, T = q.shape
        device = q.device
        if kc_ids is None:
            kc_ids = (q % self.num_kc).clamp(1, self.num_kc)
        inter = self.interaction_proj(torch.cat([self.q_embed(q), self.kc_embed(kc_ids), self.r_embed(r)], dim=-1))
        inter = self.dropout(inter)
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        h = self.transformer(inter, mask=causal_mask, is_causal=True)
        return torch.sigmoid(self.pred_head(h))


class CMKTBaseline(nn.Module):
    is_baseline = True

    def __init__(self, num_problems, num_kp, d=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.inner = CMKTModel(num_q=num_problems, num_kc=num_kp, d=d,
                               num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.num_problems = num_problems
        self.num_kp = num_kp

    def get_q_regularization_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    @staticmethod
    def _prob_to_logit(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    def forward(self, batch):
        q, vt, seq_lens, next_q = batch['problem_ids'], batch['verdict_types'], batch['seq_lens'], batch['next_problem_ids']
        r = (vt == 0).long()
        B = q.size(0)
        kc_ids = (q % self.num_kp).clamp(1, self.num_kp)
        predictions = self.inner(q, r, kc_ids)
        last_idx = (seq_lens - 1).long()
        last_step_pred = predictions[torch.arange(B, device=q.device), last_idx]
        prob = last_step_pred[torch.arange(B, device=q.device), next_q.long()]
        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=q.device)
        return ac_logits, mastery


# ============================================================
# MambaKT (Mamba-based KT, 2024)
# ============================================================

class MambaBlock(nn.Module):
    """Simplified Mamba SSM block (PyTorch, no CUDA kernels)"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self._selective_scan(x_conv)
        return self.out_proj(x_ssm * F.silu(z))

    def _selective_scan(self, x):
        B, T, D = x.shape
        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        for t in range(T):
            x_t = x[:, t]
            dt = F.softplus(self.dt_proj(x_t))
            B_t = self.x_proj(x_t).unsqueeze(1)
            C_t = self.x_proj(x_t).unsqueeze(1)
            h = h + dt.unsqueeze(-1) * (torch.einsum('ds,bds->bds', self.A, h) +
                                         B_t.transpose(1, 2) * x_t.unsqueeze(-1))
            y_t = torch.einsum('bds,bds->bd', C_t.expand_as(h), h) + self.D * x_t
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


class MambaKTModel(nn.Module):
    """KT with Mamba SSM: linear complexity O(TN)"""

    def __init__(self, num_q, emb_size=64, d_model=64, d_state=16, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_q = num_q
        self.interaction_emb = nn.Embedding(num_q * 2 + 1, emb_size, padding_idx=0)
        self.emb_proj = nn.Linear(emb_size, d_model) if emb_size != d_model else nn.Identity()
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state=d_state) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.pred_head = nn.Linear(d_model, num_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, r):
        x = self.emb_proj(self.interaction_emb(q + self.num_q * r))
        x = self.dropout(x)
        for layer in self.layers:
            x = self.norm(layer(x) + x)
        return torch.sigmoid(self.pred_head(x))


class MambaKTBaseline(_AKTSimpleKTBaseWrapper):
    def __init__(self, num_problems, num_kp, emb_size=64, d_model=64, d_state=16,
                 num_layers=2, dropout=0.2):
        inner = MambaKTModel(num_q=num_problems, emb_size=emb_size, d_model=d_model,
                             d_state=d_state, num_layers=num_layers, dropout=dropout)
        super().__init__(inner, num_problems, num_kp)
