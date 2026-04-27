"""
Code-DKT Baseline 实现

基于 Code-DKT (Liu et al., 2022) 的核心思想:
- 从学生提交的 Java 源代码中提取 token 类型序列
- 用小型 LSTM 编码器将 token 序列压缩为固定维度向量
- 将代码特征与传统 DKT 的交互嵌入融合

适配到 oj_kt 的 Trainer 接口:
  forward(batch_dict) → (ac_logits [B,1], mastery_pred [B,K])
"""
import torch
import torch.nn as nn
import numpy as np


class CodeEncoder(nn.Module):
    """将 token 类型序列编码为固定维度向量"""

    def __init__(self, num_token_types, token_emb_dim=16, code_hidden_dim=32):
        super().__init__()
        self.token_emb = nn.Embedding(num_token_types + 1, token_emb_dim, padding_idx=0)
        self.code_lstm = nn.LSTM(token_emb_dim, code_hidden_dim, batch_first=True)
        self.out_dim = code_hidden_dim

    def forward(self, token_ids, token_lens):
        """
        token_ids: [B, T, max_code_len] — 每个时间步的代码 token 序列
        token_lens: [B, T] — 每个代码的实际 token 数
        返回: [B, T, code_hidden_dim]
        """
        B, T, L = token_ids.shape
        # 展平为 [B*T, L]
        flat_ids = token_ids.view(B * T, L)
        flat_lens = token_lens.view(B * T)

        emb = self.token_emb(flat_ids)  # [B*T, L, emb_dim]
        # 用 pack_padded_sequence 处理变长序列
        # 但为了简单和效率，直接跑 LSTM 取最后隐状态
        _, (h, _) = self.code_lstm(emb)  # h: [1, B*T, hidden]
        code_feat = h.squeeze(0)  # [B*T, hidden]

        return code_feat.view(B, T, -1)  # [B, T, code_hidden_dim]


class CodeDKT(nn.Module):
    """
    Code-DKT: DKT + 代码特征融合

    interaction_emb(q + num_q * r) 与 code_feature 拼接后送入主 LSTM
    """

    def __init__(self, num_q, num_token_types, emb_size=64, hidden_size=64,
                 code_emb_dim=16, code_hidden_dim=32, dropout_rate=0.2):
        super().__init__()
        self.num_q = num_q
        self.interaction_emb = nn.Embedding(num_q * 2, emb_size, padding_idx=0)
        self.code_encoder = CodeEncoder(num_token_types, code_emb_dim, code_hidden_dim)

        # 融合层: interaction_emb + code_feature → hidden
        fused_dim = emb_size + code_hidden_dim
        self.lstm_layer = nn.LSTM(fused_dim, hidden_size, batch_first=True)
        self.out_layer = nn.Linear(hidden_size, num_q)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, r, token_ids, token_lens):
        """
        q: [B, T] problem ids
        r: [B, T] correctness (0/1)
        token_ids: [B, T, max_code_len]
        token_lens: [B, T]
        返回: predictions [B, T, num_q] (sigmoid'd)
        """
        x = q + self.num_q * r
        interaction = self.interaction_emb(x)  # [B, T, emb_size]
        interaction = self.dropout(interaction)

        code_feat = self.code_encoder(token_ids, token_lens)  # [B, T, code_hidden]
        code_feat = self.dropout(code_feat)

        # 融合
        fused = torch.cat([interaction, code_feat], dim=-1)  # [B, T, emb+code_hidden]

        h, _ = self.lstm_layer(fused)
        h = self.dropout(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y


class CodeDKTBaseline(nn.Module):
    """
    Code-DKT Wrapper — 适配到 oj_kt Trainer 接口

    forward(batch) → (ac_logits [B,1], mastery_pred [B,K])
    """
    is_baseline = True

    def __init__(self, num_problems, num_kp, num_token_types=12,
                 emb_size=64, hidden_size=64, code_emb_dim=16,
                 code_hidden_dim=32, dropout_rate=0.2):
        super().__init__()
        self.inner = CodeDKT(
            num_q=num_problems,
            num_token_types=num_token_types,
            emb_size=emb_size,
            hidden_size=hidden_size,
            code_emb_dim=code_emb_dim,
            code_hidden_dim=code_hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.num_problems = num_problems
        self.num_kp = num_kp

    @staticmethod
    def _prob_to_logit(p):
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    def get_q_regularization_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, batch):
        problem_ids = batch['problem_ids']          # [B, T]
        verdict_types = batch['verdict_types']      # [B, T]
        seq_lens = batch['seq_lens']                # [B]
        next_q = batch['next_problem_ids']          # [B]
        token_ids = batch['code_token_ids']         # [B, T, max_code_len]
        token_lens = batch['code_token_lens']       # [B, T]

        correctness = (verdict_types == 0).long()
        B = problem_ids.size(0)

        predictions = self.inner(problem_ids, correctness, token_ids, token_lens)

        last_idx = (seq_lens - 1).long()
        last_step_pred = predictions[torch.arange(B, device=problem_ids.device), last_idx]
        prob = last_step_pred[torch.arange(B, device=problem_ids.device), next_q.long()]

        ac_logits = self._prob_to_logit(prob).unsqueeze(-1)
        mastery = torch.zeros(B, self.num_kp, device=problem_ids.device)
        return ac_logits, mastery
