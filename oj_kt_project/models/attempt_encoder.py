"""
AttemptEncoder — 对每道题的完整提交序列（如 CE→WA→WA→TLE→AC）编码为单个向量

内层 GRU 编码器，捕捉调试过程模式。
"""
import torch
import torch.nn as nn


class AttemptEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout_rate = config.DROPOUT

        self.verdict_emb = nn.Embedding(config.NUM_VERDICT_TYPES, 8)
        self.score_proj = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
        )
        gru_input = 16  # 8 + 8
        self.input_dropout = nn.Dropout(config.DROPOUT)
        self.gru = nn.GRU(
            input_size=gru_input,
            hidden_size=config.ATTEMPT_ENCODER_HIDDEN,
            batch_first=True,
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.ATTEMPT_ENCODER_HIDDEN, config.ATTEMPT_ENCODER_OUTPUT_DIM),
            nn.ReLU(),
        )

    def forward(self, attempt_verdicts, attempt_scores, attempt_lens):
        """
        Args:
            attempt_verdicts: [B, T, A] long
            attempt_scores:   [B, T, A, 3] float
            attempt_lens:     [B, T] long
        Returns:
            [B, T, output_dim]
        """
        B, T, A = attempt_verdicts.shape
        # flatten to [B*T, A, ...]
        av = attempt_verdicts.reshape(B * T, A)
        asc = attempt_scores.reshape(B * T, A, 3)
        al = attempt_lens.reshape(B * T)

        # 训练时随机丢弃非末尾 attempt 步（数据增强）
        if self.training and A > 1:
            # 对每个序列，以 dropout_rate 概率 mask 掉非最后一步
            keep_mask = torch.rand(B * T, A, device=av.device) > self.dropout_rate
            # 始终保留最后有效步
            last_idx = (al.clamp(min=1) - 1).unsqueeze(-1)  # [B*T, 1]
            keep_mask.scatter_(1, last_idx, True)
            # padding 位置也保持不变（不影响）
            av = av * keep_mask.long()
            asc = asc * keep_mask.unsqueeze(-1).float()

        # embed
        v_emb = self.verdict_emb(av)        # [B*T, A, 8]
        s_enc = self.score_proj(asc)         # [B*T, A, 8]
        gru_in = torch.cat([v_emb, s_enc], dim=-1)  # [B*T, A, 16]
        gru_in = self.input_dropout(gru_in)

        # GRU forward
        gru_out, _ = self.gru(gru_in)       # [B*T, A, hidden]

        # 取每个序列最后有效步的 hidden state
        al_clamped = al.clamp(min=1) - 1     # [B*T]
        idx = al_clamped.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, gru_out.size(-1))
        last_hidden = gru_out.gather(1, idx).squeeze(1)  # [B*T, hidden]

        # 对 lens==0 的位置输出零向量
        mask = (al == 0).unsqueeze(-1)
        last_hidden = last_hidden.masked_fill(mask, 0.0)

        out = self.output_proj(last_hidden)  # [B*T, output_dim]
        return out.reshape(B, T, -1)
