"""
序列模型 - LSTM 和 Transformer (SAINT-style)
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class LSTMSequenceModel(nn.Module):
    """LSTM 序列模型"""
    def __init__(self, config, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, sequence_input, seq_lens):
        """
        Args:
            sequence_input: [batch_size, seq_len, input_dim]
            seq_lens: [batch_size]
        Returns:
            sequence_output: [batch_size, seq_len, hidden_dim]
            final_hidden: [batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = sequence_input.shape

        packed_input = nn.utils.rnn.pack_padded_sequence(
            sequence_input, seq_lens.cpu(),
            batch_first=True, enforce_sorted=False,
        )
        packed_output, (hidden, _) = self.lstm(packed_input)
        sequence_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len,
        )
        sequence_output = self.dropout(sequence_output)
        final_hidden = hidden[-1]
        return sequence_output, final_hidden


class TransformerSequenceModel(nn.Module):
    """SAINT-style Transformer 序列模型，可学习位置编码 + causal mask"""
    def __init__(self, config, input_dim):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.input_projection = nn.Linear(input_dim, config.HIDDEN_DIM)

        max_pos_len = config.MAX_PROBLEM_SEQ_LEN
        self.positional_encoding = nn.Embedding(max_pos_len, config.HIDDEN_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.HIDDEN_DIM,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.FF_DIM,
            dropout=config.DROPOUT,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.NUM_TRANSFORMER_LAYERS,
        )
        self.dropout = nn.Dropout(config.DROPOUT)
        self.layer_norm = nn.LayerNorm(config.HIDDEN_DIM)

    @staticmethod
    def _generate_causal_mask(seq_len, device):
        """生成 causal mask（上三角为 True → 被屏蔽）"""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    @staticmethod
    def _generate_padding_mask(seq_lens, max_len, device):
        """生成 padding mask（True → 被屏蔽）"""
        arange = torch.arange(max_len, device=device).unsqueeze(0)
        return arange >= seq_lens.unsqueeze(1)

    def forward(self, sequence_input, seq_lens):
        batch_size, seq_len, _ = sequence_input.shape
        device = sequence_input.device

        x = self.input_projection(sequence_input)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.positional_encoding(positions)
        x = self.dropout(x)

        causal_mask = self._generate_causal_mask(seq_len, device)
        padding_mask = self._generate_padding_mask(seq_lens, seq_len, device)

        sequence_output = self.transformer_encoder(
            x, mask=causal_mask, src_key_padding_mask=padding_mask,
        )
        sequence_output = self.layer_norm(sequence_output)

        last_indices = (seq_lens - 1).long().clamp(min=0)
        final_hidden = sequence_output[
            torch.arange(batch_size, device=device), last_indices
        ]
        return sequence_output, final_hidden


class AttentionLayer(nn.Module):
    """加性注意力层，用于聚合序列（LSTM 模式下使用）"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence_output, seq_lens):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_dim]
            seq_lens: [batch_size]
        Returns:
            attended_output: [batch_size, hidden_dim]
            attention_weights: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = sequence_output.shape

        scores = self.attention(sequence_output).squeeze(-1)  # [B, T]

        mask = torch.arange(seq_len, device=seq_lens.device).unsqueeze(0)
        mask = mask < seq_lens.unsqueeze(1)  # [B, T] True=valid

        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = torch.softmax(scores, dim=1)

        attended_output = torch.sum(
            sequence_output * attention_weights.unsqueeze(-1), dim=1,
        )
        return attended_output, attention_weights


class KnowledgeGuidedAttention(nn.Module):
    """
    知识引导注意力层

    在标准加性注意力基础上，加入知识相关性偏置：
    score[b,t] = content_score[b,t] + temperature * kg_relevance[b,t]

    其中 kg_relevance 由 Q-Matrix 余弦相似度计算，表示历史题目与下一题的知识关联程度。
    论文参考：领域知识引导的注意力知识追踪（学习相关性矩阵）
    """
    def __init__(self, hidden_dim, temperature_init=1.0):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # 可学习温度参数，控制知识相关性偏置的强度
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

    def forward(self, sequence_output, seq_lens, kg_relevance=None):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_dim]
            seq_lens: [batch_size]
            kg_relevance: [batch_size, seq_len] 知识相关性分数 (0~1)，可选
        Returns:
            attended_output: [batch_size, hidden_dim]
            attention_weights: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = sequence_output.shape

        scores = self.attention(sequence_output).squeeze(-1)  # [B, T]

        # 注入知识相关性偏置
        if kg_relevance is not None:
            scores = scores + self.temperature * kg_relevance

        mask = torch.arange(seq_len, device=seq_lens.device).unsqueeze(0)
        mask = mask < seq_lens.unsqueeze(1)  # [B, T] True=valid

        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = torch.softmax(scores, dim=1)

        attended_output = torch.sum(
            sequence_output * attention_weights.unsqueeze(-1), dim=1,
        )
        return attended_output, attention_weights


def build_sequence_model(config, input_dim):
    """工厂函数：根据 config.MODEL_TYPE 选择序列模型"""
    model_type = config.MODEL_TYPE.lower()
    if model_type == 'transformer':
        return TransformerSequenceModel(config, input_dim)
    elif model_type == 'lstm':
        return LSTMSequenceModel(config, input_dim)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，请选择 'transformer' 或 'lstm'")
