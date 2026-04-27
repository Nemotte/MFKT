import os
import numpy as np

import torch
from torch import nn
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class DKTPlus(nn.Module):
    """
    DKT+ 模型

    Args:
        num_q:      题目（或KC）的总数量
        emb_size:   交互 embedding 维度
        hidden_size: LSTM 隐层维度
        lambda_r:   当前题一致性损失的权重
        lambda_w1:  相邻时间步 L1 正则的权重
        lambda_w2:  相邻时间步 L2^2 正则的权重
    """

    def __init__(self, num_q, emb_size, hidden_size,
                 lambda_r, lambda_w1, lambda_w2):
        super().__init__()

        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2

        # q ∈ [0, num_q-1], r ∈ {0,1}
        # 交互索引 x = q + num_q * r ∈ [0, num_q*2-1]
        self.interaction_emb = nn.Embedding(num_q * 2, emb_size)

        self.lstm_layer = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # 对所有题目做预测：[B, T, H] -> [B, T, num_q]
        self.out_layer = nn.Linear(hidden_size, num_q)
        self.dropout_layer = nn.Dropout()

    def forward(self, q, r):
        """
        Args:
            q: [batch_size, seq_len] 题目索引序列
            r: [batch_size, seq_len] 作答结果序列 (0/1)

        Returns:
            y: [batch_size, seq_len, num_q]
               对每个时间步、每个题目的掌握程度预测（sigmoid 后的概率）
        """
        # 交互编码
        x = q + self.num_q * r  # [B, T]
        emb = self.interaction_emb(x.long())  # [B, T, E]

        # LSTM 编码
        h, _ = self.lstm_layer(emb)  # [B, T, H]

        # 输出到所有题目
        logits = self.out_layer(h)   # [B, T, num_q]
        logits = self.dropout_layer(logits)
        y = torch.sigmoid(logits)

        return y

    def _select_question_prob(self, y, q_indices):
        """
        从 y 中取出对应题目的预测概率。

        Args:
            y: [B, T, num_q]
            q_indices: [B, T]，每个时间步对应的题目 id

        Returns:
            probs: [B, T]，每个时间步对应题目的预测概率
        """
        # 生成 one-hot，然后在最后一维上加权求和
        # 注意：collate_fn 已经把 padding 位置的 q 置为 0 且 mask 为 False，
        # 这里产生的值会被后续 mask 掉，所以不影响结果。
        oh = one_hot(q_indices.long(), num_classes=self.num_q)  # [B, T, num_q]
        probs = (y * oh).sum(dim=-1)  # [B, T]
        return probs

    def train_model(self, train_loader, test_loader,
                    num_epochs, opt, ckpt_path):
        """
        训练入口，接口与现有 train.py 完全兼容。

        Args:
            train_loader: DataLoader，返回 (q, r, qshft, rshft, mask)
            test_loader:  DataLoader，返回 (q, r, qshft, rshft, mask)
            num_epochs:   训练轮数
            opt:          优化器
            ckpt_path:    保存模型参数的目录

        Returns:
            aucs:        每个 epoch 的测试 AUC 列表
            loss_means:  每个 epoch 的平均 loss 列表
        """
        aucs = []
        loss_means = []

        best_auc = 0.0

        for epoch in range(1, num_epochs + 1):
            self.train()
            epoch_losses = []

            # ----------- 训练阶段 -----------
            for batch in train_loader:
                q, r, qshft, rshft, m = batch  # 形状：[B, T]

                # 前向传播：得到对所有题目的预测
                y_all = self(q.long(), r.long())  # [B, T, num_q]

                # 当前题目 / 下一题目的预测概率
                y_curr = self._select_question_prob(y_all, q)
                y_next = self._select_question_prob(y_all, qshft)

                # 按 mask 取出有效位置
                m_bool = m.bool()
                y_curr = torch.masked_select(y_curr, m_bool)
                y_next = torch.masked_select(y_next, m_bool)
                r_curr = torch.masked_select(r, m_bool)
                r_next = torch.masked_select(rshft, m_bool)

                # 平滑正则项：时间维度上的差
                # delta: [B, T-1, num_q]
                delta = y_all[:, 1:, :] - y_all[:, :-1, :]

                # L1：对题目维度求 1 范数 -> [B, T-1]
                loss_w1_all = torch.norm(delta, p=1, dim=-1)
                # L2^2：对题目维度求 2 范数后平方 -> [B, T-1]
                loss_w2_all = torch.norm(delta, p=2, dim=-1).pow(2)

                # 对应的 mask（时间上要去掉第一步）
                m_delta = m_bool[:, 1:]

                loss_w1 = torch.masked_select(loss_w1_all, m_delta)
                loss_w2 = torch.masked_select(loss_w2_all, m_delta)

                # 清梯度
                opt.zero_grad()

                # 三部分 loss
                # 1. 下一题预测的 BCE
                loss_main = binary_cross_entropy(y_next, r_next)

                # 2. 当前题预测与当前作答的一致性（论文里的 consistency）
                loss_r = binary_cross_entropy(y_curr, r_curr)

                # 3. 时序平滑（L1 + L2^2）
                #    注意除以 num_q，让正则规模与题目数量无关
                loss_smooth = (
                    self.lambda_w1 * loss_w1.mean() / self.num_q +
                    self.lambda_w2 * loss_w2.mean() / self.num_q
                )

                total_loss = loss_main + self.lambda_r * loss_r + loss_smooth

                total_loss.backward()
                opt.step()

                epoch_losses.append(total_loss.detach().cpu().numpy())

            # 该 epoch 平均 loss
            mean_loss = float(np.mean(epoch_losses))

            # ----------- 测试阶段 -----------
            self.eval()
            with torch.no_grad():
                all_pred = []
                all_true = []

                for batch in test_loader:
                    q, r, qshft, rshft, m = batch

                    y_all = self(q.long(), r.long())
                    y_next = self._select_question_prob(y_all, qshft)

                    m_bool = m.bool()
                    y_next = torch.masked_select(y_next, m_bool).detach().cpu()
                    r_next = torch.masked_select(rshft, m_bool).detach().cpu()

                    all_pred.append(y_next)
                    all_true.append(r_next)

                all_pred = torch.cat(all_pred).numpy()
                all_true = torch.cat(all_true).numpy()

                auc = metrics.roc_auc_score(
                    y_true=all_true,
                    y_score=all_pred
                )

            print(f"Epoch: {epoch}, AUC: {auc:.6f}, Loss Mean: {mean_loss:.6f}")

            # 保存最优 AUC 的模型
            if auc > best_auc:
                best_auc = auc
                torch.save(
                    self.state_dict(),
                    os.path.join(ckpt_path, "model.ckpt"),
                )

            aucs.append(auc)
            loss_means.append(mean_loss)

        return aucs, loss_means
