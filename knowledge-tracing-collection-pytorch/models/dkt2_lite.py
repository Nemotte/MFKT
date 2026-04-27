import os
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, sigmoid
from sklearn import metrics


class DKT2Lite(nn.Module):
    """
    DKT2-Lite: 面向实际部署的 DKT2 简化版

    主要特性：
    - Rasch-like 输入表征（题目 + 难度 + 答题结果）
    - xLSTM-lite 序列编码（带指数门控 + 残差块）
    - IRT 风格知识分解（A_t, K_t, K⁺, K⁻）
    - 双头输出：
        * 下一题答对概率（通过选取对应题目的预测）
        * 当前时刻对所有题目的掌握度向量 KS_t ∈ R^{num_q}

    与现有代码兼容：
    - forward(q, r) -> [batch_size, seq_len, num_q]
    - train_model(...) 签名与 DKT 保持一致
    - DataLoader 仍然输出: q, r, q_next, r_next, mask
    """

    def __init__(
        self,
        num_q: int,
        emb_size: int,
        hidden_size: int,
        dropout_rate: float = 0.2,
        lambda_state: float = 0.1,   # 知识状态辅助监督权重
        lambda_smooth: float = 0.0,  # 时间平滑正则权重（可先设 0）
    ):
        super().__init__()

        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.lambda_state = lambda_state
        self.lambda_smooth = lambda_smooth

        # ---------- 1. Rasch-like 输入表征 ----------
        # 题目 embedding（简单起见，把“题目 = 概念”）
        self.q_embed = nn.Embedding(num_q, emb_size)

        # 回答结果 embedding：0 = 错，1 = 对
        self.r_embed = nn.Embedding(2, emb_size)

        # 每个题目的“难度”标量（Rasch 里的 b_q）
        # shape: [num_q, 1]
        self.q_difficulty = nn.Embedding(num_q, 1)

        # 将难度标量映射到向量空间的方向向量 μ, g_r
        self.mu = nn.Parameter(torch.randn(emb_size))          # 题目变异向量
        self.g_r = nn.Parameter(torch.randn(2, emb_size))      # 回答变异向量（对/错各一个）

        # ---------- 2. xLSTM-lite 序列编码 ----------
        # 指数门控，用于对交互 embedding 做缩放
        self.input_gate = nn.Linear(emb_size, emb_size)

        # LSTM 主干
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_rate)

        # mLSTM-lite：用一个双线性近似 + 残差 + LayerNorm
        self.mlp_m = nn.Linear(hidden_size, hidden_size)
        self.ln_m = nn.LayerNorm(hidden_size)

        # ---------- 3. IRT 知识分解 ----------
        # ability 映射：H -> A_t
        self.ability_proj = nn.Linear(hidden_size, emb_size)

        # ---------- 4. 知识解码（全题目掌握向量） ----------
        # 融合 Q_t, K_t, K⁺, K⁻ 之后降维到 emb_size 再解码
        fusion_in_dim = emb_size * 4
        self.fusion_proj = nn.Linear(fusion_in_dim, emb_size)
        self.fusion_act = nn.ReLU()

        # 使用题目 embedding 作为“概念矩阵”进行解码：
        # KS_t = sigmoid( h_state @ q_embed.weight^T )
        # 不额外增加一个巨大的 Linear(num_q)
        # 只使用现有的 q_embed.weight

        self._init_weights()

    # -----------------------------------------------------
    # 初始化
    # -----------------------------------------------------
    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # -----------------------------------------------------
    # 前向：输入 (q, r) 输出对所有题目的掌握度 [B, T, num_q]
    # -----------------------------------------------------
    def forward(self, q, r):
        """
        Args:
            q: [batch_size, seq_len] 题目索引
            r: [batch_size, seq_len] 答题结果 (0/1)

        Returns:
            ks_all: [batch_size, seq_len, num_q]
                    每个时间步对所有题目的掌握度预测
        """
        device = q.device
        batch_size, seq_len = q.shape

        # ---------- 1. Rasch-like 表征 ----------
        # 题目 embedding
        q_emb = self.q_embed(q.long())       # [B, T, E]

        # 回答 embedding
        r_emb = self.r_embed(r.long())       # [B, T, E]

        # 题目难度标量 -> 向量
        # q_d: [B, T, 1]
        q_d = self.q_difficulty(q.long())
        # difficulty 向量：d_q * μ
        # d_vec: [B, T, E]
        d_vec = q_d * self.mu.view(1, 1, -1)

        # 题目流（暂时只在后面融合使用）
        Q_t = q_emb + d_vec

        # 交互流 S_t = e_q + e_r + d_q * g_r
        # g_r_sel: [B, T, E]
        g_r_sel = self.g_r[r.long()]
        S_t = q_emb + r_emb + q_d * g_r_sel

        # ---------- 2. xLSTM-lite 序列编码 ----------
        # 指数门控
        gate_raw = self.input_gate(S_t)          # [B, T, E]
        gate_exp = torch.exp(gate_raw)           # > 0
        gate = gate_exp / (1.0 + gate_exp)       # 映射到 (0,1)
        S_t_tilde = gate * S_t                   # [B, T, E]

        # LSTM 编码
        H, _ = self.lstm(S_t_tilde)              # [B, T, H]
        H = self.dropout(H)

        # mLSTM-lite：用一个简单的残差 MLP + LayerNorm 模拟矩阵记忆增强
        M = self.mlp_m(H)
        H_m = self.ln_m(H + M)                   # [B, T, H]

        # ---------- 3. IRT 知识分解 ----------
        # 能力向量 A_t
        A_t = self.ability_proj(H_m)             # [B, T, E]

        # K_t = 能力 - 难度向量
        K_t = A_t - d_vec                        # [B, T, E]

        # 熟悉 / 不熟悉知识分解（K⁺ / K⁻）
        r_float = r.unsqueeze(-1).float()        # [B, T, 1]
        K_plus = torch.exp(r_float) * K_t        # 熟悉部分
        K_minus = torch.exp(1.0 - r_float) * K_t # 不熟悉部分

        # ---------- 4. 融合 + 知识解码 ----------
        fusion = torch.cat([Q_t, K_t, K_plus, K_minus], dim=-1)   # [B, T, 4E]
        h_state = self.fusion_act(self.fusion_proj(fusion))       # [B, T, E]

        # 使用题目 embedding 作为“概念矩阵”解码
        # q_embed.weight: [num_q, E]
        # logits: [B, T, num_q]
        logits = torch.matmul(h_state, self.q_embed.weight.t())
        ks_all = sigmoid(logits)  # [B, T, num_q]

        return ks_all

    # -----------------------------------------------------
    # 训练过程
    # -----------------------------------------------------
    def train_model(
        self,
        train_loader,
        test_loader,
        num_epochs,
        optimizer,
        ckpt_path,
        scheduler=None,
        early_stopping_patience: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        与现有 DKT.train_model 接口保持一致
        """
        os.makedirs(ckpt_path, exist_ok=True)
        self.to(device)

        train_losses = []
        test_aucs = []
        best_auc = 0.0
        patience_counter = 0

        print(f"开始训练 DKT2-Lite，设备: {device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            self.train()
            epoch_losses = []

            for batch_idx, batch_data in enumerate(train_loader):
                q, r, q_next, r_next, mask = [x.to(device) for x in batch_data]

                # 前向：得到对所有题目的掌握度 [B, T, num_q]
                ks_all = self(q.long(), r.long())

                # ---------- 主任务：下一题答对预测 ----------
                # 取出 q_next 对应题目的预测概率
                # one_hot: [B, T, num_q]
                oh_next = one_hot(q_next.long(), num_classes=self.num_q).float()
                # [B, T]
                p_next_all = (ks_all * oh_next).sum(dim=-1)

                # 按 mask 选出有效位置
                mask_bool = mask.bool()
                p_next = torch.masked_select(p_next_all, mask_bool)
                y_next = torch.masked_select(r_next, mask_bool)

                loss_next = binary_cross_entropy(p_next, y_next)

                # ---------- 辅助任务：当前题目的知识状态监督 ----------
                # 使用 q, r 监督 KS_t 在当前题目的维度
                oh_curr = one_hot(q.long(), num_classes=self.num_q).float()
                p_curr_all = (ks_all * oh_curr).sum(dim=-1)  # [B, T]

                mask_curr = (q > 0)  # 简单认为 q==0 为 padding（若无 padding 则可全部为 True）
                p_curr = torch.masked_select(p_curr_all, mask_curr)
                y_curr = torch.masked_select(r, mask_curr)

                if p_curr.numel() > 0:
                    loss_state = binary_cross_entropy(p_curr, y_curr)
                else:
                    loss_state = torch.tensor(0.0, device=device)

                # ---------- 时间平滑正则（可选） ----------
                if self.lambda_smooth > 0.0:
                    # 对 ks_all 在时间维度上做平滑：∑||K_t - K_{t-1}||
                    if ks_all.size(1) > 1:
                        delta = ks_all[:, 1:, :] - ks_all[:, :-1, :]  # [B, T-1, num_q]
                        smooth_term = torch.mean(torch.abs(delta))
                    else:
                        smooth_term = torch.tensor(0.0, device=device)
                else:
                    smooth_term = torch.tensor(0.0, device=device)

                # ---------- 总损失 ----------
                loss = loss_next + self.lambda_state * loss_state + self.lambda_smooth * smooth_term

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

                # 也可以在第一个 batch 打印各项损失的量级，方便调参
                if epoch == 1 and batch_idx == 0:
                    print(
                        f"[DEBUG] loss_next={loss_next.item():.4f}, "
                        f"lambda_state*loss_state={(self.lambda_state * loss_state).item():.4f}, "
                        f"lambda_smooth*smooth_term={(self.lambda_smooth * smooth_term).item():.4f}"
                    )

            if scheduler is not None:
                # 如果使用 ReduceLROnPlateau 等 scheduler，需要传入监控指标
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # 先评估再 step
                    test_auc = self._evaluate(test_loader, device)
                    scheduler.step(test_auc)
                else:
                    scheduler.step()

            # 评估（注意：若上面已经评估过，可以复用）
            test_auc = self._evaluate(test_loader, device)
            avg_train_loss = float(np.mean(epoch_losses))

            train_losses.append(avg_train_loss)
            test_aucs.append(test_auc)

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"轮次 {epoch:3d}/{num_epochs} - "
                f"训练损失: {avg_train_loss:.4f}, "
                f"测试AUC: {test_auc:.4f}, "
                f"学习率: {lr}"
            )

            # 保存最佳模型
            if test_auc > best_auc:
                best_auc = test_auc
                patience_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "train_loss": avg_train_loss,
                }
                torch.save(checkpoint, os.path.join(ckpt_path, "best_model.ckpt"))
                print(f"保存最佳模型，AUC: {best_auc:.4f}")
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"早停触发，在第 {epoch} 轮停止训练")
                break

        print(f"训练完成！最佳AUC: {best_auc:.4f}")
        return train_losses, test_aucs

    # -----------------------------------------------------
    # 评估函数：与 DKT 一样，用下一题预测算 AUC
    # -----------------------------------------------------
    def _evaluate(self, test_loader, device: str):
        self.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in test_loader:
                q, r, q_next, r_next, mask = [x.to(device) for x in batch_data]

                ks_all = self(q.long(), r.long())  # [B, T, num_q]

                oh_next = one_hot(q_next.long(), num_classes=self.num_q).float()
                p_next_all = (ks_all * oh_next).sum(dim=-1)  # [B, T]

                mask_bool = mask.bool()
                p_next = torch.masked_select(p_next_all, mask_bool)
                y_next = torch.masked_select(r_next, mask_bool)

                all_predictions.extend(p_next.cpu().numpy())
                all_targets.extend(y_next.cpu().numpy())

        if len(set(all_targets)) > 1:
            auc = metrics.roc_auc_score(all_targets, all_predictions)
        else:
            auc = 0.5

        return auc
