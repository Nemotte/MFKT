import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy, sigmoid
from sklearn import metrics
from typing import Tuple, Optional

class DKT(Module):
    """
    深度知识追踪模型 (Deep Knowledge Tracing)
    基于论文: "Deep Knowledge Tracing" (Piech et al., 2015)
    
    该模型使用LSTM来建模学生在一系列练习中的知识状态演化，
    通过学生的历史答题记录预测其在新问题上的表现。
    
    Args:
        num_q: 知识概念(Knowledge Components)的总数
        emb_size: 嵌入向量的维度
        hidden_size: LSTM隐藏层的维度
        dropout_rate: dropout比率，用于防止过拟合
    """
    
    def __init__(self, num_q, emb_size, hidden_size, dropout_rate=0.2):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # 交互嵌入层：将(题目ID, 答题结果)对映射到嵌入空间
        # 论文中提到：x_t = (q_t, a_t)，其中q_t是题目，a_t是答案
        # 正确答案：q_id，错误答案：q_id + num_q
        self.interaction_emb = Embedding(
            num_embeddings=num_q * 2,  # 每个题目有两种状态：正确/错误
            embedding_dim=emb_size,
            padding_idx=0
        )
        
        # LSTM层：建模学生知识状态的时序演化
        # 论文核心：h_t = LSTM(x_t, h_{t-1})
        self.lstm_layer = LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # 输出层：从隐藏状态预测所有知识概念的掌握概率
        # 论文中：y_t = sigmoid(W_out * h_t + b_out)
        self.out_layer = Linear(hidden_size, num_q)
        
        # Dropout层用于正则化
        self.dropout_layer = Dropout(dropout_rate)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # LSTM输入权重使用Xavier初始化
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # LSTM隐藏权重使用正交初始化
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # 偏置初始化为0，但遗忘门偏置设为1（LSTM最佳实践）
                param.data.fill_(0)
                if 'bias_ih' in name:
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)
            elif 'weight' in name and 'embedding' not in name:
                torch.nn.init.xavier_uniform_(param.data)

    def forward(self, q, r):
        """
        前向传播
        
        Args:
            q: 题目序列 [batch_size, seq_len]
            r: 答题结果序列 [batch_size, seq_len] (1表示正确，0表示错误)
            
        Returns:
            y: 对所有知识概念的掌握概率预测 [batch_size, seq_len, num_q]
        """
        batch_size, seq_len = q.shape
        
        # 构建交互表示：将题目ID和答题结果编码为单一索引
        # 论文思想：将(question, response)对作为原子单位进行建模
        # 正确答案：x = q，错误答案：x = q + num_q
        interaction_indices = q + self.num_q * r
        
        # 获取交互嵌入
        # 论文中的x_t向量表示
        interaction_emb = self.interaction_emb(interaction_indices)  # [batch_size, seq_len, emb_size]
        
        # LSTM前向传播：建模知识状态的时序演化
        # 论文核心：使用RNN/LSTM捕获学生知识状态的动态变化
        lstm_out, (final_hidden, final_cell) = self.lstm_layer(interaction_emb)
        # lstm_out: [batch_size, seq_len, hidden_size]
        
        # 应用dropout进行正则化
        lstm_out = self.dropout_layer(lstm_out)
        
        # 输出层：预测每个知识概念的掌握概率
        # 论文中：从隐藏状态h_t预测所有技能的掌握情况
        logits = self.out_layer(lstm_out)  # [batch_size, seq_len, num_q]
        
        # 使用sigmoid激活函数得到概率
        # 论文中明确使用sigmoid作为最终激活函数
        predictions = torch.sigmoid(logits)
        
        return predictions

    def compute_loss(self, predictions, q_next, r_next, mask):
        """
        计算损失函数
        
        Args:
            predictions: 模型预测 [batch_size, seq_len, num_q]
            q_next: 下一时刻的题目 [batch_size, seq_len]
            r_next: 下一时刻的答题结果 [batch_size, seq_len]  
            mask: 有效位置掩码 [batch_size, seq_len]
            
        Returns:
            loss: 二元交叉熵损失
        """
        # 从预测中选择对应题目的概率
        # 论文中：只对学生实际回答的题目计算损失
        one_hot_q = one_hot(q_next.long(), self.num_q).float()
        selected_predictions = (predictions * one_hot_q).sum(dim=-1)  # [batch_size, seq_len]
        
        # 应用掩码，只计算有效位置的损失
        masked_predictions = torch.masked_select(selected_predictions, mask)
        masked_targets = torch.masked_select(r_next, mask)
        
        # 计算二元交叉熵损失
        # 论文使用的标准损失函数
        loss = binary_cross_entropy(masked_predictions, masked_targets)
        
        return loss

    def train_model(self, train_loader, test_loader, num_epochs, optimizer, 
                   ckpt_path, scheduler=None, early_stopping_patience=10, 
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            num_epochs: 训练轮数
            optimizer: 优化器
            ckpt_path: 模型保存路径
            scheduler: 学习率调度器
            early_stopping_patience: 早停耐心值
            device: 训练设备
        """
        # 确保检查点目录存在
        os.makedirs(ckpt_path, exist_ok=True)
        
        # 将模型移至指定设备
        self.to(device)
        
        # 训练历史记录
        train_losses = []
        test_aucs = []
        best_auc = 0.0
        patience_counter = 0
        
        print(f"开始训练，设备: {device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            # 训练阶段
            self.train()
            epoch_losses = []
            
            for batch_idx, batch_data in enumerate(train_loader):
                # 数据移至设备
                q, r, q_next, r_next, mask = [data.to(device) for data in batch_data]
                
                # 前向传播
                predictions = self(q.long(), r.long())
                
                # 计算损失（基于论文的损失函数设计）
                loss = self.compute_loss(predictions, q_next, r_next, mask)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_losses.append(loss.item())
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step()
            
            # 评估阶段
            test_auc = self._evaluate(test_loader, device)
            avg_train_loss = np.mean(epoch_losses)
            
            # 记录历史
            train_losses.append(avg_train_loss)
            test_aucs.append(test_auc)
            
            # 打印训练信息
            lr = optimizer.param_groups[0]['lr'] if optimizer else 'N/A'
            print(f"轮次 {epoch:3d}/{num_epochs} - "
                  f"训练损失: {avg_train_loss:.4f}, "
                  f"测试AUC: {test_auc:.4f}, "
                  f"学习率: {lr}")
            
            # 保存最佳模型
            if test_auc > best_auc:
                best_auc = test_auc
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                    'train_loss': avg_train_loss
                }
                torch.save(checkpoint, os.path.join(ckpt_path, "best_model.ckpt"))
                print(f"保存最佳模型，AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= early_stopping_patience:
                print(f"早停触发，在第 {epoch} 轮停止训练")
                break
        
        print(f"训练完成！最佳AUC: {best_auc:.4f}")
        return train_losses, test_aucs
    
    def _evaluate(self, test_loader, device):
        """评估模型性能"""
        self.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                q, r, q_next, r_next, mask = [data.to(device) for data in batch_data]
                
                # 前向传播
                predictions = self(q.long(), r.long())
                
                # 选择对应题目的预测概率
                one_hot_q = one_hot(q_next.long(), self.num_q).float()
                selected_predictions = (predictions * one_hot_q).sum(dim=-1)
                
                # 应用掩码
                masked_predictions = torch.masked_select(selected_predictions, mask)
                masked_targets = torch.masked_select(r_next, mask)
                
                all_predictions.extend(masked_predictions.cpu().numpy())
                all_targets.extend(masked_targets.cpu().numpy())
        
        # 计算AUC
        if len(set(all_targets)) > 1:
            auc = metrics.roc_auc_score(all_targets, all_predictions)
        else:
            auc = 0.5  # 只有一个类别时的随机性能
        
        return auc
    
    def predict_next_performance(self, q_sequence, r_sequence, next_questions):
        """
        预测学生在新题目上的表现
        
        Args:
            q_sequence: 历史题目序列
            r_sequence: 历史答题结果序列  
            next_questions: 待预测的题目
            
        Returns:
            predictions: 预测概率
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # 确保输入为tensor并移至正确设备
            if not isinstance(q_sequence, torch.Tensor):
                q_sequence = torch.tensor(q_sequence).unsqueeze(0)
            if not isinstance(r_sequence, torch.Tensor):
                r_sequence = torch.tensor(r_sequence).unsqueeze(0)
            if not isinstance(next_questions, torch.Tensor):
                next_questions = torch.tensor(next_questions)
            
            q_sequence = q_sequence.to(device)
            r_sequence = r_sequence.to(device)
            next_questions = next_questions.to(device)
            
            # 获取学生当前的知识状态
            knowledge_state = self(q_sequence, r_sequence)  # [1, seq_len, num_q]
            
            # 使用最后一个时间步的知识状态进行预测
            final_state = knowledge_state[:, -1, :]  # [1, num_q]
            
            # 预测指定题目的表现
            predictions = final_state[0, next_questions]  # [len(next_questions)]
            
        return predictions.cpu().numpy()


# 辅助函数：创建优化器和调度器
def create_optimizer_and_scheduler(model, learning_rate=0.001, weight_decay=1e-5):
    """创建Adam优化器和学习率调度器"""
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # 使用ReduceLROnPlateau调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # 监控AUC最大化
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    return optimizer, scheduler


# 使用示例
def example_usage():
    """DKT模型使用示例"""
    # 模型配置
    num_questions = 124  # 假设有124个知识概念
    emb_size = 128
    hidden_size = 128
    
    # 创建模型
    model = DKT(
        num_q=num_questions,
        emb_size=emb_size, 
        hidden_size=hidden_size,
        dropout_rate=0.2
    )
    
    # 创建优化器
    optimizer, scheduler = create_optimizer_and_scheduler(model)
    
    print("DKT模型创建成功！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, optimizer, scheduler


if __name__ == "__main__":
    # 创建模型示例
    model, optimizer, scheduler = example_usage()
