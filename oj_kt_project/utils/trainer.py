"""
模型训练器 — AC预测 + 掌握度 + Q-Matrix正则化
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging

from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss with optional label smoothing"""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()


class Trainer:
    """
    训练器: L = λ_ac * FocalLoss/BCE + λ_mastery * MSE + λ_q * Q_reg
    """
    def __init__(self, model, config, train_loader, val_loader=None, pos_weight=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.DEVICE
        self.model.to(self.device)

        # 损失函数
        label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
        if getattr(config, 'USE_FOCAL_LOSS', False):
            self.ac_criterion = FocalLoss(
                alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA,
                label_smoothing=label_smoothing,
            )
            logger.info(f"使用 Focal Loss (alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA})")
        else:
            if pos_weight is not None:
                pw = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
                self.ac_criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                self.ac_criterion = nn.BCEWithLogitsLoss()

        self.mastery_criterion = nn.MSELoss(reduction='none')

        # 活跃 KP mask
        if hasattr(model, 'q_matrix') and hasattr(model.q_matrix, 'annotated_q'):
            active = (model.q_matrix.annotated_q.max(dim=0).values > 0).float()
            self.active_kp_mask = active.to(self.device)
            logger.info(f"活跃 KP 数: {int(active.sum().item())}/{active.shape[0]}")
        else:
            self.active_kp_mask = None

        self.best_threshold = config.AC_THRESHOLD

        # 多任务权重
        self.ac_weight = config.AC_LOSS_WEIGHT
        self.mastery_weight = config.MASTERY_LOSS_WEIGHT
        self.q_weight = config.QMATRIX_LOSS_WEIGHT

        # Baseline 模型不需要 mastery / Q-matrix loss
        if getattr(model, 'is_baseline', False):
            self.mastery_weight = 0.0
            self.q_weight = 0.0
            logger.info("Baseline 模型: mastery_weight=0, q_weight=0")

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
        )

        # 学习率调度器
        warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 0)
        use_cosine = getattr(config, 'USE_COSINE_ANNEALING', False)
        self.warmup_epochs = warmup_epochs

        if use_cosine and warmup_epochs > 0:
            from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
            warmup_scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: max(0.1, (epoch + 1) / warmup_epochs),
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS - warmup_epochs,
                eta_min=config.LEARNING_RATE * 0.01,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            self.scheduler_type = 'epoch'
            logger.info(f"使用 Warmup({warmup_epochs}ep) + CosineAnnealing 调度器")
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max',
                factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE,
            )
            self.scheduler_type = 'plateau'

        # 最佳模型跟踪
        self.best_val_auc = -1.0
        self.best_val_loss = float('inf')
        self.best_model_state = None

        # Early Stopping
        self.early_stop_patience = config.EARLY_STOP_PATIENCE
        self.early_stop_counter = 0

        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

        # SWA
        self.use_swa = getattr(config, 'USE_SWA', False)
        if self.use_swa:
            from torch.optim.swa_utils import AveragedModel
            self.swa_model = AveragedModel(model, device=self.device)
            self.swa_n = 0
            logger.info("已启用 SWA")
        else:
            self.swa_model = None

    def _compute_loss(self, model_output, batch):
        ac_logits, mastery_pred = model_output
        ac_targets = batch['targets'].unsqueeze(-1)

        ac_loss = self.ac_criterion(ac_logits, ac_targets)

        mastery_targets = batch['mastery_targets']

        # Shape guard: baseline 输出 [B, K]
        if self.mastery_weight > 0 and mastery_pred.shape == mastery_targets.shape:
            loss_per_elem = self.mastery_criterion(mastery_pred, mastery_targets)
            if self.active_kp_mask is not None:
                mask = self.active_kp_mask
                loss_per_elem = loss_per_elem * mask
                mastery_loss = loss_per_elem.sum() / (mask.sum() * mastery_pred.size(0))
            else:
                mastery_loss = loss_per_elem.mean()
        else:
            mastery_loss = torch.tensor(0.0, device=mastery_pred.device)

        q_reg_loss = self.model.get_q_regularization_loss()

        total_loss = (
            self.ac_weight * ac_loss +
            self.mastery_weight * mastery_loss +
            self.q_weight * q_reg_loss
        )
        return total_loss, ac_loss.item(), mastery_loss.item(), q_reg_loss.item()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_ac_loss = 0
        total_mastery_loss = 0
        total_q_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            model_output = self.model(batch)

            loss, ac_l, mast_l, q_l = self._compute_loss(model_output, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.GRAD_CLIP_NORM)
            self.optimizer.step()

            total_loss += loss.item()
            total_ac_loss += ac_l
            total_mastery_loss += mast_l
            total_q_loss += q_l
            num_batches += 1

        n = max(num_batches, 1)
        metrics = {
            'auc': 0.0,
            'balanced_accuracy': 0.0,
            'ac_loss': total_ac_loss / n,
            'mastery_loss': total_mastery_loss / n,
            'q_loss': total_q_loss / n,
        }
        return total_loss / n, metrics

    def validate(self):
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0
        all_ac_logits = []
        all_ac_targets = []
        all_mastery_pred = []
        all_mastery_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                model_output = self.model(batch)
                loss, _, _, _ = self._compute_loss(model_output, batch)

                total_loss += loss.item()
                all_ac_logits.append(model_output[0].cpu())
                all_ac_targets.append(batch['targets'].cpu())
                all_mastery_pred.append(model_output[1].cpu())
                all_mastery_targets.append(batch['mastery_targets'].cpu())

        metrics = MetricsCalculator.compute_all(
            torch.cat(all_ac_logits), torch.cat(all_ac_targets),
            torch.cat(all_mastery_pred), torch.cat(all_mastery_targets),
            threshold=self.best_threshold,
        )
        return total_loss / len(self.val_loader), metrics

    def search_best_threshold(self, dataloader):
        self.model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in dataloader:
                batch = self._to_device(batch)
                model_output = self.model(batch)
                all_logits.append(model_output[0].cpu())
                all_targets.append(batch['targets'].cpu())

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        probs = MetricsCalculator._to_ac_probs(logits)
        tgts = targets.numpy().flatten()

        best_threshold, best_ba = 0.5, 0.0
        from sklearn.metrics import balanced_accuracy_score
        for threshold in np.arange(0.2, 0.8, 0.01):
            preds = (probs >= threshold).astype(float)
            ba = balanced_accuracy_score(tgts, preds)
            if ba > best_ba:
                best_ba = ba
                best_threshold = threshold

        logger.info(f"最优阈值: {best_threshold:.2f}, Balanced Acc={best_ba:.4f}")
        return float(best_threshold)

    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS

        logger.info(f"开始训练 {num_epochs} epochs | 设备: {self.device} | 模型: {self.config.MODEL_TYPE}")
        logger.info(f"训练样本: {len(self.train_loader.dataset)}"
                     + (f" | 验证样本: {len(self.val_loader.dataset)}" if self.val_loader else ""))

        for epoch in range(num_epochs):
            logger.info(f"{'='*50} Epoch {epoch+1}/{num_epochs}")

            train_loss, train_metrics = self.train_epoch()
            logger.info(f"训练 Loss={train_loss:.4f} | AC_Loss={train_metrics['ac_loss']:.4f}")

            if self.val_loader:
                val_loss, val_metrics = self.validate()
                val_auc = val_metrics.get('auc', 0.0)
                logger.info(f"验证 Loss={val_loss:.4f} | AUC={val_auc:.4f} "
                             f"| BA={val_metrics.get('balanced_accuracy',0):.4f}")

                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_auc)
                else:
                    self.scheduler.step()

                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    self.best_val_loss = val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                    self.early_stop_counter = 0
                    logger.info(f">>> 保存最佳模型 (val AUC={val_auc:.4f})")
                else:
                    self.early_stop_counter += 1
                    logger.info(f">>> Early Stop {self.early_stop_counter}/{self.early_stop_patience} "
                                f"(best={self.best_val_auc:.4f})")
                    if self.early_stop_counter >= self.early_stop_patience:
                        logger.info("提前停止！")
                        break
            else:
                if self.scheduler_type == 'epoch':
                    self.scheduler.step()
                self.save_checkpoint(epoch, train_loss, is_best=False)

            # SWA
            swa_start = int(num_epochs * getattr(self.config, 'SWA_START_FRAC', 0.5))
            if self.swa_model is not None and epoch >= swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_n += 1

        # SWA 评估
        if self.swa_model is not None and self.swa_n > 0 and self.val_loader:
            self._evaluate_swa()

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            if getattr(self.config, 'SEARCH_BEST_THRESHOLD', False) and self.val_loader:
                self.best_threshold = self.search_best_threshold(self.val_loader)

    def _evaluate_swa(self):
        logger.info(f"评估 SWA 模型（平均了 {self.swa_n} 个 epoch）...")
        self.swa_model.eval()
        swa_logits, swa_targets = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                model_output = self.swa_model(batch)
                swa_logits.append(model_output[0].cpu())
                swa_targets.append(batch['targets'].cpu())

        swa_auc = MetricsCalculator.auc(torch.cat(swa_logits), torch.cat(swa_targets))
        logger.info(f"SWA AUC={swa_auc:.4f} | best AUC={self.best_val_auc:.4f}")

        if not np.isnan(swa_auc) and swa_auc > self.best_val_auc:
            logger.info(">>> 采用 SWA 模型")
            self.best_model_state = copy.deepcopy(self.swa_model.module.state_dict())
            self.best_val_auc = swa_auc
        else:
            logger.info(">>> 保留 best 单点模型")

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_val_auc': self.best_val_auc,
        }
        path = os.path.join(self.config.MODEL_SAVE_PATH, 'latest_model.pth')
        torch.save(checkpoint, path)
        if is_best:
            path = os.path.join(self.config.MODEL_SAVE_PATH, 'best_model.pth')
            torch.save(checkpoint, path)

    def _to_device(self, batch):
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
