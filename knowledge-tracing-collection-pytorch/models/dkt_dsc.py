import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, MultiheadAttention
from sklearn import metrics

class DKT_CLE(Module):
    '''
    DKT-CLE: Deep Knowledge Tracing with Cognitive Load Estimation
    A dual-stream neural network architecture for personalized learning path generation
    
    Args:
        num_q: the total number of questions(KCs) in the dataset
        emb_size: dimension of embedding vectors
        hidden_size: dimension of hidden vectors
        cognitive_size: dimension of cognitive load estimation vectors
        lambda_kt: weight for knowledge tracing loss
        lambda_cle: weight for cognitive load estimation loss
        lambda_balance: weight for balancing knowledge acquisition and cognitive load
        dropout_rate: dropout rate for regularization
    '''
    def __init__(
        self, 
        num_q, 
        emb_size=128, 
        hidden_size=128,
        cognitive_size=64,
        lambda_kt=1.0,
        lambda_cle=0.5,
        lambda_balance=0.3,
        dropout_rate=0.2
    ):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.cognitive_size = cognitive_size
        self.lambda_kt = lambda_kt
        self.lambda_cle = lambda_cle
        self.lambda_balance = lambda_balance
        
        # Interaction embedding for knowledge tracing
        self.interaction_emb = Embedding(num_q * 2, emb_size)
        
        # Question embedding for cognitive load estimation
        self.question_emb = Embedding(num_q, emb_size)
        
        # Knowledge Tracing Stream
        self.kt_lstm = LSTM(emb_size, hidden_size, batch_first=True)
        self.kt_output = Linear(hidden_size, num_q)
        
        # Cognitive Load Estimation Stream
        self.cle_lstm = LSTM(emb_size, cognitive_size, batch_first=True)
        self.cle_attention = MultiheadAttention(cognitive_size, num_heads=4, batch_first=True)
        self.cle_output = Linear(cognitive_size, 1)  # Single cognitive load score
        
        # Feature fusion layers
        self.fusion_layer = Linear(hidden_size + cognitive_size, hidden_size)
        self.final_kt_output = Linear(hidden_size, num_q)
        
        # Dropout layers
        self.dropout = Dropout(dropout_rate)
        
        # Cognitive load difficulty estimation
        self.difficulty_layer = Linear(emb_size, 1)
        
    def forward(self, q, r, return_cognitive_load=False):
        '''
        Forward pass for DKT-CLE model
        
        Args:
            q: question sequence [batch_size, seq_len]
            r: response sequence [batch_size, seq_len]
            return_cognitive_load: whether to return cognitive load estimates
            
        Returns:
            kt_pred: knowledge tracing predictions
            cle_pred: cognitive load estimates (if requested)
        '''
        batch_size, seq_len = q.size()
        
        # Knowledge Tracing Stream
        # Create interaction embeddings (question + response)
        x_kt = q + self.num_q * r
        kt_emb = self.interaction_emb(x_kt)
        kt_hidden, _ = self.kt_lstm(kt_emb)
        kt_output = self.kt_output(kt_hidden)
        kt_output = self.dropout(kt_output)
        
        # Cognitive Load Estimation Stream
        # Use question embeddings for cognitive load estimation
        q_emb = self.question_emb(q)
        cle_hidden, _ = self.cle_lstm(q_emb)
        
        # Apply attention mechanism for cognitive load
        cle_attended, _ = self.cle_attention(cle_hidden, cle_hidden, cle_hidden)
        cle_output = self.cle_output(cle_attended).squeeze(-1)  # [batch_size, seq_len]
        
        # Estimate question difficulty
        difficulty = self.difficulty_layer(q_emb).squeeze(-1)  # [batch_size, seq_len]
        
        # Fusion of knowledge tracing and cognitive load information
        fused_features = torch.cat([kt_hidden, cle_attended], dim=-1)
        fused_output = self.fusion_layer(fused_features)
        fused_output = torch.relu(fused_output)
        fused_output = self.dropout(fused_output)
        
        # Final knowledge prediction with cognitive load consideration
        final_kt_pred = self.final_kt_output(fused_output)
        final_kt_pred = torch.sigmoid(final_kt_pred)
        
        # Cognitive load prediction (normalized)
        cle_pred = torch.sigmoid(cle_output)
        
        if return_cognitive_load:
            return final_kt_pred, cle_pred, difficulty
        else:
            return final_kt_pred
    
    def compute_loss(self, q, r, qshft, rshft, m):
        '''
        Compute multi-objective loss function
        
        Args:
            q, r: current question and response sequences
            qshft, rshft: shifted question and response sequences
            m: mask for valid positions
        '''
        # Forward pass
        kt_pred, cle_pred, difficulty = self.forward(q, r, return_cognitive_load=True)
        
        # Knowledge Tracing Loss
        kt_next = (kt_pred * F.one_hot(qshft.long(), self.num_q)).sum(-1)
        kt_next = torch.masked_select(kt_next, m)
        rshft_masked = torch.masked_select(rshft, m)
        
        # Convert to float for binary_cross_entropy
        kt_next = kt_next.float()
        rshft_masked = rshft_masked.float()
        
        loss_kt = F.binary_cross_entropy(kt_next, rshft_masked)
        
        # Cognitive Load Estimation Loss
        # Estimate expected cognitive load based on question difficulty and student ability
        student_ability = kt_pred.mean(dim=-1)  # Average knowledge level as ability proxy
        expected_cognitive_load = torch.sigmoid(difficulty - student_ability)
        
        cle_masked = torch.masked_select(cle_pred, m)
        expected_cl_masked = torch.masked_select(expected_cognitive_load, m)
        
        loss_cle = F.mse_loss(cle_masked, expected_cl_masked)
        
        # Balance regularization (encourage optimal cognitive load)
        # Cognitive load should be moderate (not too high, not too low)
        optimal_load = 0.6  # Optimal cognitive load level
        balance_penalty = torch.mean((cle_pred - optimal_load) ** 2)
        
        # Knowledge consistency regularization
        if kt_pred.size(1) > 1:  # Check if sequence length > 1
            knowledge_consistency = torch.mean(torch.norm(kt_pred[:, 1:] - kt_pred[:, :-1], p=2, dim=-1) ** 2)
        else:
            knowledge_consistency = torch.tensor(0.0, device=kt_pred.device)
        
        # Total loss
        total_loss = (self.lambda_kt * loss_kt + 
                     self.lambda_cle * loss_cle + 
                     self.lambda_balance * balance_penalty +
                     0.1 * knowledge_consistency)
        
        return total_loss, loss_kt, loss_cle, balance_penalty
    
    def train_model(self, train_loader, test_loader, num_epochs, optimizer, ckpt_path):
        '''
        Training loop for DKT-CLE model
        '''
        aucs = []
        loss_means = []
        cognitive_load_scores = []
        
        max_auc = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_losses = []
            epoch_kt_losses = []
            epoch_cle_losses = []
            
            # Training phase
            self.train()
            for batch_idx, data in enumerate(train_loader):
                q, r, qshft, rshft, m = data
                
                # Ensure all data is on the same device and correct dtype
                device = next(self.parameters()).device
                q = q.to(device).long()
                r = r.to(device).long()
                qshft = qshft.to(device).long()
                rshft = rshft.to(device).float()  # Convert to float for loss calculation
                m = m.to(device).bool()
                
                optimizer.zero_grad()
                
                try:
                    total_loss, loss_kt, loss_cle, balance_penalty = self.compute_loss(
                        q, r, qshft, rshft, m
                    )
                    
                    total_loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_losses.append(total_loss.item())
                    epoch_kt_losses.append(loss_kt.item())
                    epoch_cle_losses.append(loss_cle.item())
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    print(f"Shapes - q: {q.shape}, r: {r.shape}, qshft: {qshft.shape}, rshft: {rshft.shape}, m: {m.shape}")
                    continue
            
            # Evaluation phase
            self.eval()
            with torch.no_grad():
                test_aucs = []
                test_cognitive_loads = []
                
                for data in test_loader:
                    q, r, qshft, rshft, m = data
                    
                    # Ensure all data is on the same device and correct dtype
                    q = q.to(device).long()
                    r = r.to(device).long()
                    qshft = qshft.to(device).long()
                    rshft = rshft.to(device).float()
                    m = m.to(device).bool()
                    
                    try:
                        kt_pred, cle_pred, _ = self.forward(q, r, return_cognitive_load=True)
                        kt_next = (kt_pred * F.one_hot(qshft.long(), self.num_q)).sum(-1)
                        
                        # Extract valid predictions
                        kt_next_masked = torch.masked_select(kt_next, m).cpu().numpy()
                        rshft_masked = torch.masked_select(rshft, m).cpu().numpy()
                        cle_masked = torch.masked_select(cle_pred, m).cpu().numpy()
                        
                        # Calculate AUC
                        if len(np.unique(rshft_masked)) > 1 and len(kt_next_masked) > 0:
                            auc = metrics.roc_auc_score(rshft_masked, kt_next_masked)
                            test_aucs.append(auc)
                        
                        if len(cle_masked) > 0:
                            test_cognitive_loads.extend(cle_masked.tolist())
                            
                    except Exception as e:
                        print(f"Error in evaluation: {e}")
                        continue
                
                # Aggregate metrics
                avg_auc = np.mean(test_aucs) if test_aucs else 0
                avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
                avg_kt_loss = np.mean(epoch_kt_losses) if epoch_kt_losses else 0
                avg_cle_loss = np.mean(epoch_cle_losses) if epoch_cle_losses else 0
                avg_cognitive_load = np.mean(test_cognitive_loads) if test_cognitive_loads else 0
                
                print(f"Epoch {epoch:3d}: AUC={avg_auc:.4f}, "
                      f"Total Loss={avg_loss:.4f}, KT Loss={avg_kt_loss:.4f}, "
                      f"CLE Loss={avg_cle_loss:.4f}, Avg CL={avg_cognitive_load:.4f}")
                
                # Save best model
                if avg_auc > max_auc and avg_auc > 0:
                    max_auc = avg_auc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'auc': avg_auc,
                        'loss': avg_loss,
                    }, os.path.join(ckpt_path, 'dkt_cle_best_model.ckpt'))
                
                aucs.append(avg_auc)
                loss_means.append(avg_loss)
                cognitive_load_scores.append(avg_cognitive_load)
        
        return aucs, loss_means, cognitive_load_scores
    
    def generate_learning_path(self, student_history, available_questions, path_length=10):
        '''
        Generate personalized learning path based on knowledge state and cognitive load
        
        Args:
            student_history: tuple of (q_seq, r_seq) - student's learning history
            available_questions: list of available question IDs
            path_length: desired length of learning path
            
        Returns:
            learning_path: list of recommended question IDs
        '''
        self.eval()
        q_seq, r_seq = student_history
        
        with torch.no_grad():
            # Convert to tensors
            device = next(self.parameters()).device
            q_tensor = torch.LongTensor(q_seq).unsqueeze(0).to(device)
            r_tensor = torch.LongTensor(r_seq).unsqueeze(0).to(device)
            
            # Get current knowledge state and cognitive load capacity
            kt_pred, cle_pred, _ = self.forward(q_tensor, r_tensor, return_cognitive_load=True)
            
            current_knowledge = kt_pred[0, -1].cpu().numpy()  # Latest knowledge state
            current_cognitive_capacity = 1.0 - cle_pred[0, -1].item()  # Remaining cognitive capacity
            
            learning_path = []
            
            for _ in range(path_length):
                best_question = None
                best_score = -float('inf')
                
                for q_id in available_questions:
                    if q_id in learning_path:
                        continue
                    
                    # Ensure q_id is within valid range
                    if q_id >= self.num_q:
                        continue
                    
                    # Estimate learning gain and cognitive cost for this question
                    knowledge_deficit = max(0, 0.8 - current_knowledge[q_id])  # Target 80% mastery
                    
                    # Estimate question difficulty (simplified)
                    question_difficulty = np.random.beta(2, 5)  # Placeholder - should use learned difficulty
                    cognitive_cost = min(question_difficulty, current_cognitive_capacity)
                    
                    # Multi-objective score: balance learning gain and cognitive efficiency
                    if cognitive_cost > 0:
                        efficiency_score = knowledge_deficit / cognitive_cost
                        score = efficiency_score - 0.1 * abs(cognitive_cost - 0.6)  # Prefer moderate cognitive load
                        
                        if score > best_score:
                            best_score = score
                            best_question = q_id
                
                if best_question is not None:
                    learning_path.append(best_question)
                    # Update cognitive capacity (simplified)
                    current_cognitive_capacity = max(0.1, current_cognitive_capacity - 0.1)
                else:
                    break
            
            return learning_path

# Example usage
def create_dkt_cle_model(num_questions):
    '''
    Create a DKT-CLE model instance with recommended hyperparameters
    '''
    model = DKT_CLE(
        num_q=num_questions,
        emb_size=128,
        hidden_size=128,
        cognitive_size=64,
        lambda_kt=1.0,
        lambda_cle=0.5,
        lambda_balance=0.3,
        dropout_rate=0.2
    )
    
    return model

# Training example
def train_dkt_cle(model, train_loader, test_loader, num_epochs=100, lr=0.001):
    '''
    Train the DKT-CLE model
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    ckpt_path = "./checkpoints"
    os.makedirs(ckpt_path, exist_ok=True)
    
    aucs, losses, cognitive_loads = model.train_model(
        train_loader, test_loader, num_epochs, optimizer, ckpt_path
    )
    
    return aucs, losses, cognitive_loads
