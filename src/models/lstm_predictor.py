"""
LSTM Failure Probability Predictor
====================================
Binary classifier: predicts probability of failure within h cycles.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

import config


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification (Lin et al., 2017).

    L = - alpha * (1 - p_t)^gamma * log(p_t)

    Down-weights well-classified examples so training focuses on the
    hard, misclassified ones. Particularly effective on imbalanced
    failure-prediction data, where standard BCE with class weighting
    can over-correct toward recall and trade away precision.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()


class LSTMPredictor(nn.Module):
    """
    LSTM-based binary classifier for failure prediction.

    Architecture:
    - 2-layer LSTM with attention mechanism
    - Dense layers with dropout → Sigmoid output

    Output: P(failure within h cycles)
    """

    def __init__(self, input_dim, hidden_dim=None, num_layers=None, dropout=None):
        super().__init__()

        self.hidden_dim = hidden_dim or config.PRED_HIDDEN_DIM
        self.num_layers = num_layers or config.PRED_NUM_LAYERS
        self.dropout = dropout or config.PRED_DROPOUT

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Classification head (outputs raw logits — sigmoid applied by BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self.input_dim = input_dim

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_dim)

        # Classification (raw logits)
        logits = self.classifier(context)  # (batch, 1)
        return logits.squeeze(-1), attn_weights.squeeze(-1)

    def predict_proba(self, x):
        """Get failure probability for input sequences."""
        self.eval()
        self.to(config.DEVICE)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        batch_size = config.PRED_BATCH_SIZE
        all_proba = []
        all_attn = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch_x = x[start:start + batch_size].to(config.DEVICE)
                logits, attn = self.forward(batch_x)
                proba = torch.sigmoid(logits)
                all_proba.append(proba.cpu().numpy())
                all_attn.append(attn.cpu().numpy())

        if not all_proba:
            return np.array([]), np.array([])

        return np.concatenate(all_proba, axis=0), np.concatenate(all_attn, axis=0)


class PredictorTrainer:
    """Training loop for LSTM failure predictor with class balancing."""

    def __init__(self, model, lr=None, epochs=None, batch_size=None, early_stopping_patience=10):
        self.model = model.to(config.DEVICE)
        self.lr = lr or config.PRED_LEARNING_RATE
        self.epochs = epochs or config.PRED_EPOCHS
        self.batch_size = batch_size or config.PRED_BATCH_SIZE
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, mode="max"
        )
        self.train_history = []
        self.val_history = []
        self.best_epoch = None
        self.stopped_epoch = None
        self.early_stopping_patience = early_stopping_patience

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the failure predictor.

        Parameters
        ----------
        X_train, y_train : np.ndarray — training sequences and binary labels
        X_val, y_val : np.ndarray, optional — validation data
        """
        # Compute class weights for imbalanced data
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count

        if pos_count == 0:
            print("[PREDICTOR] WARNING: Training data has ZERO positive (failure) samples!")
            print("[PREDICTOR] The model cannot learn failure patterns. Check your data split.")
        elif pos_count < 10:
            print(f"[PREDICTOR] WARNING: Only {int(pos_count)} positive samples — "
                  "model may not generalize well.")

        # Loss function: focal loss (default) tends to give a better
        # precision/recall trade-off on imbalanced failure data than BCE
        # with positive class weighting.
        if config.PRED_LOSS == "focal":
            criterion = FocalLoss(
                alpha=config.PRED_FOCAL_ALPHA,
                gamma=config.PRED_FOCAL_GAMMA,
            )
            pos_weight = torch.tensor([1.0]).to(config.DEVICE)  # for logging only
        else:
            if pos_count > 0:
                computed_weight = min(neg_count / pos_count, 20.0)
            else:
                computed_weight = 20.0
            pos_weight = torch.tensor([computed_weight]).to(config.DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_tensor_x = torch.FloatTensor(X_train)
        train_tensor_y = torch.FloatTensor(y_train)
        train_loader = DataLoader(
            TensorDataset(train_tensor_x, train_tensor_y),
            batch_size=self.batch_size, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
            persistent_workers=config.NUM_WORKERS > 0,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                batch_size=self.batch_size, shuffle=False,
                num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
                persistent_workers=config.NUM_WORKERS > 0,
            )

        best_val_f1 = 0
        best_state = None
        epochs_since_best_f1 = 0

        # Automatic Mixed Precision — ~1.5-2x faster on CUDA, no-op on CPU
        use_amp = torch.cuda.is_available() and str(config.DEVICE) != "cpu"
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        _orig_model = self.model

        print(f"\n[PREDICTOR] Training on {config.DEVICE} "
              f"({len(X_train)} samples, pos_rate={pos_count/len(y_train):.2%}, "
              f"batch_size={self.batch_size}, AMP={use_amp})")
        if config.PRED_LOSS == "focal":
            print(f"[PREDICTOR] Loss: focal (alpha={config.PRED_FOCAL_ALPHA}, "
                  f"gamma={config.PRED_FOCAL_GAMMA})")
        else:
            print(f"[PREDICTOR] Loss: BCE (pos_weight={pos_weight.item():.2f})")

        n_batches = len(train_loader)
        epoch_bar = tqdm(range(self.epochs), desc="[PRED] Epochs", unit="epoch")
        for epoch in epoch_bar:
            # Training
            self.model.train()
            train_loss = 0
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader, 1):
                batch_x = batch_x.to(config.DEVICE, non_blocking=config.PIN_MEMORY)
                batch_y = batch_y.to(config.DEVICE, non_blocking=config.PIN_MEMORY)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits, _ = self.model(batch_x)
                    loss = criterion(logits, batch_y)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                train_loss += loss.item() * len(batch_x)
                epoch_bar.set_postfix(batch=f"{batch_idx}/{n_batches}",
                                      loss=f"{loss.item():.4f}")

            train_loss /= len(X_train)
            self.train_history.append(train_loss)

            # Validation every 2 epochs — 100 checkpoints over 200 epochs (vs 5 originally)
            postfix = {"train_loss": f"{train_loss:.4f}"}
            if val_loader is not None and (epoch + 1) % 2 == 0:
                metrics = self._evaluate(val_loader, y_val)
                self.val_history.append(metrics)
                self.scheduler.step(metrics["auc"])

                if metrics["f1"] > best_val_f1:
                    best_val_f1 = metrics["f1"]
                    best_state = {k: v.cpu().clone() for k, v in _orig_model.state_dict().items()}
                    self.best_epoch = epoch + 1
                    epochs_since_best_f1 = 0
                else:
                    epochs_since_best_f1 += 1

                postfix.update({"F1": f"{metrics['f1']:.4f}",
                                "AUC": f"{metrics['auc']:.4f}"})

                if self.early_stopping_patience is not None and epochs_since_best_f1 >= self.early_stopping_patience:
                    self.stopped_epoch = epoch + 1
                    print(f"[PREDICTOR] Early stopping after {epoch + 1} epochs; no F1 improvement for {epochs_since_best_f1} epochs.")
                    break

            epoch_bar.set_postfix(postfix)

        # Final epoch after training or early stopping
        if self.stopped_epoch is None:
            self.stopped_epoch = epoch + 1

        # Restore best model into original (uncompiled) model
        if best_state is not None:
            _orig_model.load_state_dict(best_state)
            _orig_model.to(config.DEVICE)
            self.model = _orig_model  # return uncompiled model for safe save/inference
            print(f"[PREDICTOR] Restored best model (F1={best_val_f1:.4f})")
        else:
            self.model = _orig_model
            print(f"[PREDICTOR] No best model found, using last model")

        return self.model

    def _evaluate(self, loader, y_true):
        """
        Evaluate model on validation/test set.

        Uses precision-recall curve to find the threshold that maximises F1,
        rather than a fixed 0.5 (which is suboptimal when pos_weight has shifted
        the decision boundary for imbalanced classes).
        """
        self.model.eval()
        all_proba = []
        use_amp = torch.cuda.is_available() and str(config.DEVICE) != "cpu"
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(config.DEVICE)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits, _ = self.model(batch_x)
                proba = torch.sigmoid(logits)
                all_proba.extend(proba.cpu().numpy())

        y_proba = np.array(all_proba)

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_proba)
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
            denom = precisions + recalls
            f1_scores = np.zeros_like(denom)
            np.divide(2 * precisions * recalls, denom,
                      out=f1_scores, where=denom > 0)
            # f1_scores has one more element than thresholds (the all-positive
            # / all-negative endpoints); restrict argmax to the matching range.
            best_idx = int(np.argmax(f1_scores[:-1]))
            optimal_threshold = float(thresholds[best_idx])
        else:
            auc = 0.0
            optimal_threshold = 0.5

        y_pred = (y_proba >= optimal_threshold).astype(int)

        return {
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "auc": auc,
            "optimal_threshold": optimal_threshold,
        }

    def plot_training_curves(self, save_path=None):
        """Plot and save training loss and validation F1 over epochs."""
        if not self.val_history:
            print("[PREDICTOR] No validation history to plot.")
            return

        epochs = list(range(2, len(self.val_history) * 2 + 1, 2))  # Every 2 epochs
        f1_scores = [m['f1'] for m in self.val_history]
        auc_scores = [m['auc'] for m in self.val_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Training loss
        ax1.plot(self.train_history, label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)

        # Validation F1 and AUC
        ax2.plot(epochs, f1_scores, label='F1 Score', marker='o')
        ax2.plot(epochs, auc_scores, label='AUC', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Validation Metrics')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(config.MODELS_DIR, 'predictor_training_curves.png')
        plt.savefig(save_path, dpi=150)
        print(f"[PREDICTOR] Training curves saved to {save_path}")

        # Print optimal F1
        best_f1 = max(f1_scores)
        best_epoch = epochs[f1_scores.index(best_f1)]
        print(f"[PREDICTOR] Best F1: {best_f1:.4f} at epoch {best_epoch}")

        # Display the plot inline in the notebook
        plt.show()

    def save_model(self, filepath=None):
        """Save trained model."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "lstm_predictor.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }, filepath)
        print(f"[PREDICTOR] Model saved to {filepath}")
        self.plot_training_curves()


def load_predictor(filepath=None):
    """Load a trained predictor."""
    filepath = filepath or os.path.join(config.MODELS_DIR, "lstm_predictor.pt")
    checkpoint = torch.load(filepath, map_location=config.DEVICE, weights_only=False)

    model = LSTMPredictor(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(config.DEVICE)
    model.eval()

    print(f"[PREDICTOR] Loaded model from {filepath}")
    return model
