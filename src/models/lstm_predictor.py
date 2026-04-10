"""
LSTM Failure Probability Predictor
====================================
Binary classifier: predicts probability of failure within h cycles.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from tqdm import tqdm

import config


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

    def __init__(self, model, lr=None, epochs=None, batch_size=None):
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

        # Dynamic pos_weight based on actual class balance (capped at 20.0)
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
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                batch_size=self.batch_size, shuffle=False,
                num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
            )

        best_val_auc = 0
        best_state = None

        print(f"\n[PREDICTOR] Training on {config.DEVICE} "
              f"({len(X_train)} samples, pos_rate={pos_count/len(y_train):.2%}, "
              f"batch_size={self.batch_size})")
        print(f"[PREDICTOR] Positive weight: {pos_weight.item():.2f}")

        n_batches = len(train_loader)
        epoch_bar = tqdm(range(self.epochs), desc="[PRED] Epochs", unit="epoch")
        for epoch in epoch_bar:
            # Training
            self.model.train()
            train_loss = 0
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader, 1):
                batch_x = batch_x.to(config.DEVICE, non_blocking=config.PIN_MEMORY)
                batch_y = batch_y.to(config.DEVICE, non_blocking=config.PIN_MEMORY)

                logits, _ = self.model(batch_x)
                loss = criterion(logits, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(batch_x)
                epoch_bar.set_postfix(batch=f"{batch_idx}/{n_batches}",
                                      loss=f"{loss.item():.4f}")

            train_loss /= len(X_train)
            self.train_history.append(train_loss)

            # Validation
            postfix = {"train_loss": f"{train_loss:.4f}"}
            if val_loader is not None and (epoch + 1) % 10 == 0:
                metrics = self._evaluate(val_loader, y_val)
                self.val_history.append(metrics)
                self.scheduler.step(metrics["auc"])

                if metrics["auc"] > best_val_auc:
                    best_val_auc = metrics["auc"]
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                postfix.update({"F1": f"{metrics['f1']:.4f}",
                                "AUC": f"{metrics['auc']:.4f}"})

            epoch_bar.set_postfix(postfix)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(config.DEVICE)
            print(f"[PREDICTOR] Restored best model (AUC={best_val_auc:.4f})")
        else:
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
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(config.DEVICE)
                logits, _ = self.model(batch_x)
                proba = torch.sigmoid(logits)
                all_proba.extend(proba.cpu().numpy())

        y_proba = np.array(all_proba)

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_proba)
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = np.where(
                (precisions + recalls) > 0,
                2 * precisions * recalls / (precisions + recalls),
                0.0,
            )
            best_idx = np.argmax(f1_scores[:-1])
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
