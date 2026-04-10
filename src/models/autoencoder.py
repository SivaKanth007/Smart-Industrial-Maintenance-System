"""
LSTM Temporal Autoencoder for Anomaly Detection
================================================
Learns compressed representations of normal sensor patterns.
Anomaly score = reconstruction error (MSE).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config


class LSTMEncoder(nn.Module):
    """LSTM encoder: compresses sequence into latent representation."""

    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        latent = self.fc(last_hidden)  # (batch, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    """LSTM decoder: reconstructs sequence from latent representation."""

    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        # latent: (batch, latent_dim)
        hidden = self.fc(latent)  # (batch, hidden_dim)
        # Repeat for each time step
        repeated = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(repeated)  # (batch, seq_len, hidden_dim)
        output = self.output_layer(lstm_out)  # (batch, seq_len, output_dim)
        return output


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based temporal autoencoder for anomaly detection.

    Architecture:
    - Encoder: 2-layer LSTM (input_dim → hidden_dim → latent_dim)
    - Decoder: 2-layer LSTM (latent_dim → hidden_dim → input_dim)

    Anomaly score = MSE between input and reconstruction.
    """

    def __init__(self, input_dim, hidden_dim=None, latent_dim=None,
                 num_layers=None, dropout=None, seq_len=None):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or config.AE_HIDDEN_DIM
        self.latent_dim = latent_dim or config.AE_LATENT_DIM
        self.num_layers = num_layers or config.AE_NUM_LAYERS
        self.dropout = dropout or config.AE_DROPOUT
        self.seq_len = seq_len or config.SEQUENCE_LENGTH

        self.encoder = LSTMEncoder(
            input_dim, self.hidden_dim, self.latent_dim,
            self.num_layers, self.dropout
        )
        self.decoder = LSTMDecoder(
            self.latent_dim, self.hidden_dim, input_dim,
            self.seq_len, self.num_layers, self.dropout
        )

        # Training stats for threshold
        self.train_losses = []
        self.threshold = None

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def compute_anomaly_score(self, x):
        """
        Compute per-sample anomaly score (reconstruction error).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        scores : np.ndarray, shape (batch,) — MSE per sample
        """
        self.eval()
        self.to(config.DEVICE)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        batch_size = config.AE_BATCH_SIZE
        scores = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch_x = x[start:start + batch_size].to(config.DEVICE)
                reconstruction = self.forward(batch_x)
                mse = ((batch_x - reconstruction) ** 2).mean(dim=(1, 2))
                scores.append(mse.cpu().numpy())

        return np.concatenate(scores, axis=0) if scores else np.array([])

    def set_threshold(self, train_scores, sigma=None):
        """
        Set anomaly threshold as mean + sigma * std of training scores.
        """
        sigma = sigma or config.AE_ANOMALY_THRESHOLD_SIGMA
        self.threshold = np.mean(train_scores) + sigma * np.std(train_scores)
        print(f"[AUTOENCODER] Threshold set: {self.threshold:.6f} "
              f"(mean={np.mean(train_scores):.6f}, std={np.std(train_scores):.6f})")
        return self.threshold

    def detect_anomalies(self, x):
        """
        Detect anomalies in input sequences.

        Returns
        -------
        scores : np.ndarray — anomaly scores
        is_anomaly : np.ndarray — boolean array
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call set_threshold first.")
        scores = self.compute_anomaly_score(x)
        is_anomaly = scores > self.threshold
        return scores, is_anomaly


class AutoencoderTrainer:
    """Training loop for the LSTM Autoencoder."""

    def __init__(self, model, lr=None, epochs=None, batch_size=None):
        self.model = model.to(config.DEVICE)
        self.lr = lr or config.AE_LEARNING_RATE
        self.epochs = epochs or config.AE_EPOCHS
        self.batch_size = batch_size or config.AE_BATCH_SIZE
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.criterion = nn.MSELoss()
        self.train_history = []
        self.val_history = []

    def train(self, X_train, X_val=None):
        """
        Train the autoencoder.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, seq_len, n_features)
        X_val : np.ndarray, optional
        """
        train_tensor = torch.FloatTensor(X_train)
        train_loader = DataLoader(
            TensorDataset(train_tensor, train_tensor),
            batch_size=self.batch_size, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        )

        val_loader = None
        if X_val is not None:
            val_tensor = torch.FloatTensor(X_val)
            val_loader = DataLoader(
                TensorDataset(val_tensor, val_tensor),
                batch_size=self.batch_size, shuffle=False,
                num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
            )

        best_val_loss = float("inf")
        best_state = None

        print(f"\n[AUTOENCODER] Training on {config.DEVICE} "
              f"({len(X_train)} samples, {self.epochs} epochs, "
              f"batch_size={self.batch_size})")

        n_batches = len(train_loader)
        epoch_bar = tqdm(range(self.epochs), desc="[AE] Epochs", unit="epoch")
        for epoch in epoch_bar:
            # Training
            self.model.train()
            train_loss = 0
            for batch_idx, (batch_x, _) in enumerate(train_loader, 1):
                batch_x = batch_x.to(config.DEVICE, non_blocking=config.PIN_MEMORY)
                reconstruction = self.model(batch_x)
                loss = self.criterion(reconstruction, batch_x)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(batch_x)
                epoch_bar.set_postfix(batch=f"{batch_idx}/{n_batches}",
                                      loss=f"{loss.item():.6f}")

            train_loss /= len(X_train)
            self.train_history.append(train_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, _ in val_loader:
                        batch_x = batch_x.to(config.DEVICE, non_blocking=config.PIN_MEMORY)
                        reconstruction = self.model(batch_x)
                        loss = self.criterion(reconstruction, batch_x)
                        val_loss += loss.item() * len(batch_x)
                val_loss /= len(X_val)
                self.val_history.append(val_loss)
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # Update epoch progress bar
            postfix = {"train_loss": f"{train_loss:.6f}"}
            if val_loss is not None:
                postfix["val_loss"] = f"{val_loss:.6f}"
            epoch_bar.set_postfix(postfix)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(config.DEVICE)
            print(f"[AUTOENCODER] Restored best model (val_loss={best_val_loss:.6f})")

        # Set anomaly threshold from validation scores (not training scores).
        # Training reconstruction errors are lower/less variable because the
        # model has partially memorized X_train — val scores better represent
        # the threshold that will generalize to unseen machines.
        self.model.eval()
        if X_val is not None:
            threshold_data = torch.FloatTensor(X_val)
            threshold_source = "val"
        else:
            threshold_data = torch.FloatTensor(X_train)
            threshold_source = "train (no val provided)"
            print("[AUTOENCODER] WARNING: No X_val provided — threshold set from training "
                  "data. This may not generalize to unseen machines.")
        threshold_scores = self.model.compute_anomaly_score(threshold_data)
        self.model.set_threshold(threshold_scores)
        print(f"[AUTOENCODER] Threshold derived from: {threshold_source}")

        return self.model

    def save_model(self, filepath=None):
        """Save trained model."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "autoencoder.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "threshold": self.model.threshold,
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim,
            "latent_dim": self.model.latent_dim,
            "num_layers": self.model.num_layers,
            "seq_len": self.model.seq_len,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }, filepath)
        print(f"[AUTOENCODER] Model saved to {filepath}")


def load_autoencoder(filepath=None):
    """Load a trained autoencoder."""
    filepath = filepath or os.path.join(config.MODELS_DIR, "autoencoder.pt")
    checkpoint = torch.load(filepath, map_location=config.DEVICE, weights_only=False)

    model = LSTMAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint.get("hidden_dim"),
        latent_dim=checkpoint.get("latent_dim"),
        num_layers=checkpoint.get("num_layers"),
        seq_len=checkpoint.get("seq_len"),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.threshold = checkpoint["threshold"]
    model.to(config.DEVICE)
    model.eval()

    print(f"[AUTOENCODER] Loaded model (threshold={model.threshold:.6f})")
    return model
