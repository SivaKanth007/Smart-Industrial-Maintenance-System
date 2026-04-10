"""
Attention Weight Visualization
================================
Extracts and visualizes temporal attention weights from the LSTM predictor
to show which time steps contribute most to failure predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import config


class AttentionVisualizer:
    """
    Visualize attention weights from the LSTM failure predictor.

    Provides:
    - Temporal attention heatmaps
    - Sensor contribution over time
    - Critical time-step identification
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : LSTMPredictor — trained model with attention mechanism
        """
        self.model = model
        self.model.eval()

    def extract_attention(self, X):
        """
        Extract attention weights for input sequences.

        Parameters
        ----------
        X : np.ndarray, shape (N, seq_len, n_features)

        Returns
        -------
        attention_weights : np.ndarray, shape (N, seq_len)
        predictions : np.ndarray, shape (N,)
        """
        # Use model's batched prediction path to avoid OOM on small GPUs.
        proba, attn = self.model.predict_proba(torch.as_tensor(X, dtype=torch.float32))
        return attn, proba

    def plot_attention_heatmap(self, X, unit_ids=None, n_samples=10,
                               save_path=None):
        """
        Plot attention heatmap showing temporal importance.

        Parameters
        ----------
        X : np.ndarray — input sequences
        unit_ids : np.ndarray — unit identifiers
        n_samples : int — number of samples to display
        save_path : str — file path to save plot
        """
        idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X_subset = X[idx]
        attn_weights, predictions = self.extract_attention(X_subset)

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            attn_weights,
            cmap="YlOrRd",
            xticklabels=5,
            yticklabels=[f"Sample {i}" for i in range(len(attn_weights))],
            ax=ax,
            cbar_kws={"label": "Attention Weight"},
        )
        ax.set_xlabel("Time Step (Cycle)", fontsize=12)
        ax.set_ylabel("Sample", fontsize=12)
        ax.set_title("Temporal Attention Weights — LSTM Failure Predictor",
                      fontsize=14, fontweight='bold')

        # Add prediction scores as annotations
        for i, pred in enumerate(predictions):
            risk = "HIGH" if pred > 0.7 else "MED" if pred > 0.4 else "LOW"
            color = "#FF4444" if pred > 0.7 else "#FFAA00" if pred > 0.4 else "#44BB44"
            ax.text(len(attn_weights[0]) + 0.5, i + 0.5,
                    f"  P={pred:.2f} [{risk}]",
                    va='center', fontsize=9, color=color, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[ATTENTION] Saved heatmap to {save_path}")
        plt.close()

    def plot_average_attention(self, X, y_binary, save_path=None):
        """
        Compare average attention patterns between failure and non-failure cases.
        """
        attn_weights, _ = self.extract_attention(X)

        # Split by failure/non-failure
        fail_mask = y_binary == 1
        normal_mask = y_binary == 0

        avg_fail = attn_weights[fail_mask].mean(axis=0) if fail_mask.sum() > 0 else np.zeros(attn_weights.shape[1])
        avg_normal = attn_weights[normal_mask].mean(axis=0) if normal_mask.sum() > 0 else np.zeros(attn_weights.shape[1])

        fig, ax = plt.subplots(figsize=(12, 5))
        x_axis = np.arange(len(avg_fail))

        ax.fill_between(x_axis, avg_fail, alpha=0.3, color="#FF4444", label="Failure Cases")
        ax.fill_between(x_axis, avg_normal, alpha=0.3, color="#44BB44", label="Normal Cases")
        ax.plot(x_axis, avg_fail, color="#FF4444", linewidth=2)
        ax.plot(x_axis, avg_normal, color="#44BB44", linewidth=2)

        ax.set_xlabel("Time Step (Most Recent Cycles)", fontsize=12)
        ax.set_ylabel("Average Attention Weight", fontsize=12)
        ax.set_title("Attention Pattern: Failure vs Normal Operation",
                      fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[ATTENTION] Saved comparison plot to {save_path}")
        plt.close()

    def plot_sensor_temporal_contribution(self, X, sensor_names=None,
                                          top_n=5, save_path=None):
        """
        Show how individual sensor contributions evolve over the time window,
        weighted by attention.

        Parameters
        ----------
        X : np.ndarray, shape (N, seq_len, n_features)
        sensor_names : list[str]
        top_n : int — number of top sensors to show
        """
        attn_weights, predictions = self.extract_attention(X)

        # High-risk samples only
        high_risk_mask = predictions > 0.5
        if high_risk_mask.sum() == 0:
            print("[ATTENTION] No high-risk predictions to visualize")
            return

        X_risk = X[high_risk_mask]
        attn_risk = attn_weights[high_risk_mask]

        # Average attention-weighted sensor values
        # weighted_X: (N, seq_len, n_features) * (N, seq_len, 1)
        weighted_X = X_risk * attn_risk[:, :, np.newaxis]
        avg_weighted = weighted_X.mean(axis=0)  # (seq_len, n_features)

        # Get top sensors by overall contribution
        sensor_importance = np.abs(avg_weighted).sum(axis=0)
        top_idx = np.argsort(sensor_importance)[-top_n:][::-1]

        if sensor_names is None:
            sensor_names = [f"Sensor {i}" for i in range(X.shape[2])]

        fig, ax = plt.subplots(figsize=(12, 6))
        x_axis = np.arange(avg_weighted.shape[0])

        colors = plt.cm.Set1(np.linspace(0, 1, top_n))
        for i, idx in enumerate(top_idx):
            ax.plot(x_axis, avg_weighted[:, idx],
                    label=sensor_names[idx], linewidth=2, color=colors[i])

        ax.set_xlabel("Time Step (Most Recent Cycles)", fontsize=12)
        ax.set_ylabel("Attention-Weighted Sensor Value", fontsize=12)
        ax.set_title(f"Top-{top_n} Sensor Contributions (High-Risk Machines)",
                      fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[ATTENTION] Saved sensor contribution plot to {save_path}")
        plt.close()

    def identify_critical_timesteps(self, X, threshold=0.1):
        """
        Identify time steps that receive disproportionately high attention.

        Returns
        -------
        pd.DataFrame — critical time steps per sample
        """
        import pandas as pd

        attn_weights, predictions = self.extract_attention(X)

        critical = []
        for i in range(len(attn_weights)):
            high_attn_steps = np.where(attn_weights[i] > threshold)[0]
            if len(high_attn_steps) > 0:
                critical.append({
                    "sample_idx": i,
                    "prediction": predictions[i],
                    "critical_steps": high_attn_steps.tolist(),
                    "max_attention": attn_weights[i].max(),
                    "max_attention_step": int(np.argmax(attn_weights[i])),
                })

        df = pd.DataFrame(critical)
        print(f"[ATTENTION] Found {len(critical)} samples with critical time steps")
        return df
