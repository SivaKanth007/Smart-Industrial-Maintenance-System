"""
SHAP-based Model Explainability
================================
Feature attribution for XGBoost and LSTM models using SHAP values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import config


class SHAPExplainer:
    """
    SHAP-based explainability for the maintenance prediction models.

    Provides:
    - Global feature importance (bar plots, beeswarm)
    - Local explanation (force plots per prediction)
    - Sensor ranking for root cause analysis
    """

    def __init__(self, model, model_type="xgboost"):
        """
        Parameters
        ----------
        model : trained model object
        model_type : str — 'xgboost' or 'lstm'
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None

    def setup_explainer(self, X_background=None):
        """Initialize SHAP explainer based on model type."""
        if self.model_type == "xgboost":
            self.explainer = shap.TreeExplainer(self.model.model)
            print("[SHAP] TreeExplainer initialized for XGBoost")
        elif self.model_type == "lstm":
            if X_background is None:
                raise ValueError("X_background required for DeepExplainer")
            import torch
            self.explainer = shap.DeepExplainer(
                self.model,
                torch.FloatTensor(X_background[:100]).to(config.DEVICE)
            )
            print("[SHAP] DeepExplainer initialized for LSTM")
        else:
            # Fallback to KernelExplainer
            if X_background is None:
                raise ValueError("X_background required for KernelExplainer")
            background = shap.sample(X_background, 100)
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            print("[SHAP] KernelExplainer initialized")

    def compute_shap_values(self, X, max_samples=500):
        """
        Compute SHAP values for input data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame — input features
        max_samples : int — limit samples for speed

        Returns
        -------
        shap.Explanation or np.ndarray of SHAP values
        """
        if self.explainer is None:
            self.setup_explainer()

        if len(X) > max_samples:
            rng = np.random.default_rng(config.RANDOM_SEED)
            idx = rng.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
        else:
            X_sample = X

        print(f"[SHAP] Computing SHAP values for {len(X_sample)} samples...")
        self.shap_values = self.explainer.shap_values(X_sample)

        if isinstance(X_sample, pd.DataFrame):
            self.feature_names = list(X_sample.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_sample.shape[-1])]

        self.X_sample = X_sample
        print("[SHAP] SHAP values computed successfully")
        return self.shap_values

    def plot_global_importance(self, save_path=None, top_n=20):
        """
        Plot global feature importance (bar chart).
        """
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values first")

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=top_n,
            show=False,
        )
        plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SHAP] Saved importance plot to {save_path}")
        plt.close()

    def plot_beeswarm(self, save_path=None, top_n=20):
        """
        Plot SHAP beeswarm (shows distribution of feature effects).
        """
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values first")

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            max_display=top_n,
            show=False,
        )
        plt.title("SHAP Feature Impact Distribution", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SHAP] Saved beeswarm plot to {save_path}")
        plt.close()

    def get_sensor_ranking(self):
        """
        Rank sensors by mean absolute SHAP value.

        Returns
        -------
        pd.DataFrame — sorted by importance
        """
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values first")

        sv = self.shap_values
        if isinstance(sv, list):
            sv = sv[0] if len(sv) > 0 else sv

        mean_abs = np.abs(sv).mean(axis=0)

        if mean_abs.ndim > 1:
            # For sequence data, average across time steps
            mean_abs = mean_abs.mean(axis=0)

        ranking = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs)],
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        print("\n[SHAP] Sensor Ranking (Top 15):")
        print(ranking.head(15).to_string(index=False))

        return ranking

    def explain_single_prediction(self, idx=0, save_path=None):
        """
        Generate force plot for a single prediction.
        """
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values first")

        sv = self.shap_values
        if isinstance(sv, list):
            sv = sv[0]

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]

        fig = shap.force_plot(
            base_value,
            sv[idx],
            self.X_sample.iloc[idx] if isinstance(self.X_sample, pd.DataFrame)
            else self.X_sample[idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SHAP] Saved force plot to {save_path}")
        plt.close()
