# Smart Industrial Maintenance System

> **FSE 570 Data Science Capstone** — Arizona State University, Ira A. Fulton Schools of Engineering

An end-to-end AI-powered maintenance decision support system that detects anomalies in industrial sensors, predicts machine failures with calibrated uncertainty, estimates Remaining Useful Life (RUL), and generates optimized crew-constrained maintenance schedules. Validated on two NASA datasets: **C-MAPSS turbofan** and **IMS Bearing** vibration data.

---

## Team

| Name | Role |
|------|------|
| Anoushka Jaydas Dighe | Team Member |
| Deva Siva Kanth Tavvala | Team Member |
| Mohit Kumar Petla | Team Member |
| Umang Rajnikant Bid | Team Member |
| Urvansh Jignesh Shah | Team Member |

---

## System Pipeline

```
Raw Sensor Data
     │
     ▼
Preprocessing & Feature Engineering
     │
     ▼
ML Model Suite
 ├── LSTM Temporal Autoencoder     (anomaly detection)
 ├── LSTM Classifier + Attention   (failure probability)
 ├── XGBoost Regression            (RUL estimation)
 └── Bayesian Weibull Survival     (uncertainty quantification)
     │
     ▼
MILP Maintenance Scheduler
     │
     ▼
Monte Carlo Policy Simulation
     │
     ▼
Streamlit Dashboard (7 pages)
```

---

## Performance Results

| Model | Metric | Value |
|-------|--------|-------|
| LSTM Failure Predictor | F1-Score | **0.933** |
| LSTM Failure Predictor | AUC-ROC | **0.997** |
| XGBoost RUL | RMSE | **10.48 cycles** |
| XGBoost RUL | R² | **0.937** |
| Bayesian Survival | C-Index | **0.992** |
| MILP Optimization | Cost Reduction | **97.4%** vs reactive |
| MILP Optimization | Downtime Reduction | **72.4%** vs reactive |
| Monte Carlo Simulation | Failure Reduction | **99.0%** vs reactive |

---

## Quick Start

### Requirements

- Python 3.9 or higher
- No GPU required — CUDA is auto-detected and used if available

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/SivaKanth007/Smart-Industrial-Maintenance-System.git
cd Smart-Industrial-Maintenance-System
```

---

### Step 2 — Create a virtual environment

Using a virtual environment keeps dependencies isolated and avoids conflicts with your system Python.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -e ".[test]"
```

This installs all required packages including PyTorch, XGBoost, Streamlit, scikit-learn, lifelines, SHAP, PuLP, and pytest.

---

### Step 4 — Launch the dashboard

Pre-trained models and results are included in the repository. You can launch the dashboard immediately after install — no training required.

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501**

Seven pages: Fleet Overview · Risk Assessment · Maintenance Schedule · Model Performance · Explainability & AI Insights · Maintenance History · Operational Context

---

### Step 5 — Run tests

```bash
python -m pytest
```

All **50 unit tests** across 4 modules should pass.

---

## Retraining the Models

Pre-trained results are already committed. Retrain only if you want to update the models with new data or changed hyperparameters.

### Path A — Python scripts (C-MAPSS turbofan dataset)

Run these three commands in order:

```bash
# 1. Train all models (downloads NASA C-MAPSS dataset automatically)
python scripts/train_all.py

# 2. Score all engines and generate maintenance recommendations
python scripts/run_pipeline.py

# 3. Launch dashboard to view updated results
streamlit run dashboard/app.py
```

**What each step does:**

| Step | Command | Output |
|------|---------|--------|
| Train | `train_all.py` | Model weights in `models/saved/`, preprocessed sequences in `data/processed/`, synthetic maintenance logs in `data/synthetic/` |
| Inference | `run_pipeline.py` | `data/processed/recommendations.csv` (per-engine risk + schedule) |
| Dashboard | `streamlit run` | Reads all outputs above and displays them |

**Training time:** ~5 minutes on CPU. Significantly faster on GPU.

For GPU support, install CUDA-enabled PyTorch first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

### Path B — Jupyter notebook (IMS Bearing vibration dataset)

The notebooks run the same 4-model pipeline on NASA IMS Bearing data (accelerometer signals from 3 bearing degradation experiments).

```bash
# Install Jupyter if not already available
pip install jupyter

# Open the notebook
jupyter notebook notebooks/Smart_Industrial_Maintenance_Repo_Pipeline.ipynb
```

- `Smart_Industrial_Maintenance_Repo_Pipeline.ipynb` — uses local `src/` imports (requires this repo)
- `Smart_Industrial_Maintenance_Standalone_Pipeline.ipynb` — fully self-contained, no local imports needed

Both notebooks download the NASA IMS Bearing dataset (~2 GB) automatically via `kagglehub`. Kaggle credentials are required — place your `kaggle.json` in `~/.kaggle/`.

Alternatively, train via script:
```bash
python scripts/train_ims.py
```

> **Note:** The notebooks train IMS bearing models and save them to `models/saved/ims_*.pt/.pkl`. They do not overwrite the C-MAPSS models or `recommendations.csv`, so the main dashboard is unaffected.

---

## Make Commands Reference

| Command | Description |
|---------|-------------|
| `make install` | Install project in editable mode |
| `make install-gpu` | Install with CUDA GPU support |
| `make train` | Train C-MAPSS models (downloads data automatically) |
| `make train-ims` | Train IMS bearing models |
| `make inference` | Run inference and generate recommendations |
| `make dashboard` | Launch Streamlit dashboard |
| `make test` | Run all 50 unit tests |
| `make clean` | Remove caches and `__pycache__` directories |
| `make help` | Show all available commands |

---

## Project Structure

```
Smart-Industrial-Maintenance-System/
│
├── config.py                        # All settings: paths, hyperparameters, constants
├── pyproject.toml                   # Python packaging (pip install -e .)
├── Makefile                         # Standardized commands (make train, make test, etc.)
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Test configuration
├── README.md                        # This file
├── PROJECT_REPORT.md                # Full capstone project report
│
├── scripts/
│   ├── train_all.py                 # End-to-end C-MAPSS training orchestrator
│   ├── train_ims.py                 # IMS bearing model training
│   └── run_pipeline.py             # Full inference pipeline → recommendations.csv
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard (7 pages)
│
├── src/
│   ├── data/
│   │   ├── download.py              # NASA C-MAPSS dataset download
│   │   ├── ims_download.py          # IMS bearing dataset download (kagglehub)
│   │   ├── ims_preprocess.py        # IMS vibration signal preprocessing + feature extraction
│   │   ├── preprocess.py            # C-MAPSS cleaning, normalization, windowing
│   │   ├── feature_engineering.py  # 200+ rolling, trend, lag, interaction features
│   │   ├── synthetic_generator.py  # Maintenance logs + operational context generation
│   │   └── synthetic_cmapss.py     # Synthetic C-MAPSS augmentation
│   │
│   ├── models/
│   │   ├── autoencoder.py           # LSTM Temporal Autoencoder (anomaly detection)
│   │   ├── lstm_predictor.py        # LSTM Classifier with Attention (failure probability)
│   │   ├── xgboost_rul.py           # XGBoost RUL regression
│   │   └── bayesian_survival.py    # Bayesian Weibull Survival Analysis
│   │
│   ├── explainability/
│   │   ├── shap_analysis.py         # SHAP feature attribution (TreeSHAP + DeepSHAP)
│   │   └── attention_viz.py         # Temporal attention heatmap visualization
│   │
│   ├── optimization/
│   │   └── milp_scheduler.py        # PuLP MILP maintenance scheduler
│   │
│   └── evaluation/
│       └── simulation.py            # Monte Carlo maintenance policy comparison
│
├── tests/                           # Unit tests — 50 tests across 4 modules
│   ├── test_preprocess.py           # 15 tests: C-MAPSS preprocessing pipeline
│   ├── test_models.py               # 16 tests: autoencoder, predictor, XGBoost, feature engineering, reproducibility
│   ├── test_optimizer.py            # 8 tests: MILP scheduling, crew constraints
│   └── test_ims.py                  # 11 tests: IMS feature extraction + model compatibility
│
├── notebooks/
│   ├── Smart_Industrial_Maintenance_Repo_Pipeline.ipynb       # IMS pipeline (repo-integrated)
│   └── Smart_Industrial_Maintenance_Standalone_Pipeline.ipynb # IMS pipeline (self-contained)
│
├── Docs/                            # Project documentation and presentations
│
├── data/                            # Auto-created after training
│   ├── raw/                         # NASA C-MAPSS dataset files
│   ├── raw_ims/                     # NASA IMS Bearing dataset (3 experiments)
│   ├── processed/                   # Preprocessed sequences + recommendations.csv
│   ├── processed_ims/               # IMS preprocessed features
│   └── synthetic/                   # Generated maintenance logs + operational context
│
├── models/saved/                    # Trained model artifacts
│   ├── autoencoder.pt               # C-MAPSS LSTM Autoencoder weights + threshold
│   ├── lstm_predictor.pt            # C-MAPSS LSTM Predictor weights + attention
│   ├── xgboost_rul.pkl              # C-MAPSS XGBoost model + feature importance
│   ├── xgboost_model.pkl            # (alias)
│   ├── bayesian_survival.pkl        # C-MAPSS Weibull AFT model
│   ├── survival_model.pkl           # (alias)
│   ├── preprocessor.pkl             # MinMaxScaler + active feature column config
│   ├── ims_autoencoder.pt           # IMS LSTM Autoencoder
│   ├── ims_predictor.pt             # IMS LSTM Predictor
│   ├── ims_xgboost.pkl              # IMS XGBoost RUL
│   └── ims_survival.pkl             # IMS Weibull Survival
│
└── assets/                          # Dashboard screenshots for documentation
```

---

## Dashboard Preview

The dashboard runs locally via `streamlit run dashboard/app.py`. Screenshots of the static pages are shown below — the **Model Performance** and **Explainability & AI Insights** pages are data-dependent and best viewed live after training.

### Fleet Overview
![Fleet Overview](assets/dashboard_fleet_overview.png)

### Risk Assessment
![Risk Assessment](assets/dashboard_risk_assessment.png)

### Maintenance Schedule
![Maintenance Schedule](assets/dashboard_maintenance_schedule.png)

### Maintenance History
![Maintenance History](assets/dashboard_maintenance_history.png)

### Operational Context
![Operational Context](assets/dashboard_operational_context.png)

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Fleet Overview** | Aggregate fleet health: machines monitored, near-failure count, avg RUL, RUL distribution histogram, per-unit health bar chart |
| **Risk Assessment** | Per-machine failure risk (Critical / Elevated / Normal), color-coded table, risk distribution pie chart |
| **Maintenance Schedule** | MILP optimizer output as interactive Gantt chart, scheduled slot details table |
| **Model Performance** | Live model metrics cards (F1, AUC, RMSE, R², C-Index), per-model performance breakdown, Monte Carlo simulation results |
| **Explainability & AI Insights** | XGBoost feature importance ranking, SHAP-based sensor contribution analysis, attention pattern description, model interpretation guide |
| **Maintenance History** | Historical maintenance logs: total events, total cost, avg downtime, cost-by-failure-type bar chart, planned vs unplanned ratio |
| **Operational Context** | Fleet composition by machine type and priority, cycles vs temperature scatter, machine specifications table |

---

## Risk Levels

| Level | Condition | Action |
|-------|-----------|--------|
| Critical | Risk >= 70% | Service Immediately |
| Elevated | Risk 40–70% | Schedule Soon |
| Normal | Risk < 40% | Continue Monitoring |

---

## Key Configuration Parameters

All hyperparameters are centralized in `config.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEQUENCE_LENGTH` | 30 | LSTM sliding window size (cycles) |
| `MAX_RUL` | 125 | Maximum RUL cap (piecewise-linear model) |
| `TRAIN_RATIO` | 0.70 | Training set proportion |
| `VAL_RATIO` | 0.15 | Validation set proportion |
| `TEST_RATIO` | 0.15 | Test set proportion |
| `AE_EPOCHS` | 50 | Autoencoder training epochs |
| `PRED_EPOCHS` | 50 | LSTM Predictor training epochs |
| `PRED_FAILURE_HORIZON` | 30 | Failure prediction horizon (cycles) |
| `AE_ANOMALY_THRESHOLD_SIGMA` | 3.0 | Anomaly threshold: mean + N * sigma |
| `MAX_CONCURRENT_CREWS` | 3 | MILP crew capacity constraint |
| `DOWNTIME_COST_PER_HOUR` | 10000 | Unplanned downtime cost (USD/hr) |
| `MAINTENANCE_COST_BASE` | 2000 | Base maintenance job cost (USD) |
| `SAFETY_RISK_THRESHOLD` | 0.7 | Mandatory service risk threshold |
| `SCHEDULING_HORIZON` | 10 | Number of scheduling time slots |

---

## GPU Support

GPU acceleration is automatic. The system detects CUDA at startup:

```python
# config.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

XGBoost also uses GPU when available (`tree_method=hist`, `device=cuda`). Tested on NVIDIA RTX 3050 Ti Laptop GPU.

To install GPU-enabled PyTorch:
```bash
make install-gpu
# OR
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

## Bring Your Own Data

This system is designed as a reusable baseline for production industrial use.

### Expected Input Format

| Column | Type | Description |
|--------|------|-------------|
| `unit_id` | int | Machine/unit identifier |
| `cycle` | int | Monotonically increasing time step per unit |
| `sensor_1` ... `sensor_N` | float | Sensor readings |
| `op_setting_1` ... `op_setting_3` | float | (Optional) Operating conditions |

### Retrain Steps

1. Update `config.py`: set `CMAPSS_COLUMNS`, `SENSORS_TO_DROP`, `ACTIVE_SENSORS` to match your data
2. Place your data in `data/raw/` as space-separated text files (no header row)
3. Run `make train` to retrain all models
4. Run `make inference` to generate recommendations
5. Run `make dashboard` to view results

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `No data found` error | Run `python scripts/train_all.py` first |
| Dashboard shows nothing | Run `python scripts/run_pipeline.py` to generate recommendations |
| Tests not found | Run `python -m pytest` from the project root directory |
| IMS download fails | Ensure `kagglehub` is installed: `pip install kagglehub` and Kaggle credentials are set |
| CUDA out of memory | Set `device=cpu` in `config.py` or reduce batch size |

---

## Datasets

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| NASA C-MAPSS FD001 | NASA Prognostics Center / Kaggle | ~2 MB | Turbofan engine run-to-failure simulation, 100 units, 21 sensors |
| NASA IMS Bearing | NASA IMS Center / Kaggle | ~2 GB | Accelerated bearing degradation, 3 experiments, 4 accelerometers each |

Both datasets are downloaded automatically by the training scripts. No manual download required.

---

## License

Developed for FSE 570 Data Science Capstone at Arizona State University.
