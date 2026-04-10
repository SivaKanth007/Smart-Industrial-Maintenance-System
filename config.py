"""
Global Configuration for Smart Industrial Maintenance System
=============================================================
Central configuration for paths, hyperparameters, and constants.
"""

import os
import psutil
import torch

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved")

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# Device Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_available_memory_gb():
    """Return (gpu_vram_gb or None, system_ram_gb)."""
    ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    vram_gb = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
    return vram_gb, ram_gb

def _auto_batch_size(memory_gb, is_gpu=True):
    """Pick batch size to utilise available memory without OOM.

    Heuristic tiers (conservative — leaves headroom for optimizer states,
    activations, and other processes):
      <=2 GB  → 16
      <=4 GB  → 32
      <=6 GB  → 64
      <=8 GB  → 128
      <=12 GB → 256
      <=16 GB → 512
      >16 GB  → 1024
    CPU/RAM uses the same tiers but with 2× the thresholds (RAM is shared
    with the OS so we need more headroom).
    """
    if not is_gpu:
        memory_gb = memory_gb / 2  # be conservative on shared RAM

    if memory_gb <= 2:
        return 16
    elif memory_gb <= 4:
        return 32
    elif memory_gb <= 6:
        return 64
    elif memory_gb <= 8:
        return 128
    elif memory_gb <= 12:
        return 256
    elif memory_gb <= 16:
        return 512
    else:
        return 1024

def _auto_hidden_dim(memory_gb, is_gpu=True):
    """Scale LSTM hidden dimension with available memory."""
    if not is_gpu:
        memory_gb = memory_gb / 2
    if memory_gb <= 2:
        return 32
    elif memory_gb <= 4:
        return 64
    elif memory_gb <= 8:
        return 128
    else:
        return 256

def _auto_num_workers():
    """DataLoader num_workers: leave 1-2 cores free for the OS."""
    cpus = os.cpu_count() or 1
    return max(0, min(cpus - 2, 4))

def print_system_info():
    """Print device and GPU/RAM information. Call explicitly when needed."""
    vram_gb, ram_gb = _get_available_memory_gb()
    print(f"[CONFIG] Using device: {DEVICE}")
    print(f"[CONFIG] System RAM available: {ram_gb:.1f} GB")
    if torch.cuda.is_available() and vram_gb is not None:
        print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[CONFIG] VRAM: {vram_gb:.1f} GB")
    print(f"[CONFIG] Auto batch size: {AE_BATCH_SIZE} (AE) / {PRED_BATCH_SIZE} (Pred)")
    print(f"[CONFIG] Auto hidden dim: {AE_HIDDEN_DIM} (AE) / {PRED_HIDDEN_DIM} (Pred)")
    print(f"[CONFIG] DataLoader workers: {NUM_WORKERS}")

# =============================================================================
# C-MAPSS Dataset Configuration
# =============================================================================
CMAPSS_DATASET = "behrad3d/nasa-cmaps"  # Kaggle dataset identifier
CMAPSS_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]  # All 4 subsets

# Column names for C-MAPSS
CMAPSS_COLUMNS = (
    ["unit_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors to drop (constant or near-constant in FD001)
SENSORS_TO_DROP = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                   "sensor_16", "sensor_18", "sensor_19"]

# Operational settings (often near-constant in FD001)
OP_SETTINGS_TO_DROP = ["op_setting_3"]

# Active sensor columns after filtering
ACTIVE_SENSORS = [f"sensor_{i}" for i in range(1, 22)
                  if f"sensor_{i}" not in SENSORS_TO_DROP]

# =============================================================================
# Preprocessing Hyperparameters
# =============================================================================
SEQUENCE_LENGTH = 30          # Sliding window length (cycles)
MAX_RUL = 125                 # Cap RUL at this value (piecewise linear)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# Synthetic Augmentation (C-MAPSS)
# =============================================================================
SYNTHETIC_AUGMENT = True              # Enable synthetic training augmentation
SYNTHETIC_TARGET_RATIO = 0.30         # Synthetic = 30% of total training data
SYNTHETIC_NOISE_LEVEL = 0.03          # Gaussian noise scale relative to feature std
SYNTHETIC_DEGRADATION_MODELS = ["exponential", "linear", "polynomial"]

# =============================================================================
# Feature Engineering
# =============================================================================
ROLLING_WINDOWS = [5, 10, 20]  # Windows for rolling statistics
ROLLING_STATS = ["mean", "std", "min", "max"]

# =============================================================================
# Auto-tuned Hardware Parameters
# =============================================================================
_VRAM_GB, _RAM_GB = _get_available_memory_gb()
_IS_GPU = _VRAM_GB is not None
_MEM_GB = _VRAM_GB if _IS_GPU else _RAM_GB
_AUTO_BATCH = _auto_batch_size(_MEM_GB, is_gpu=_IS_GPU)
_AUTO_HIDDEN = _auto_hidden_dim(_MEM_GB, is_gpu=_IS_GPU)
NUM_WORKERS = _auto_num_workers()
PIN_MEMORY = _IS_GPU  # pin_memory speeds up GPU transfers

# =============================================================================
# LSTM Autoencoder (Anomaly Detection)
# =============================================================================
AE_HIDDEN_DIM = _AUTO_HIDDEN
AE_LATENT_DIM = max(16, _AUTO_HIDDEN // 2)
AE_NUM_LAYERS = 2
AE_DROPOUT = 0.2
AE_LEARNING_RATE = 1e-3
AE_EPOCHS = 50
AE_BATCH_SIZE = _AUTO_BATCH
AE_ANOMALY_THRESHOLD_SIGMA = 3.0  # mean + 3*sigma

# =============================================================================
# LSTM Failure Predictor
# =============================================================================
PRED_HIDDEN_DIM = _AUTO_HIDDEN
PRED_NUM_LAYERS = 2
PRED_DROPOUT = 0.3
PRED_LEARNING_RATE = 1e-3
PRED_EPOCHS = 50
PRED_BATCH_SIZE = _AUTO_BATCH
PRED_FAILURE_HORIZON = 30    # Predict failure within h cycles

# =============================================================================
# XGBoost RUL
# =============================================================================
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}
if torch.cuda.is_available():
    XGB_PARAMS["device"] = "cuda"
    XGB_PARAMS["tree_method"] = "hist"

# =============================================================================
# Bayesian Survival Analysis
# =============================================================================
SURVIVAL_CONFIDENCE_LEVELS = [0.90, 0.95]

# =============================================================================
# MILP Optimization
# =============================================================================
MAX_CONCURRENT_CREWS = 3          # Max simultaneous maintenance jobs
DOWNTIME_COST_PER_HOUR = 10000    # $ per hour of unplanned downtime
MAINTENANCE_COST_BASE = 2000      # $ base maintenance cost
SAFETY_RISK_THRESHOLD = 0.7       # Risk above this = mandatory service
SCHEDULING_HORIZON = 10           # Time slots to schedule over

# =============================================================================
# Risk Categories
# =============================================================================
RISK_LEVELS = {
    "critical": {"threshold": 0.7, "label": "Service Immediately", "color": "#FF4444"},
    "elevated": {"threshold": 0.4, "label": "Schedule Soon", "color": "#FFAA00"},
    "normal":   {"threshold": 0.0, "label": "Continue Monitoring", "color": "#44BB44"},
}

# =============================================================================
# Random Seed
# =============================================================================
import random as _random

RANDOM_SEED = 42

import numpy as _np

_random.seed(RANDOM_SEED)
_np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# IMS Bearing Dataset Configuration
# =============================================================================
IMS_DATASET = "vinayak123tyagi/bearing-dataset"  # Kaggle dataset identifier
IMS_RAW_DIR = os.path.join(DATA_DIR, "raw_ims")
IMS_PROCESSED_DIR = os.path.join(DATA_DIR, "processed_ims")

# Ensure IMS directories exist
for d in [IMS_RAW_DIR, IMS_PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

IMS_SAMPLING_RATE = 20480        # 20 kHz, 1-second snapshots → 20,480 points
IMS_SNAPSHOT_LENGTH = 20480      # Data points per snapshot file

IMS_EXPERIMENTS = {
    1: {
        "channels": 8,  # 2 per bearing (X + Y axis)
        "bearings": 4,
        "failed_bearings": [3, 4],
        "failure_modes": ["inner_race", "roller_element"],
        "folder": "1st_test",
    },
    2: {
        "channels": 4,  # 1 per bearing
        "bearings": 4,
        "failed_bearings": [1],
        "failure_modes": ["outer_race"],
        "folder": "2nd_test",
    },
    3: {
        "channels": 4,  # 1 per bearing
        "bearings": 4,
        "failed_bearings": [3],
        "failure_modes": ["outer_race"],
        "folder": "3rd_test",
    },
}

# IMS preprocessing hyperparameters
IMS_MAX_RUL = 125               # Cap pseudo-RUL at 125 snapshots (aligned with notebook)
IMS_SEQUENCE_LENGTH = 30        # Sliding window for LSTM (30 snapshots, aligned with notebook)
IMS_FFT_BANDS = 5               # Frequency bands for spectral energy
IMS_ROLLING_WINDOWS = [10, 50, 100]  # Rolling windows for trend features
