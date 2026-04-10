"""
IMS Bearing Dataset Download Module
=====================================
Downloads and loads the NASA IMS bearing vibration dataset.
Supports Kaggle download with synthetic fallback.
"""

import os
import zipfile
import numpy as np
import pandas as pd

import config


def download_ims(output_dir=None):
    """
    Download IMS bearing dataset from Kaggle.

    Falls back to synthetic data generation if download fails.

    Parameters
    ----------
    output_dir : str
        Directory to save downloaded files.
    """
    output_dir = output_dir or config.IMS_RAW_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Check if data already exists (look for experiment folders)
    existing = [d for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d))]
    if existing:
        print(f"[IMS DOWNLOAD] Dataset already present ({len(existing)} folders). Skipping.")
        for d in sorted(existing):
            n_files = len(os.listdir(os.path.join(output_dir, d)))
            print(f"  - {d}/ ({n_files} files)")
        return

    # Try Kaggle API download
    print("[IMS DOWNLOAD] Attempting Kaggle API download...")
    try:
        _download_via_kaggle(output_dir)
        _organize_ims_files(output_dir)
        return
    except Exception as e:
        print(f"[IMS DOWNLOAD] Kaggle download failed: {e}")

    # Fallback: generate synthetic data
    print("\n[IMS DOWNLOAD] Generating synthetic IMS-like bearing data for development...")
    _generate_fallback_ims_data(output_dir)


def _download_via_kaggle(output_dir):
    """Download via Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError("kaggle package not installed. Install with: pip install kaggle")

    api = KaggleApi()
    api.authenticate()
    print(f"[IMS DOWNLOAD] Downloading {config.IMS_DATASET}...")
    api.dataset_download_files(config.IMS_DATASET, path=output_dir, unzip=True)
    print("[IMS DOWNLOAD] Kaggle download complete.")


def _organize_ims_files(output_dir):
    """Organize downloaded files into experiment folders if needed."""
    # Extract any nested zips
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.zip'):
                zip_path = os.path.join(root, f)
                print(f"[IMS DOWNLOAD] Extracting: {f}")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(root)
                    os.remove(zip_path)
                except zipfile.BadZipFile:
                    pass

    # Verify experiment folders exist
    for exp_id, exp_info in config.IMS_EXPERIMENTS.items():
        folder = exp_info["folder"]
        exp_path = os.path.join(output_dir, folder)
        if os.path.isdir(exp_path):
            n_files = len([f for f in os.listdir(exp_path)
                          if os.path.isfile(os.path.join(exp_path, f))])
            print(f"[IMS DOWNLOAD] Experiment {exp_id}: {n_files} files in {folder}/")
        else:
            print(f"[IMS DOWNLOAD] Warning: {folder}/ not found")


def _generate_fallback_ims_data(output_dir):
    """
    Generate synthetic IMS-like vibration data for development/testing.

    Creates small experiment folders with synthetic bearing vibration snapshots
    that simulate degradation patterns similar to real IMS data.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)

    for exp_id, exp_info in config.IMS_EXPERIMENTS.items():
        folder = exp_info["folder"]
        exp_path = os.path.join(output_dir, folder)
        os.makedirs(exp_path, exist_ok=True)

        n_channels = exp_info["channels"]
        n_snapshots = 200  # Small for dev/testing (real has 1000-6000+)
        failed_bearings = exp_info["failed_bearings"]

        print(f"[IMS DOWNLOAD] Generating experiment {exp_id}: "
              f"{n_snapshots} snapshots, {n_channels} channels")

        for snap_idx in range(n_snapshots):
            # Create timestamp-like filename
            filename = f"snapshot_{snap_idx:04d}.csv"
            filepath = os.path.join(exp_path, filename)

            # Generate vibration data for each channel
            n_points = config.IMS_SNAPSHOT_LENGTH
            data = np.zeros((n_points, n_channels))
            degradation_progress = snap_idx / n_snapshots

            for ch in range(n_channels):
                # Determine which bearing this channel belongs to
                if n_channels == 8:
                    bearing_id = ch // 2 + 1  # 2 channels per bearing
                else:
                    bearing_id = ch + 1  # 1 channel per bearing

                # Base vibration signal
                t = np.linspace(0, 1, n_points)
                base_freq = 100 + rng.normal(0, 5)
                signal = rng.normal(0, 0.5, n_points)
                signal += np.sin(2 * np.pi * base_freq * t) * 0.3

                # Add degradation for failed bearings
                if bearing_id in failed_bearings:
                    # Increasing RMS and kurtosis as bearing degrades
                    deg_factor = degradation_progress ** 2
                    signal *= (1 + deg_factor * 3)
                    # Add impulse-like defect signals
                    n_impulses = int(deg_factor * 20)
                    for _ in range(n_impulses):
                        pos = rng.integers(0, n_points)
                        signal[pos] += rng.normal(0, deg_factor * 5)

                data[:, ch] = signal

            # Save as tab-delimited (matching IMS format)
            np.savetxt(filepath, data, delimiter='\t', fmt='%.6f')

        print(f"  -> {exp_path}/ ({n_snapshots} files)")


def load_ims_experiment(experiment=1, data_dir=None):
    """
    Load one IMS experiment's data.

    Each file is a 1-second vibration snapshot. Files are loaded in
    sorted order (temporal ordering).

    Parameters
    ----------
    experiment : int
        Experiment number (1, 2, or 3).
    data_dir : str
        Directory containing IMS data.

    Returns
    -------
    pd.DataFrame
        Columns: file_index, bearing1_ch1, bearing1_ch2, ..., bearing4_ch1
        Each row = summary statistics from one snapshot file.
    """
    data_dir = data_dir or config.IMS_RAW_DIR

    exp_info = config.IMS_EXPERIMENTS[experiment]
    exp_folder = exp_info["folder"]
    exp_path = os.path.join(data_dir, exp_folder)

    if not os.path.isdir(exp_path):
        raise FileNotFoundError(
            f"Experiment {experiment} folder not found at {exp_path}. "
            f"Run download_ims() first."
        )

    # Get all data files, sorted by name (temporal order)
    files = sorted([f for f in os.listdir(exp_path)
                    if os.path.isfile(os.path.join(exp_path, f))
                    and not f.startswith('.')])

    if not files:
        raise FileNotFoundError(f"No data files found in {exp_path}")

    n_channels = exp_info["channels"]
    n_bearings = exp_info["bearings"]

    # Build channel names
    if n_channels == 8:
        # 2 channels per bearing (X + Y axis)
        channel_names = []
        for b in range(1, n_bearings + 1):
            channel_names.extend([f"bearing{b}_x", f"bearing{b}_y"])
    else:
        # 1 channel per bearing
        channel_names = [f"bearing{b}" for b in range(1, n_bearings + 1)]

    print(f"[IMS LOAD] Loading experiment {experiment}: "
          f"{len(files)} files, {n_channels} channels")

    # Load each file and extract raw signal data
    all_snapshots = []
    for file_idx, filename in enumerate(files):
        filepath = os.path.join(exp_path, filename)
        try:
            # IMS files are tab-delimited ASCII
            data = np.loadtxt(filepath, delimiter='\t')
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            # Ensure correct number of channels
            actual_channels = min(data.shape[1], n_channels)
            snapshot = {"file_index": file_idx, "filename": filename}

            for ch_idx in range(actual_channels):
                snapshot[channel_names[ch_idx]] = data[:, ch_idx]

            all_snapshots.append(snapshot)
        except Exception as e:
            if file_idx == 0:
                print(f"[IMS LOAD] Warning: Could not load {filename}: {e}")

    if not all_snapshots:
        raise RuntimeError(f"Could not load any files from {exp_path}")

    # Return raw snapshot data (dict with arrays) for feature extraction
    print(f"[IMS LOAD] Loaded {len(all_snapshots)} snapshots")
    return all_snapshots, channel_names, exp_info


if __name__ == "__main__":
    print("=" * 60)
    print("IMS Bearing Dataset Download & Loading")
    print("=" * 60)

    # Step 1: Download
    download_ims()

    # Step 2: Load and verify
    for exp in [1, 2, 3]:
        try:
            snapshots, ch_names, info = load_ims_experiment(exp)
            print(f"\nExperiment {exp}: {len(snapshots)} snapshots, "
                  f"channels: {ch_names}")
            print(f"  Failed bearings: {info['failed_bearings']}")
            print(f"  Failure modes: {info['failure_modes']}")
        except FileNotFoundError as e:
            print(f"\nExperiment {exp}: {e}")
