"""
Dataset Download Module
========================
Downloads the NASA C-MAPSS turbofan engine degradation dataset.
Supports direct URL download (no authentication needed) with Kaggle API fallback.
"""

import os
import subprocess
import zipfile
import urllib.request
import pandas as pd
import numpy as np

import config



# Direct download URLs for public C-MAPSS mirrors (tried in order)
DIRECT_DOWNLOAD_URLS = [
    # NASA S3 official mirror (Turbofan Dataset 2 — includes CMAPSSData files)
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    # Kaggle dataset CDN (public, no auth needed for direct zip)
    "https://storage.googleapis.com/kaggle-data-sets/131118/598048/bundle/archive.zip",
]

KAGGLE_DATASET_ID = config.CMAPSS_DATASET


def download_cmapss(output_dir=None):
    """
    Download NASA C-MAPSS dataset.

    Tries direct URL download first (no auth needed).
    Falls back to Kaggle API if direct download fails.
    If all else fails, generates synthetic C-MAPSS-like data.

    Parameters
    ----------
    output_dir : str
        Directory to save downloaded files.
    """
    output_dir = output_dir or config.RAW_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Check if data already exists
    existing = [f for f in os.listdir(output_dir)
                if f.endswith(('.txt', '.csv')) and os.path.getsize(os.path.join(output_dir, f)) > 1000]
    if existing:
        print(f"[DOWNLOAD] Dataset already present ({len(existing)} files). Skipping download.")
        for f in sorted(existing):
            size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1e6
            print(f"  - {f} ({size_mb:.2f} MB)")
        return

    # ---- Method 1: Direct URL download (no auth) ----
    print("[DOWNLOAD] Attempting direct download (no authentication required)...")
    for url in DIRECT_DOWNLOAD_URLS:
        try:
            zip_path = os.path.join(output_dir, "CMAPSSData.zip")
            print(f"[DOWNLOAD] Downloading from: {url[:80]}...")

            # Download with progress
            def _report(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 / total_size)
                    print(f"\r[DOWNLOAD] Progress: {pct:.0f}% ({downloaded/1e6:.1f} MB)", end="", flush=True)

            urllib.request.urlretrieve(url, zip_path, reporthook=_report)
            print()  # newline after progress

            # Verify download is actually a zip
            if not zipfile.is_zipfile(zip_path):
                print(f"[DOWNLOAD] Downloaded file is not a valid ZIP. Skipping.")
                os.remove(zip_path)
                continue

            # Unzip
            print("[DOWNLOAD] Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(output_dir)
            os.remove(zip_path)

            # Move files from nested dirs to output_dir
            _flatten_extracted(output_dir)

            # Verify we got actual data files
            data_files = [f for f in os.listdir(output_dir)
                         if f.endswith('.txt') and os.path.getsize(os.path.join(output_dir, f)) > 1000]
            if data_files:
                print("[DOWNLOAD] Direct download successful!")
                _list_files(output_dir)
                return
            else:
                print("[DOWNLOAD] Downloaded archive had no valid data files.")
                continue

        except Exception as e:
            print(f"\n[DOWNLOAD] Direct download failed: {e}")
            # Clean up partial download
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue

    # ---- Method 2: Kaggle API (requires credentials) ----
    print("\n[DOWNLOAD] Trying Kaggle API...")
    try:
        _download_via_kaggle(output_dir)
        # Verify download succeeded
        data_files = [f for f in os.listdir(output_dir)
                     if f.endswith('.txt') and os.path.getsize(os.path.join(output_dir, f)) > 1000]
        if data_files:
            return
    except Exception as e:
        print(f"[DOWNLOAD] Kaggle API failed: {e}")

    # ---- Method 3: Generate minimal synthetic fallback ----
    print("\n[DOWNLOAD] All download methods failed. Generating synthetic C-MAPSS-like data...")
    print("[DOWNLOAD] (The pipeline will run with generated data for development/testing)")
    _generate_fallback_data(output_dir)


def _download_via_kaggle(output_dir):
    """Download via Kaggle API. Raises Exception if fails."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[DOWNLOAD] Kaggle package not installed. Skipping Kaggle API.")
        raise RuntimeError("kaggle package not available")

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as auth_err:
        raise RuntimeError(f"Kaggle authentication failed (no kaggle.json?): {auth_err}")

    print(f"[DOWNLOAD] Downloading {KAGGLE_DATASET_ID} via Kaggle API...")
    api.dataset_download_files(KAGGLE_DATASET_ID, path=output_dir, unzip=True)
    _flatten_extracted(output_dir)
    _list_files(output_dir)


def _flatten_extracted(output_dir):
    """
    Move files from nested subdirectories up to output_dir.
    Also extracts any nested .zip files found.
    """
    # First, extract any nested zip files
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.zip'):
                nested_zip = os.path.join(root, f)
                print(f"[DOWNLOAD] Extracting nested zip: {f}")
                try:
                    with zipfile.ZipFile(nested_zip, 'r') as zf:
                        zf.extractall(root)
                    os.remove(nested_zip)
                except zipfile.BadZipFile:
                    print(f"[DOWNLOAD] Warning: {f} is not a valid zip, skipping")

    # Move all files from subdirectories up to output_dir
    for root, dirs, files in os.walk(output_dir):
        if root == output_dir:
            continue
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(output_dir, f)
            if not os.path.exists(dst):
                os.rename(src, dst)

    # Clean up empty subdirectories
    for root, dirs, files in os.walk(output_dir, topdown=False):
        if root != output_dir:
            try:
                os.rmdir(root)
            except OSError:
                pass  # directory not empty, skip



def _generate_fallback_data(output_dir):
    """
    Generate synthetic C-MAPSS-like data as a fallback when download fails.
    This allows the pipeline to still run for development/testing.
    """
    np.random.seed(config.RANDOM_SEED)
    print("[DOWNLOAD] Generating synthetic fallback data (100 units)...")

    rows = []
    for unit_id in range(1, 101):
        n_cycles = np.random.randint(120, 360)
        for cycle in range(1, n_cycles + 1):
            row = [unit_id, cycle]
            # 3 operational settings
            row.extend([
                np.random.normal(0, 0.002),       # op_setting_1
                np.random.normal(0, 0.0003),      # op_setting_2
                100.0,                             # op_setting_3 (constant)
            ])
            # 21 sensors with degradation trend
            degradation = (cycle / n_cycles) ** 1.5
            for s in range(21):
                base = 50 + s * 10
                noise = np.random.normal(0, 0.5)
                trend = degradation * np.random.uniform(5, 20) * (1 if s % 2 == 0 else -1)
                row.append(base + trend + noise)
            rows.append(row)

    df = pd.DataFrame(rows)
    train_path = os.path.join(output_dir, "train_FD001.txt")
    df.to_csv(train_path, sep=" ", header=False, index=False)
    print(f"[DOWNLOAD] Generated: {train_path} ({len(df)} rows, 100 units)")

    # Generate test data (50 units, truncated)
    test_rows = []
    rul_values = []
    for unit_id in range(1, 51):
        n_cycles = np.random.randint(120, 360)
        cutoff = np.random.randint(50, n_cycles - 10)
        rul_values.append(n_cycles - cutoff)
        for cycle in range(1, cutoff + 1):
            row = [unit_id, cycle]
            row.extend([
                np.random.normal(0, 0.002),
                np.random.normal(0, 0.0003),
                100.0,
            ])
            degradation = (cycle / n_cycles) ** 1.5
            for s in range(21):
                base = 50 + s * 10
                noise = np.random.normal(0, 0.5)
                trend = degradation * np.random.uniform(5, 20) * (1 if s % 2 == 0 else -1)
                row.append(base + trend + noise)
            test_rows.append(row)

    df_test = pd.DataFrame(test_rows)
    test_path = os.path.join(output_dir, "test_FD001.txt")
    df_test.to_csv(test_path, sep=" ", header=False, index=False)

    rul_df = pd.DataFrame(rul_values)
    rul_path = os.path.join(output_dir, "RUL_FD001.txt")
    rul_df.to_csv(rul_path, sep=" ", header=False, index=False)

    print(f"[DOWNLOAD] Generated: {test_path} ({len(df_test)} rows, 50 units)")
    print(f"[DOWNLOAD] Generated: {rul_path} ({len(rul_values)} RUL labels)")


def _list_files(output_dir):
    """List downloaded files."""
    print("[DOWNLOAD] Available files:")
    for f in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / 1e6
            print(f"  - {f} ({size_mb:.2f} MB)")


def load_cmapss_train(subset="FD001", data_dir=None):
    """
    Load C-MAPSS training data.

    Returns
    -------
    pd.DataFrame
        Training data with named columns and computed RUL.
    """
    data_dir = data_dir or config.RAW_DATA_DIR

    # Try various common file naming patterns
    possible_names = [
        f"train_{subset}.txt",
        f"train_{subset}.csv",
        f"PM_train.txt",
    ]

    filepath = None
    for name in possible_names:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            filepath = candidate
            break

    # Search recursively if not found at top level
    if filepath is None:
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if "train" in f.lower() and subset.lower() in f.lower():
                    filepath = os.path.join(root, f)
                    break
                elif "pm_train" in f.lower():
                    filepath = os.path.join(root, f)
                    break
            if filepath:
                break

    if filepath is None:
        raise FileNotFoundError(
            f"Could not find training file for {subset} in {data_dir}. "
            f"Available files: {os.listdir(data_dir)}"
        )

    print(f"[DOWNLOAD] Loading training data from: {filepath}")

    df = pd.read_csv(filepath, sep=r"\s+", header=None)

    # Assign column names
    if len(df.columns) == len(config.CMAPSS_COLUMNS):
        df.columns = config.CMAPSS_COLUMNS
    elif len(df.columns) > len(config.CMAPSS_COLUMNS):
        # Some files have trailing whitespace columns
        df = df.iloc[:, :len(config.CMAPSS_COLUMNS)]
        df.columns = config.CMAPSS_COLUMNS
    else:
        cols = config.CMAPSS_COLUMNS[:len(df.columns)]
        df.columns = cols

    # Compute RUL (Remaining Useful Life) for each unit
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)

    # Cap RUL at MAX_RUL (piecewise linear degradation)
    df["RUL"] = df["RUL"].clip(upper=config.MAX_RUL)

    print(f"[DOWNLOAD] Loaded {len(df)} rows, {df['unit_id'].nunique()} units")
    print(f"[DOWNLOAD] Cycle range: {df['cycle'].min()} - {df['cycle'].max()}")
    print(f"[DOWNLOAD] RUL range: {df['RUL'].min()} - {df['RUL'].max()}")

    return df


def load_cmapss_all_subsets(subsets=None, data_dir=None):
    """
    Load and combine training data from multiple C-MAPSS subsets.

    Unit IDs are remapped to be globally unique across subsets.
    E.g., FD001 units 1-100, FD002 units 101-360, etc.

    Parameters
    ----------
    subsets : list[str]
        Subset identifiers (default: all from config).
    data_dir : str
        Data directory.

    Returns
    -------
    pd.DataFrame
        Combined training data with unique unit_ids and a 'subset' column.
    """
    subsets = subsets or config.CMAPSS_SUBSETS

    dfs = []
    unit_offset = 0

    for subset in subsets:
        try:
            df = load_cmapss_train(subset=subset, data_dir=data_dir)
            df["subset"] = subset

            # Remap unit_ids to be globally unique
            df["unit_id"] = df["unit_id"] + unit_offset
            unit_offset = df["unit_id"].max()

            dfs.append(df)
            print(f"[DOWNLOAD] {subset}: {df['unit_id'].nunique()} units, {len(df)} rows")
        except FileNotFoundError:
            print(f"[DOWNLOAD] WARNING: {subset} not found, skipping.")
            continue

    if not dfs:
        raise FileNotFoundError(
            f"No C-MAPSS subsets found. Tried: {subsets}. "
            f"Run download_cmapss() first."
        )

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[DOWNLOAD] Combined: {df_all['unit_id'].nunique()} total units, "
          f"{len(df_all)} total rows from {len(dfs)} subsets")

    return df_all


def load_cmapss_test(subset="FD001", data_dir=None):
    """
    Load C-MAPSS test data and RUL labels.

    Returns
    -------
    tuple(pd.DataFrame, pd.Series)
        Test data and true RUL values.
    """
    data_dir = data_dir or config.RAW_DATA_DIR

    # Find test file
    test_path = None
    rul_path = None

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            fl = f.lower()
            if ("test" in fl and subset.lower() in fl) or "pm_test" in fl:
                test_path = test_path or os.path.join(root, f)
            if ("rul" in fl and subset.lower() in fl) or "pm_truth" in fl:
                rul_path = rul_path or os.path.join(root, f)

    if test_path is None:
        raise FileNotFoundError(f"Could not find test file for {subset} in {data_dir}")

    print(f"[DOWNLOAD] Loading test data from: {test_path}")
    df_test = pd.read_csv(test_path, sep=r"\s+", header=None)

    if len(df_test.columns) == len(config.CMAPSS_COLUMNS):
        df_test.columns = config.CMAPSS_COLUMNS
    elif len(df_test.columns) > len(config.CMAPSS_COLUMNS):
        df_test = df_test.iloc[:, :len(config.CMAPSS_COLUMNS)]
        df_test.columns = config.CMAPSS_COLUMNS
    else:
        df_test.columns = config.CMAPSS_COLUMNS[:len(df_test.columns)]

    # Load RUL truth
    rul_true = None
    if rul_path:
        print(f"[DOWNLOAD] Loading RUL labels from: {rul_path}")
        rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None)
        rul_true = rul_true.iloc[:, 0]
        rul_true.name = "RUL_true"

    return df_test, rul_true


if __name__ == "__main__":
    print("=" * 60)
    print("NASA C-MAPSS Dataset Download & Loading")
    print("=" * 60)

    # Step 1: Download
    download_cmapss()

    # Step 2: Load and verify
    df_train = load_cmapss_train()
    print(f"\nTraining data shape: {df_train.shape}")
    print(df_train.head())

    df_test, rul_true = load_cmapss_test()
    print(f"\nTest data shape: {df_test.shape}")
    if rul_true is not None:
        print(f"RUL labels: {len(rul_true)} entries")
