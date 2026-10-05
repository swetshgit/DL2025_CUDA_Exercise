"""
temporal_features_wrapper.py

Python-level wrapper around the compiled CUDA extension.

Provides:
    create_sequences_cuda(df, target_col, features, seq_len, device)

This is a drop-in replacement for create_sequences() in data/init.py.
It handles:
  - Converting a pandas DataFrame to contiguous CUDA float32 tensors
  - Calling the compiled CUDA kernel
  - Returning (X, y) as CPU tensors for compatibility with the rest of
    the training pipeline (which moves them to GPU inside GRUModel)

Usage
-----
    from cuda_kernels.temporal_features_wrapper import create_sequences_cuda

    seq_x, seq_y = create_sequences_cuda(
        df,
        target='CO2(kg/h)',
        features=feature_list,   # from constants.py
        seq_len=64,
        device='cuda'
    )
"""

import time
import numpy as np
import pandas as pd
import torch

# ---- Try to import the compiled extension ----------------------------------
# If it hasn't been built yet, we fall back to the pure-Python implementation
# and print a clear message so you know what happened.
try:
    from . import temporal_features_ext as _ext
    _CUDA_AVAILABLE = True
except (ImportError, OSError):
    try:
        import temporal_features_ext as _ext
        _CUDA_AVAILABLE = True
    except (ImportError, OSError):
        _ext = None
        _CUDA_AVAILABLE = False

if not _CUDA_AVAILABLE:
    print(
        "[temporal_features_wrapper] WARNING: compiled extension not found.\n"
        "  Build it with:  cd cuda_kernels && python setup.py build_ext --inplace\n"
        "  Falling back to pure-Python create_sequences() for now."
    )

# ---- Pure-Python fallback (identical logic to data/init.py) ----------------
def _create_sequences_python(data: pd.DataFrame, target: str, features: list, seq_len: int):
    """CPU reference implementation — same behaviour as data/init.py."""
    X, y = [], []
    total = len(data)
    for i in range(seq_len, total):
        X.append(data[features].iloc[i - seq_len:i].values)
        y.append(data[target].iloc[i])
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32),
    )


# ---- Main public function --------------------------------------------------
def create_sequences_cuda(
    df: pd.DataFrame,
    target: str,
    features: list,
    seq_len: int = 64,
    device: str = "cuda",
    benchmark: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sliding-window sequence creation using a custom CUDA kernel.

    Parameters
    ----------
    df        : pandas DataFrame containing both feature and target columns.
    target    : column name of the target variable (e.g. 'CO2(kg/h)').
    features  : list of feature column names.
    seq_len   : sliding window length.
    device    : 'cuda' or 'cpu'.  Falls back to CPU if CUDA ext not built.
    benchmark : if True, also runs the Python version and prints a speedup.

    Returns
    -------
    (X, y):  CPU float32 tensors ready for torch.utils.data.TensorDataset.
        X shape: (N, seq_len, len(features))
        y shape: (N,)
    """
    # ---- Validate values before selecting either implementation ------------
    feat_np   = df[features].values.astype(np.float32)  # (T, F)
    target_np = df[target].values.astype(np.float32)    # (T,)

    if not np.isfinite(feat_np).all():
        invalid_columns = [
            column for index, column in enumerate(features)
            if not np.isfinite(feat_np[:, index]).all()
        ]
        raise ValueError(f"Non-finite sequence input features: {invalid_columns}")
    if not np.isfinite(target_np).all():
        raise ValueError(f"Non-finite sequence target values in: {target}")

    if not _CUDA_AVAILABLE or device == "cpu" or not torch.cuda.is_available():
        return _create_sequences_python(df, target, features, seq_len)

    data_cuda   = torch.from_numpy(feat_np).to(device).contiguous()
    target_cuda = torch.from_numpy(target_np).to(device).contiguous()

    # ---- Call the CUDA kernel (TODO: implement the kernel first!) -----------
    t0 = time.perf_counter()
    X_cuda, y_cuda = _ext.create_sequences(data_cuda, target_cuda, seq_len)
    torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t0

    # Move results back to CPU for compatibility with rest of pipeline
    X_cpu = X_cuda.cpu()
    y_cpu = y_cuda.cpu()

    if benchmark:
        t1 = time.perf_counter()
        X_ref, y_ref = _create_sequences_python(df, target, features, seq_len)
        t_cpu = time.perf_counter() - t1
        print(f"[benchmark]  CPU (Python): {t_cpu*1000:.1f} ms")
        print(f"[benchmark]  GPU (CUDA):   {t_gpu*1000:.1f} ms")
        print(f"[benchmark]  Speedup:      {t_cpu / t_gpu:.1f}x")

        # Sanity check: outputs should match within float precision
        if not torch.allclose(X_cpu, X_ref, atol=1e-5):
            print("[benchmark] WARNING: X mismatch between CUDA and CPU outputs!")
        if not torch.allclose(y_cpu, y_ref, atol=1e-5):
            print("[benchmark] WARNING: y mismatch between CUDA and CPU outputs!")

    return X_cpu, y_cpu
