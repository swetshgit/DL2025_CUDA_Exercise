# CUDA Temporal Feature Extraction Extension

Custom CUDA kernels to accelerate preprocessing for the CO2 prediction pipeline.

## What This Replaces

The Python bottleneck in `data/init.py → create_sequences()`:

`python
# SLOW: pure Python loop — O(N * seq_len * features) on CPU
for i in range(sequence_length, total):
    X.append(data[features].iloc[i-sequence_length:i].values)
    y.append(data[target].iloc[i])
`

This kernel parallelizes the sliding-window copy entirely on the GPU.

## Files

| File                            | Role                                               |
|---------------------------------|----------------------------------------------------|
| temporal_features.cu            | **Your CUDA kernel — fill in the TODOs**           |
| temporal_features_cpu.cpp       | CPU fallback + PyTorch C++ binding declarations    |
| setup.py                        | Builds the extension with torch.utils.cpp_extension|
| temporal_features_wrapper.py    | Python API — drop-in replacement for create_sequences() |

## Build

`ash
cd cuda_kernels/
python setup.py build_ext --inplace
`

Verify the build worked:
`ash
python -c "import temporal_features_ext; print('Build OK')"
`

## Using in the Pipeline

In data/init.py, swap out create_sequences():

`python
# Before (CPU):
from data.init import create_sequences
seq_x, seq_y = create_sequences(df)

# After (GPU — after you implement the kernel):
from cuda_kernels.temporal_features_wrapper import create_sequences_cuda
seq_x, seq_y = create_sequences_cuda(df)
`

## Prerequisites

- CUDA toolkit (nvcc) matching your PyTorch CUDA version
  - Check: python -c "import torch; print(torch.version.cuda)"
  - Install toolkit: https://developer.nvidia.com/cuda-downloads
- PyTorch with CUDA support
- A C++17-capable compiler (MSVC on Windows)

## Learning Resources

- PyTorch Custom C++ and CUDA Extensions:
  https://pytorch.org/tutorials/advanced/cpp_extension.html
- CUDA C++ Programming Guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
