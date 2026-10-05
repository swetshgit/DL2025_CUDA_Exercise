/*
 * temporal_features_cpu.cpp
 *
 * Two responsibilities:
 *   1. Declare the CUDA function(s) defined in temporal_features.cu
 *      so the linker can find them.
 *   2. Expose the Python-callable entry points via PYBIND11_MODULE.
 *
 * This file is compiled by the host C++ compiler (MSVC on Windows).
 * temporal_features.cu is compiled by nvcc.
 * setup.py links both object files into a single .pyd (Windows shared library).
 */

#include <torch/extension.h>   // includes pybind11 transitively
#include <vector>

// ---------------------------------------------------------------------------
// Forward-declare the raw CUDA launcher (defined in temporal_features.cu).
// ---------------------------------------------------------------------------
void launch_create_sequences_cuda(
    const float *input_ptr,
    float *X_ptr,
    const float *target_ptr,
    float *y_ptr,
    int T,
    int F,
    int seq_len,
    int N
);

// ---------------------------------------------------------------------------
// PyTorch C++ wrapper — validates tensors, allocates output, calls CUDA launcher
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> create_sequences_cuda(
    torch::Tensor data_tensor,
    torch::Tensor target_tensor,
    int seq_len
) {
    TORCH_CHECK(data_tensor.is_cuda(), "data_tensor must be on CUDA");
    TORCH_CHECK(target_tensor.is_cuda(), "target_tensor must be on CUDA");
    TORCH_CHECK(data_tensor.dtype() == torch::kFloat32, "data_tensor must be float32");
    TORCH_CHECK(target_tensor.dtype() == torch::kFloat32, "target_tensor must be float32");
    TORCH_CHECK(data_tensor.is_contiguous(), "data_tensor must be contiguous");
    TORCH_CHECK(target_tensor.is_contiguous(), "target_tensor must be contiguous");

    const int T = data_tensor.size(0);
    const int F = data_tensor.size(1);
    const int N = T - seq_len;

    TORCH_CHECK(N > 0, "seq_len >= T: no sequences can be formed");

    auto opts = data_tensor.options();
    torch::Tensor X = torch::empty({N, seq_len, F}, opts);
    torch::Tensor y = torch::empty({N}, opts);

    const float *input_ptr = data_tensor.data_ptr<float>();
    const float *target_ptr = target_tensor.data_ptr<float>();
    float *X_ptr = X.data_ptr<float>();
    float *y_ptr = y.data_ptr<float>();

    launch_create_sequences_cuda(input_ptr, X_ptr, target_ptr, y_ptr, T, F, seq_len, N);

    return {X, y};
}

// ---------------------------------------------------------------------------
// PYBIND11_MODULE registers the C++ functions so Python can call them.
//
// After building, you can do:
//   import temporal_features_ext
//   X, y = temporal_features_ext.create_sequences(data_t, target_t, seq_len)
//
// m.def("python_name", &cpp_function, "docstring")
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "create_sequences",
        &create_sequences_cuda,
        R"doc(
Sliding-window sequence creation on CUDA.

Args:
    data_tensor   (Tensor): Shape (T, F), float32, on CUDA.
                            The F feature columns for T timesteps.
    target_tensor (Tensor): Shape (T,),   float32, on CUDA.
                            The scalar target (CO2 kg/h) for each timestep.
    seq_len       (int):    Window length.

Returns:
    (X, y):
        X  — Tensor (N, seq_len, F) float32 — sliding windows of features.
        y  — Tensor (N,)            float32 — target for each window.
        where N = T - seq_len.
        )doc"
    );
}
