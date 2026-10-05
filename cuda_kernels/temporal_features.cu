/*
 * temporal_features.cu
 *
 * Custom CUDA kernel: sliding-window sequence creation for time-series data.
 *
 * WHAT IT DOES (conceptually):
 *   Given a 2D input tensor of shape (T, F) — T timesteps, F features —
 *   produce:
 *     X : (N, seq_len, F)   float32   — N sliding windows of the features
 *     y : (N,)              float32   — corresponding target values
 *   where N = T - seq_len, and each window X[i] = input[i : i+seq_len, :]
 *
 * PARALLELISATION STRATEGY:
 *   - Launch one CUDA thread per OUTPUT ELEMENT in X.
 *   - Total threads = N * seq_len * F
 *   - Each thread computes its (sample_idx, time_idx, feat_idx) from its
 *     global thread ID, then does a single memory copy from input → output.
 *   - A separate 1-D kernel handles the target vector y (one thread per
 * sample).
 *
 * BUILDING:
 *   python setup.py build_ext --inplace
 *
 * KEY CUDA CONCEPTS TO RECALL:
 *   threadIdx  — thread's position WITHIN its block (x, y, z)
 *   blockIdx   — which block this thread belongs to (x, y, z)
 *   blockDim   — how many threads per block (x, y, z)
 *   gridDim    — how many blocks in the grid (x, y, z)
 *
 *   Global thread index (1-D):
 *     int tid = blockIdx.x * blockDim.x + threadIdx.x;
 *
 * MEMORY ACCESS PATTERN (important for performance!):
 *   Coalesced access = adjacent threads read adjacent memory addresses.
 *   The layout (N, seq_len, F) in C-contiguous (row-major) order means
 *   the feature dimension F is the fastest-varying. Assigning the innermost
 *   loop dimension to threadIdx.x gives coalesced reads & writes.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Macro: check CUDA errors and throw a descriptive C++ exception.
// Use after every CUDA API call while learning — remove in production.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(                                                \
          std::string("CUDA error in " __FILE__ " line ") +                    \
          std::to_string(__LINE__) + ": " + cudaGetErrorString(err));          \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// KERNEL 1: create_sequences_kernel
//
// Each thread copies ONE element from the input into the output tensor X.
//
// Parameters
// ----------
// input      : (T, F) float32 — raw sensor readings, row-major
// output_X   : (N, seq_len, F) float32 — sliding windows, row-major
// T          : total number of timesteps
// F          : number of features
// seq_len    : length of each window
// N          : number of windows = T - seq_len
// ---------------------------------------------------------------------------
__global__ void create_sequences_kernel(
    const float *__restrict__ input, // [T * F] flat input
    float *__restrict__ output_X,    // [N * seq_len * F] flat output
    int T, int F, int seq_len, int N) {
  // -----------------------------------------------------------------------
  // TODO 1 — Compute the global thread ID.
  //
  //   Total work = N * seq_len * F elements.
  //   Map tid → (sample_idx, time_idx, feat_idx).
  //
  //   Hint:
  //     int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //
  //   Then decompose tid into 3D coordinates (this is just integer division
  //   in C-contiguous row-major order):
  //     feat_idx   = tid % F
  //     time_idx   = (tid / F) % seq_len
  //     sample_idx = tid / (F * seq_len)
  // -----------------------------------------------------------------------
  // int tid        = ...;
  // int feat_idx   = ...;
  // int time_idx   = ...;
  // int sample_idx = ...;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int feat_idx = tid % F;
  int time_idx = (tid / F) % seq_len;
  int sample_idx = tid / (F * seq_len);

  // -----------------------------------------------------------------------
  // TODO 2 — Guard against out-of-bounds threads.
  //
  //   We launch enough blocks to cover all N*seq_len*F elements, but the
  //   last block may have extra threads beyond the array end. Return early.
  //
  //   Hint:  if (tid >= N * seq_len * F) return;
  // -----------------------------------------------------------------------
  if (tid >= N * seq_len * F)
    return;

  // -----------------------------------------------------------------------
  // TODO 3 — Compute the source and destination flat indices.
  //
  //   In the INPUT tensor (T, F) row-major:
  //     input row  = sample_idx + time_idx   (window starts at sample_idx)
  //     input col  = feat_idx
  //     src_idx    = (sample_idx + time_idx) * F + feat_idx
  //
  //   In the OUTPUT tensor (N, seq_len, F) row-major:
  //     dst_idx = sample_idx * (seq_len * F) + time_idx * F + feat_idx
  //
  //   Then:
  //     output_X[dst_idx] = input[src_idx];
  // -----------------------------------------------------------------------
  // int src_idx = ...;
  // int dst_idx = ...;
  // output_X[dst_idx] = input[src_idx];
  int src_idx = (sample_idx + time_idx) * F + feat_idx;
  int dst_idx = sample_idx * (seq_len * F) + time_idx * F + feat_idx;
  output_X[dst_idx] = input[src_idx];
}

// ---------------------------------------------------------------------------
// KERNEL 2: create_targets_kernel
//
// One thread per window sample. Copies the target value (last row of window
// + 1, or whichever offset your label scheme uses).
//
// Parameters
// ----------
// input    : (T, F) float32 input — but target column is separate in Python;
//            here we accept a 1-D target vector of length T already extracted.
// output_y : (N,) float32 — target for each window
// seq_len  : offset into target
// N        : number of windows
// ---------------------------------------------------------------------------
__global__ void create_targets_kernel(
    const float *__restrict__ target_col, // [T] flat 1-D target values
    float *__restrict__ output_y,         // [N] flat output targets
    int seq_len, int N) {
  // -----------------------------------------------------------------------
  // TODO 4 — Compute the global thread ID (1-D kernel, simpler than above).
  //
  //   int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //   if (tid >= N) return;
  //
  //   The label for window `tid` is the value at position (tid + seq_len)
  //   in the original target column:
  //     output_y[tid] = target_col[tid + seq_len];
  // -----------------------------------------------------------------------
  // int tid = ...;
  // if (tid >= N) return;
  // output_y[tid] = target_col[...];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  output_y[tid] = target_col[tid + seq_len];
}

// ---------------------------------------------------------------------------
// BONUS KERNEL 3: normalize_minmax_kernel  (second exercise)
//
// In-place min-max normalisation over the feature dimension.
// Each thread normalises ONE element of X using precomputed per-feature
// min and max values.
//
// Parameters
// ----------
// data     : (N, seq_len, F) — in-place normalisation target
// feat_min : (F,) — minimum per feature
// feat_max : (F,) — maximum per feature
// total    : N * seq_len * F
// F        : number of features
// ---------------------------------------------------------------------------
__global__ void normalize_minmax_kernel(float *__restrict__ data,
                                        const float *__restrict__ feat_min,
                                        const float *__restrict__ feat_max,
                                        int total, int F) {
  // -----------------------------------------------------------------------
  // TODO 5 (bonus) — implement min-max normalisation.
  //
  //   int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  //   if (tid >= total) return;
  //
  //   int feat_idx = tid % F;    // which feature column this element belongs
  //   to float mn     = feat_min[feat_idx]; float mx     = feat_max[feat_idx];
  //   float range  = mx - mn;
  //
  //   Avoid division by zero when all values are the same:
  //   data[tid] = (range > 1e-8f) ? (data[tid] - mn) / range : 0.0f;
  // -----------------------------------------------------------------------
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;
  float f_min = feat_min[tid % F];
  float f_max = feat_max[tid % F];
  float range = f_max - f_min;
  if (range > 1e-8f)
    data[tid] = (data[tid] - f_min) / range;
  else
    data[tid] = 0.0f;
}

// ---------------------------------------------------------------------------
// Host-side launcher — called from temporal_features_cpu.cpp
//
// Launches both create_sequences_kernel and create_targets_kernel.
// ---------------------------------------------------------------------------
void launch_create_sequences_cuda(const float *input_ptr, float *X_ptr,
                                  const float *target_ptr, float *y_ptr, int T,
                                  int F, int seq_len, int N) {
  const int THREADS = 256;

  const int total_X = N * seq_len * F;
  const int blocks_X = (total_X + THREADS - 1) / THREADS;
  create_sequences_kernel<<<blocks_X, THREADS>>>(input_ptr, X_ptr, T, F,
                                                 seq_len, N);
  CUDA_CHECK(cudaGetLastError());

  const int blocks_y = (N + THREADS - 1) / THREADS;
  create_targets_kernel<<<blocks_y, THREADS>>>(target_ptr, y_ptr, seq_len, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
