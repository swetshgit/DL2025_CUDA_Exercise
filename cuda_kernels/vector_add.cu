#include <stdio.h>

__global__ void vectorAdd_Single(int *A, int *B, int *C, int N) {
  int idx = threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void vectorAdd_Multi(int *A, int *B, int *C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

int main() {

  // vector add single thread
  /*
  int N = 5;
  int A[] = {1, 2, 3, 4, 5};
  int B[] = {10, 20, 30, 40, 50};
  int C[5];

  int *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, N * sizeof(int));
  cudaMalloc(&d_B, N * sizeof(int));
  cudaMalloc(&d_C, N * sizeof(int));

  cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);
  vectorAdd_Single<<<1, 5>>>(d_A, d_B, d_C, N);
  cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
      printf("%d ", C[i]);
  }
  printf("\n");
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  */
  // vector add multi thread
  int N = 1024;
  int num_workers = 32;
  int num_blocks = (N + num_workers - 1) / num_workers;
  int A[1024];
  int B[1024];
  int C[1024];
  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = i;
  }
  int *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, N * sizeof(int));
  cudaMalloc(&d_B, N * sizeof(int));
  cudaMalloc(&d_C, N * sizeof(int));
  cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);
  vectorAdd_Multi<<<num_workers, num_blocks>>>(d_A, d_B, d_C, N);
  cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    printf("%d ", C[i]);
  }
  printf("\n");
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}