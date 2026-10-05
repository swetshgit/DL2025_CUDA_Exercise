#include <stdio.h>

__global__ void helloworld() {
  printf("Hello World from GPU thread %d\n", threadIdx.x);
}

int main() {
  printf("Hello World from CPU \n");
  helloworld<<<2, 3>>>();
  cudaDeviceSynchronize();
  return 0;
}
