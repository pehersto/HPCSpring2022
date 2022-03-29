#include <stdio.h>

__global__ void print_hello() {
  printf("hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
  //printf("hello from thread %d %d %d of block %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

int main() {

  // grid of 3 blocks, each block running 5 threads
  print_hello<<<3, 5>>>();
  cudaDeviceSynchronize();

  return 0;

  // grid of 4x3x2 blocks, each running 3x2 threads
  dim3 block ( 3 ,2) ;
  dim3 grid (4 , 3 , 2 ) ;
  print_hello<<<block, grid>>>();
  cudaDeviceSynchronize();
  

  return 0;
}
