#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void vec_add(double* c, const double* a, const double* b, long N){
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__
void vec_add_kernel(double* c, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] + b[idx];
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  long N = (1UL<<25);

  double *x, *y, *z;
  cudaMallocManaged(&x, N * sizeof(double));
  cudaMallocManaged(&y, N * sizeof(double));
  cudaMallocManaged(&z, N * sizeof(double));
  double* z_ref = (double*) malloc(N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = i+2;
    y[i] = 1.0/(i+1);
    z[i] = 0;
    z_ref[i] = 0;
  }

  double tt = omp_get_wtime();
  vec_add(z_ref, x, y, N);
  printf("CPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  tt = omp_get_wtime();
  vec_add_kernel<<<N/1024,1024>>>(z, x, y, N);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double err = 0;
  for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", err);

  return 0;
}
