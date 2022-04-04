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
void vec_add_kernel(double* c, const double* a, const double* b, long N, long offset){
  int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
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
  //long N = (1UL<<25);

  const int blockSize = 1024, nStreams = 4;
  long N = 10000 * blockSize * nStreams;
  const int streamSize = N / nStreams;
  const int streamBytes = streamSize * sizeof(double);

  printf("N: %ld\n", N);
  printf("streamSize: %d\n", streamSize);

  double *x, *y, *z;
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&z, N * sizeof(double));
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

  double *x_d, *y_d, *z_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, N*sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  vec_add_kernel<<<N/1024,1024>>>(z_d, x_d, y_d, N, 0);
  cudaMemcpyAsync(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double err = 0;
  for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", err);

  cudaStream_t stream[nStreams];
  for (int i = 0; i < nStreams; ++i)
    cudaStreamCreate(&stream[i]);

  tt = omp_get_wtime();
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    cudaMemcpyAsync(&x_d[offset], &x[offset],
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]);
    cudaMemcpyAsync(&y_d[offset], &y[offset],
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]);
    vec_add_kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(z_d, x_d, y_d, N, offset);
    cudaMemcpyAsync(&z[offset], &z_d[offset],
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]);
  }
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  err = 0;
  for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", err);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);

  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(z);
  free(z_ref);

  return 0;
}

