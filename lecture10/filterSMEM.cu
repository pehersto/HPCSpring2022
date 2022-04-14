#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

struct RGBImage {
  long Xsize;
  long Ysize;
  float* A;
};
void read_image(const char* fname, RGBImage* I) {
  I->Xsize = 0;
  I->Ysize = 0;
  I->A = NULL;

  FILE* f = fopen(fname, "rb");
  if (f == NULL) return;
  fscanf(f, "P6\n%ld %ld\n255\n", &I->Ysize, &I->Xsize);
  long N = I->Xsize * I->Ysize;
  if (N) {
    I->A = (float*) malloc(3*N * sizeof(float));
    unsigned char* I0 = (unsigned char*) malloc(3*N * sizeof(unsigned char));
    fread(I0, sizeof(unsigned char), 3*N, f);
    for (long i0 = 0; i0 < N; i0++) {
      for (long i1 = 0; i1 < 3; i1++) {
        I->A[i1*N+i0] = I0[i0*3+i1];
      }
    }
    free(I0);
  }
  fclose(f);
}
void write_image(const char* fname, const RGBImage I) {
  long N = I.Xsize * I.Ysize;
  if (!N) return;

  FILE* f = fopen(fname, "wb");
  if (f == NULL) return;
  fprintf(f, "P6\n%ld %ld\n255\n", I.Ysize, I.Xsize);
  unsigned char* I0 = (unsigned char*) malloc(3*N * sizeof(unsigned char));
  for (long i0 = 0; i0 < 3; i0++) {
    for (long i1 = 0; i1 < N; i1++) {
      I0[i1*3+i0] = I.A[i0*N+i1];
    }
  }
  fwrite(I0, sizeof(unsigned char), 3*N, f);
  free(I0);
  fclose(f);
}
void free_image(RGBImage* I) {
  long N = I->Xsize * I->Ysize;
  if (N) free(I->A);
  I->A = NULL;
}


#define FWIDTH 7

// identity kernel
//float filter[FWIDTH][FWIDTH] = {
//  0,0,0,0,0,0,0,
//  0,0,0,0,0,0,0,
//  0,0,0,0,0,0,0,
//  0,0,0,1,0,0,0,
//  0,0,0,0,0,0,0,
//  0,0,0,0,0,0,0,
//  0,0,0,0,0,0,0,
//};

// 45 degree motion blur
float filter[FWIDTH][FWIDTH] = {
     0,      0,      0,      0,      0, 0.0145,      0,
     0,      0,      0,      0, 0.0376, 0.1283, 0.0145,
     0,      0,      0, 0.0376, 0.1283, 0.0376,      0,
     0,      0, 0.0376, 0.1283, 0.0376,      0,      0,
     0, 0.0376, 0.1283, 0.0376,      0,      0,      0,
0.0145, 0.1283, 0.0376,      0,      0,      0,      0,
     0, 0.0145,      0,      0,      0,      0,      0};

// mexican hat kernel
//float filter[FWIDTH][FWIDTH] = {
//  0, 0,-1,-1,-1, 0, 0,
//  0,-1,-3,-3,-3,-1, 0,
// -1,-3, 0, 7, 0,-3,-1,
// -1,-3, 7,24, 7,-3,-1,
// -1,-3, 0, 7, 0,-3,-1,
//  0,-1,-3,-3,-3,-1, 0,
//  0, 0,-1,-1,-1, 0, 0
//};


void CPU_convolution(float* I, const float* I0, long Xsize, long Ysize) {
  constexpr long FWIDTH_HALF = (FWIDTH-1)/2;
  long N = Xsize * Ysize;
  #pragma omp parallel for collapse(3) schedule(static)
  for (long k = 0; k < 3; k++) {
    for (long i0 = 0; i0 <= Xsize-FWIDTH; i0++) {
      for (long i1 = 0; i1 <= Ysize-FWIDTH; i1++) {
        float sum = 0;
        for (long j0 = 0; j0 < FWIDTH; j0++) {
          for (long j1 = 0; j1 < FWIDTH; j1++) {
            sum += I0[k*N + (i0+j0)*Ysize + (i1+j1)] * filter[j0][j1];
          }
        }
        I[k*N + (i0+FWIDTH_HALF)*Ysize + (i1+FWIDTH_HALF)] = (float)fabs(sum);
      }
    }
  }
}


#define BLOCK_DIM 32
__constant__ float filter_gpu[FWIDTH][FWIDTH];

__global__ void GPU_convolution_no_smem(float* I, const float* I0, long Xsize, long Ysize) {
  constexpr long FWIDTH_HALF = (FWIDTH-1)/2;
  long offset_x = blockIdx.x * (BLOCK_DIM-FWIDTH);
  long offset_y = blockIdx.y * (BLOCK_DIM-FWIDTH);

  float sum = 0;
  for (long j0 = 0; j0 < FWIDTH; j0++) {
    for (long j1 = 0; j1 < FWIDTH; j1++) {
      sum += I0[(offset_x + threadIdx.x + j0)*Ysize + (offset_y + threadIdx.y + j1)] * filter_gpu[j0][j1];
    }
  }

  if (threadIdx.x+FWIDTH < BLOCK_DIM && threadIdx.y+FWIDTH < BLOCK_DIM) // check if inside block (not halo)
    if (offset_x+threadIdx.x+FWIDTH <= Xsize && offset_y+threadIdx.y+FWIDTH <= Ysize) // check if inside picture
      I[(offset_x+threadIdx.x+FWIDTH_HALF)*Ysize + (offset_y+threadIdx.y+FWIDTH_HALF)] = (float)fabs(sum);
}

__global__ void GPU_convolution(float* I, const float* I0, long Xsize, long Ysize) {
  constexpr long FWIDTH_HALF = (FWIDTH-1)/2;
  __shared__ float smem[BLOCK_DIM+FWIDTH][BLOCK_DIM+FWIDTH];
  long offset_x = blockIdx.x * (BLOCK_DIM-FWIDTH);
  long offset_y = blockIdx.y * (BLOCK_DIM-FWIDTH);

  smem[threadIdx.x][threadIdx.y] = 0;
  if (offset_x + threadIdx.x < Xsize && offset_y + threadIdx.y < Ysize)
    smem[threadIdx.x][threadIdx.y] = I0[(offset_x + threadIdx.x)*Ysize + (offset_y + threadIdx.y)];
  __syncthreads();

  float sum = 0;
  for (long j0 = 0; j0 < FWIDTH; j0++) {
    for (long j1 = 0; j1 < FWIDTH; j1++) {
      sum += smem[threadIdx.x+j0][threadIdx.y+j1] * filter_gpu[j0][j1];
    }
  }

  if (threadIdx.x+FWIDTH < BLOCK_DIM && threadIdx.y+FWIDTH < BLOCK_DIM)
    if (offset_x+threadIdx.x+FWIDTH <= Xsize && offset_y+threadIdx.y+FWIDTH <= Ysize)
      I[(offset_x+threadIdx.x+FWIDTH_HALF)*Ysize + (offset_y+threadIdx.y+FWIDTH_HALF)] = (float)fabs(sum);
}


int main() {
  long repeat = 500;
  const char fname[] = "bike.ppm";

  // Load image from file
  RGBImage I0, I1, I1_ref;
  read_image(fname, &I0);
  read_image(fname, &I1);
  read_image(fname, &I1_ref);
  long Xsize = I0.Xsize;
  long Ysize = I0.Ysize;

  // Filter on CPU
  Timer t;
  t.tic();
  for (long i = 0; i < repeat; i++) CPU_convolution(I1_ref.A, I0.A, Xsize, Ysize);
  double tt = t.toc();
  printf("CPU time = %fs\n", tt);
  printf("CPU flops = %fGFlop/s\n", repeat * 2*(Xsize-FWIDTH)*(Ysize-FWIDTH)*FWIDTH*FWIDTH/tt*1e-9);

  // Allocate GPU memory
  float *I0gpu, *I1gpu;
  cudaMalloc(&I0gpu, 3*Xsize*Ysize*sizeof(float));
  cudaMalloc(&I1gpu, 3*Xsize*Ysize*sizeof(float));
  cudaMemcpy(I0gpu, I0.A, 3*Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(I1gpu, I1.A, 3*Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(filter_gpu, filter, sizeof(filter_gpu)); // Initialize filter_gpu

  // Create streams
  cudaStream_t streams[3];
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);
  cudaStreamCreate(&streams[2]);

  // Dry run
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim(Xsize/(BLOCK_DIM-FWIDTH)+1, Ysize/(BLOCK_DIM-FWIDTH)+1);
  GPU_convolution<<<gridDim,blockDim, 0, streams[0]>>>(I1gpu+0*Xsize*Ysize, I0gpu+0*Xsize*Ysize, Xsize, Ysize);
  GPU_convolution<<<gridDim,blockDim, 0, streams[1]>>>(I1gpu+1*Xsize*Ysize, I0gpu+1*Xsize*Ysize, Xsize, Ysize);
  GPU_convolution<<<gridDim,blockDim, 0, streams[2]>>>(I1gpu+2*Xsize*Ysize, I0gpu+2*Xsize*Ysize, Xsize, Ysize);

  // Filter on GPU
  cudaDeviceSynchronize();
  t.tic();
  for (long i = 0; i < repeat; i++) {
    GPU_convolution<<<gridDim,blockDim, 0, streams[0]>>>(I1gpu+0*Xsize*Ysize, I0gpu+0*Xsize*Ysize, Xsize, Ysize);
    GPU_convolution<<<gridDim,blockDim, 0, streams[1]>>>(I1gpu+1*Xsize*Ysize, I0gpu+1*Xsize*Ysize, Xsize, Ysize);
    GPU_convolution<<<gridDim,blockDim, 0, streams[2]>>>(I1gpu+2*Xsize*Ysize, I0gpu+2*Xsize*Ysize, Xsize, Ysize);
  }
  cudaDeviceSynchronize();
  tt = t.toc();
  printf("GPU time = %fs\n", tt);
  printf("GPU flops = %fGFlop/s\n", repeat * 2*(Xsize-FWIDTH)*(Ysize-FWIDTH)*FWIDTH*FWIDTH/tt*1e-9);

  // Print error
  float err = 0;
  cudaMemcpy(I1.A, I1gpu, 3*Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
  for (long i = 0; i < 3*Xsize*Ysize; i++) err = std::max(err, fabs(I1.A[i] - I1_ref.A[i]));
  printf("Error = %e\n", err);

  // Write output
  write_image("CPU.ppm", I1_ref);
  write_image("GPU.ppm", I1);

  // Free memory
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
  cudaStreamDestroy(streams[2]);
  cudaFree(I0gpu);
  cudaFree(I1gpu);
  free_image(&I0);
  free_image(&I1);
  free_image(&I1_ref);
  return 0;
}
