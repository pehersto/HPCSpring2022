#include <stdio.h>

int main() {
  int nDevices;
  cudaGetDeviceCount(&nDevices); // get number of GPUs

  for (int i = 0; i < nDevices; i++) { // loop over all GPUs
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i); // get GPU properties
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Number of SM: %d\n\n", prop.multiProcessorCount);

    printf("  Total memory: %f GB\n", prop.totalGlobalMem*1.0e-9);
    printf("  Total const memory: %f KB\n", prop.totalConstMem/1024.0);
    printf("  Shared memory per block: %f KB\n", prop.sharedMemPerBlock/1024.0);

    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
  }
}
