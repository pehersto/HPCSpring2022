#include <stdio.h>

__global__ void kernel() {
	double a = 2.73; // register variable, automatic
	double c[100]; // local variable, automatic
	__shared__ double b; // shared variable
	int tx = threadIdx.x; // register variable

	if(tx == 0) {
		b = 3.1415;
	}
	//__syncthreads();
	printf("id = %d, a = %.5f, b = %.5f\n", tx, a, b);

}

int main() {

  // grid of 1 blocks, each block running 8 threads
  kernel<<<1, 8>>>();
  cudaDeviceSynchronize();

  return 0;
}
