// $ gcc -O3 -fopenmp omp-vec-add.c && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv) {
  long n = 100000000; // 100;

  double* x = (double*) malloc(n * sizeof(double));
  double* y = (double*) malloc(n * sizeof(double));
  double* z = (double*) malloc(n * sizeof(double));
  for (long i = 0; i < n; i++) {
    x[i] = i+1;
    y[i] = 2.0 / (i+1);
    z[i] = 0;
  }

  double t = omp_get_wtime();
  #pragma omp parallel for
  for (long i = 0; i < n; i++) {
    z[i] = x[i] + y[i];
  }
  t = omp_get_wtime() - t;
  printf("time elapsed = %f\n", t);
  printf("%f GB/s\n", 3*n*sizeof(double) / 1e9 / t);
  printf("%f Gflops/s\n", n / 1e9 / t);

  free(x);
  free(y);
  free(z);

  return 0;
}
