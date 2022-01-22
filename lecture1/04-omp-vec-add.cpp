// $ g++ -std=c++11 -O3 -fopenmp 04-omp-vec-add.cpp && ./a.out -n 100000000 -repeat 10

#include <omp.h>
#include <stdio.h>
#include "utils.h"

int main(int argc, char** argv) {
  Timer t;
  long n = read_option<long>("-n", argc, argv);
  long repeat = read_option<long>("-repeat", argc, argv, "1");

  double* x = (double*) malloc(n * sizeof(double));
  double* y = (double*) malloc(n * sizeof(double));
  double* z = (double*) malloc(n * sizeof(double));
  for (long i = 0; i < n; i++) {
    x[i] = i+1;
    y[i] = 2.0 / (i+1);
  }

  t.tic();
  for (long j = 0; j < repeat; j++) {
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
      z[i] = x[i] + y[i];
    }
  }
  printf("time elapsed = %f\n", t.toc());
  printf("%f GB/s\n", repeat * 3*n*sizeof(double) / 1e9 / t.toc());
  printf("%f Gflops/s\n", repeat * n / 1e9 / t.toc());

  free(x);
  free(y);
  free(z);

  return 0;
}
