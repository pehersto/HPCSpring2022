#include <stdio.h>
#include "utils.h"
#include <immintrin.h>

int main(int argc, char** argv)
{
  long n = read_option<long>("-n", argc, argv);
  Timer t;

  double* x = (double*) malloc(n * sizeof(double));
  double* y = (double*) malloc(n * sizeof(double));
  double z = 0;
  for (long i = 0; i < n; i++) {
    x[i] = (i+1)/10000.0;
    y[i] = (i-1)/10000.0;
  }

  t.tic();
  // #pragma omp simd reduction (+:z)
  for(long i = 0; i < n; ++i) {
    z = z + x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i];
  }
  printf("time: %8.2f s\n", t.toc());
  printf("z = %f\n", z);

  free(x);
  free(y);

  return 0;
}
