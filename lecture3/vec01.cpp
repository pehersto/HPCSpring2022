#include <stdio.h>
#include "utils.h"
#include <immintrin.h>
#include <math.h>

#define STRIDE 1

int main(int argc, char** argv)
{
  long n = read_option<long>("-n", argc, argv);
  Timer t;

  double* x = (double*) malloc(n * sizeof(double));
  double* y = (double*) malloc(n * sizeof(double));
  double* z = (double*) malloc(n * sizeof(double));
  for (long i = 0; i < n; i++) {
    x[i] = 1;
    y[i] = 1;
    z[i] = 1;
  }
  t.tic();
  for(int i = 0; i < n; i+=STRIDE) {
    z[i] = z[i] + x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i];
  }
  printf("time: %8.2f s\n", t.toc());

  free(x);
  free(y);
  free(z);

  return 0;
}
