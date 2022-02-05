#include <stdio.h>
#include "utils.h"
#include <immintrin.h>

double ginline(double x, double y, double z) {
  return z + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y;
}

void f(long n, double *x, double *y, double *z) {
  // #pragma GCC ivdep
  for(int i = 0; i < n; ++i) {
    z[i] = z[i] + x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i];
  }
}

int main(int argc, char** argv)
{
  long n = read_option<long>("-n", argc, argv);
  Timer t;

  double* x = (double*) malloc(n * sizeof(double));
  double* y = (double*) malloc(n * sizeof(double));
  double* z = (double*) malloc(n * sizeof(double));
  //double* z = x;
  for (long i = 0; i < n; i++) {
    x[i] = i+1;
    y[i] = i-1;
    z[i] = i;
  }

  t.tic();
  for(int i = 0; i < n; ++i) {
    z[i] = ginline(x[i], y[i], z[i]);
  }
  printf("time inline: %8.2f s\n", t.toc());

  t.tic();
  f(n, x, y, z);
  printf("time f: %8.2f s\n", t.toc());

  free(x);
  free(y);
  //free(z);

  return 0;
}
