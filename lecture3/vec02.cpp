#include <stdio.h>
#include "utils.h"
#include <immintrin.h>
#include <math.h>

double g (double x, double y, double z);

double ginline(double x, double y, double z) {
  return z + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y + x*y;
}

int main(int argc, char** argv)
{
  //long n = 100000000;
  long n = read_option<long>("-n", argc, argv);
  Timer t;

  double* x = (double*) malloc(n * sizeof(double));
  double* y = (double*) malloc(n * sizeof(double));
  double* z = (double*) malloc(n * sizeof(double));
  for (long i = 0; i < n; i++) {
    x[i] = 0.001;
    y[i] = 0.001;
    z[i] = 0.001;
  }
  t.tic();
  for(int i = 0; i < n; ++i) {
    z[i] = g(x[i], y[i], z[i]);
  }
  printf("time g: %8.2f s\n", t.toc());
  t.tic();
  for(int i = 0; i < n; ++i) {
    z[i] = z[i] + x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i]+ x[i]*y[i];
  }
  printf("time: %8.2f s\n", t.toc());
  t.tic();
  for(int i = 0; i < n; ++i) {
    z[i] = ginline(x[i], y[i], z[i]);
  }
  printf("time inline: %8.2f s\n", t.toc());

  free(x);
  free(y);
  free(z);

  return 0;
}
