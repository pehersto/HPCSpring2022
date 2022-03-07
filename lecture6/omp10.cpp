// $ gcc -O3 -fopenmp omp-reduction.c && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

double inner(long n, double* v, double* w) {
  double prod = 0;
  #pragma omp parallel for reduction(+:prod) // ordered
  for (long i = 0; i < n; i++) {
    prod += v[i] * w[i];
    // #pragma omp ordered
    printf("prod[%ld] = %f \n", i, prod);
  }
  return prod;
}

int main(int argc, char** argv) {
  long n = 20;

  double* x = (double*) malloc(n * sizeof(double));
  double* y = (double*) malloc(n * sizeof(double));
  for (long i = 0; i < n; i++) {
    x[i] = i+1;
    y[i] = 2.0 / (i+1);
  }

  double t = omp_get_wtime();
  double x_dot_y = inner(n, x, y);
  t = omp_get_wtime() - t;
  printf("inner-product = %.20f\n", x_dot_y);
  printf("time elapsed = %f\n", t);

  printf("%f GB/s\n", 2 * n * sizeof(double) / 1e9 / t);
  printf("%f Gflop/s\n", 2 * n / 1e9 / t);

  free(x);
  free(y);

  return 0;
}
