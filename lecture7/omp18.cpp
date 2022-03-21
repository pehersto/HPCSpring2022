// numactl --hardware
// export OMP_PROC_BIND=close/spread

#include <omp.h>
#include <stdio.h>
#include "utils.h"

double reduce(long n, double* A) {
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < n; i++) sum += A[i];
  return sum;
}

int main(int argc, char** argv) {
  long n = read_option<long>("-n", argc, argv, "10000000000");
  double* A = (double*) malloc(n * sizeof(double));

  //#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; i++) A[i] = i+1;

  double t = omp_get_wtime();
  double sum = reduce(n, A);
  t = omp_get_wtime() - t;
  printf("sum = %f\n", sum);
  printf("time elapsed = %f\n", t);
  printf("%f GB/s\n", n * sizeof(double) / 1e9 / t);

  free(A);
  return 0;
}
