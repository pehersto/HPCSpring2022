// $ gcc -O3 -fopenmp nbody.c && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int kernel_eval(int i) {
  printf("kernel: %d\n", i);
  #pragma omp critical
  {
    kernel_eval(i - 1);
  }
  return 0;
}

int main(int argc, char** argv) {
  long N = 80000;

  double t = omp_get_wtime();
  #pragma omp parallel num_threads(4)
  kernel_eval(N);
  printf("Evaluation time: %f seconds \n", omp_get_wtime()-t);

  return 0;
}
