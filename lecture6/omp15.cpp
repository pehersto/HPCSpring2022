// $ gcc -O3 -fopenmp nbody.c && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>


int f(int i) {
  usleep(100);
  return i;
}

int main(int argc, char** argv) {
  long N = 80000;

  double* x = (double*)malloc(sizeof(double)*N);
  int* indx = (int*)malloc(sizeof(int)*N);

  for(long i = 0; i < N; i++) {
    indx[i] = rand() % N;
  }

  double t = omp_get_wtime();
  #pragma omp parallel for num_threads(4)
  for(long i = 0; i < N; i++) {
    #pragma omp critical
    x[indx[i]] = f(i);
  }
  printf("Evaluation time: %f seconds \n", omp_get_wtime()-t);

  t = omp_get_wtime();
  #pragma omp parallel for num_threads(4)
  for(long i = 0; i < N; i++) {
    #pragma omp atom write
    x[indx[i]] = f(i);
  }
  printf("Evaluation time: %f seconds \n", omp_get_wtime()-t);

  free(x);
  return 0;
}
