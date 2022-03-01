// $ gcc -O3 -fopenmp omp-vec-add.c && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv) {
  const long n = 1000;
  int nrThreads[5] = {1, 2, 4, 6, 8};

  double** A = (double**)malloc(sizeof(double*)*n);
  double** B = (double**)malloc(sizeof(double*)*n);
  double** C = (double**)malloc(sizeof(double*)*n);
  for(int i = 0; i < n; i++) {
    A[i] = (double*)malloc(sizeof(double)*n);
    B[i] = (double*)malloc(sizeof(double)*n);
    C[i] = (double*)malloc(sizeof(double)*n);
  }

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      A[i][j] = 1;
      B[i][j] = 2;
      C[i][j] = 0;
    }
  }

  double singleC = 1;
  for(int nrTIndex = 0; nrTIndex < 5; nrTIndex++) {
    int nrT = nrThreads[nrTIndex];
    double t = omp_get_wtime();
    #pragma omp parallel for num_threads(nrT)
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
        double cij = 0;
        for(int k = 0; k < n; k++) {
          cij = cij + A[i][k]*B[k][j];
        }
        C[i][j] = cij;
      }
    }
    t = omp_get_wtime() - t;
    if(nrT == 1)
      singleC = t;
    printf("%d: time elapsed = %f, speedup = %f\n", nrT, t, singleC/t);
    printf("%d: %f GB/s\n", nrT, 3*n*n*n*sizeof(double) / 1e9 / t);
    printf("%d: %f Gflops/s\n", nrT, 2*n*n*n / 1e9 / t);


  }

  for(int i = 0; i < n; i++) {
    free(A[i]);
    free(B[i]);
    free(C[i]);
  }
  free(A);
  free(B);
  free(C);

  return 0;
}
