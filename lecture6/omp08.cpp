// $ gcc -O3 -fopenmp omp-vec-add.c && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv) {
  const long n = 1000;
  int nrT = 8;
  int totalRep = 1;

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

  // not nested
  double t = omp_get_wtime();
  for(int nrRep = 0; nrRep < totalRep; nrRep++) {
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
  }
  t = (omp_get_wtime() - t)/totalRep;
  printf("time elapsed = %f\n", t);

  //nested
  t = omp_get_wtime();
  for(int nrRep = 0; nrRep < totalRep; nrRep++) {
    #pragma omp parallel for num_threads(nrT/2)
    for(int i = 0; i < n; i++) {
      #pragma omp parallel for num_threads(nrT/2)
      for(int j = 0; j < n; j++) {
        double cij = 0;
        for(int k = 0; k < n; k++) {
          cij = cij + A[i][k]*B[k][j];
        }
        C[i][j] = cij;
      }
    }
  }
  t = (omp_get_wtime() - t)/totalRep;
  printf("time elapsed = %f\n", t);

  //collapse
  t = omp_get_wtime();
  for(int nrRep = 0; nrRep < totalRep; nrRep++) {
    #pragma omp parallel for num_threads(nrT) collapse(2)
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
        double cij = 0;
        for(int k = 0; k < n; k++) {
          cij = cij + A[i][k]*B[k][j];
        }
        C[i][j] = cij;
      }
    }
  }
  t = (omp_get_wtime() - t)/totalRep;
  printf("time elapsed = %f\n", t);

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
