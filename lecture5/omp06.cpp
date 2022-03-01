// $ gcc -O3 -fopenmp omp-vec-add.c && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv) {
  const long n = 100000000;
  int nrThreads[5] = {1, 2, 4, 6, 8};

  double* x = (double*) malloc(n * sizeof(double));
  double* y = (double*) malloc(n * sizeof(double));
  double* z = (double*) malloc(n * sizeof(double));
  for (long i = 0; i < n; i++) {
    x[i] = i+1;
    y[i] = 2.0 / (i+1);
    z[i] = 0;
  }

  double singleC = 1;
  for(int nrTIndex = 0; nrTIndex < 5; nrTIndex++) {
    int nrT = nrThreads[nrTIndex];
    double t = omp_get_wtime();
    #pragma omp parallel for num_threads(nrT) // schedule(static)
    for (long i = 0; i < n; i++) {
      z[i] = x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i]*x[i]*y[i];
    }
    t = omp_get_wtime() - t;
    if(nrT == 1)
      singleC = t;
    printf("%d: time elapsed = %f, speedup = %f\n", nrT, t, singleC/t);

  }

  free(x);
  free(y);
  free(z);

  return 0;
}
