// $ gcc -O3 -fopenmp nbody.c && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

void kernel_eval(long N, const double* x, const double* y, const double* z, const double* f, double* u) {
  for (long trg = 0; trg < N; trg++) { // loop over targets
    double tx = x[trg];
    double ty = y[trg];
    double tz = z[trg];
    double potential = 0;
    for (long src = 0; src < N; src++) { // loop over sourcer
      double dx = tx - x[src];
      double dy = ty - y[src];
      double dz = tz - z[src];
      double r2 = dx*dx + dy*dy + dz*dz;
      double rinv = (r2 > 0 ? 1/sqrt(r2) : 0);
      potential += rinv * f[src]; // sum potential
    }
    u[trg] = potential;
  }
}

int main(int argc, char** argv) {
  long N = 80000;

  double* x = (double*) malloc(N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));
  double* z = (double*) malloc(N * sizeof(double));
  double* f = (double*) malloc(N * sizeof(double));
  double* u = (double*) malloc(N * sizeof(double));

  for (long i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    f[i] = drand48();
    u[i] = 0;
  }

  double t = omp_get_wtime();
  kernel_eval(N, x, y, z, f, u);
  printf("Evaluation time: %f seconds \n", omp_get_wtime()-t);

  return 0;
}
