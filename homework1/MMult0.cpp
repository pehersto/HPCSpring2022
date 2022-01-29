// + Experiment with different optimization levels from -O0 to -O3 and report
//   the flop-rate and the bandwidth observed on your machine.
// + Specify the the compiler version (using the command: "g++ -v")
// + Try to find out the frequency, the maximum flop-rate and the maximum main
//   memory bandwidth for your processor.
// $ g++ -O3 -std=c++11 MMult0.cpp && ./a.out

#include <stdio.h>
#include "utils.h"

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0( long m, long n, long k, double *a,
                                     double *b,
                                     double *c) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < k; p++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

int main(int argc, char** argv) {
  const long NREPEATS = 50;
  const long PFIRST = 20;
  const long PLAST = 600;
  const long PINC = 20;

  printf(" Dimension       Time    Gflop/s       GB/s\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    double* a = (double*) malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c[i] = drand48();

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult0(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = 0; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth = 0; // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10d %10f %10f %10f\n", p, time, flops, bandwidth);

    free(a);
    free(b);
    free(c);
  }

  return 0;
}

