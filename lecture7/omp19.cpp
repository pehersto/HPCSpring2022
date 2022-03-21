// module load intel-2018
// icpc -std=c++11 -O3 -march=native -fopenmp -mkl 01-MMultBLAS.cpp && ./a.out

// module load gcc-8.2
// g++ -std=c++11 -O3 -march=native -fopenmp 01-MMultBLAS.cpp -lblas && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

extern "C" {
  void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
}

void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  #pragma omp parallel for schedule(static)
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) { // Vectorized
  char TRANSA = 'N';
  char TRANSB = 'N';
  char TRANSC = 'N';
  int MM = m;
  int NN = n;
  int KK = k;
  double alpha = 1.0;
  double beta = 1.0;
  dgemm_(&TRANSA, &TRANSB, &MM, &NN, &KK, &alpha, a, &MM, b, &KK, &beta, c, &MM);
}


int main(int argc, char** argv) {
  const long PFIRST = 40;
  const long PLAST = 4000;
  const long PINC = 40;

  printf(" Dimension       Time    Gflop/s\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult0(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = 2.0*m*n*k*NREPEATS*1e-9/time;
    double bandwidth = 3.0*m*n*k*NREPEATS*sizeof(double)*1e-9/time;
    printf("%10ld %10f %10f", p, time, flops);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}
