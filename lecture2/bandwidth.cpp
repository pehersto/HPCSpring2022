// Timing memory operations and calculating bandwidth
// $ g++ -std=c++11 -O3 -march=native bandwidth.cpp && ./a.out -n 400000000 -repeat 1 -skip 1

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

int main(int argc, char** argv) {
  Timer t;
  long n = read_option<long>("-n", argc, argv);
  long repeat = read_option<long>("-repeat", argc, argv, "1");
  long skip = read_option<long>("-skip", argc, argv, "1");
  
  t.tic();
  double* x = (double*) malloc(n * sizeof(double)); // dynamic allocation on heap
  //double x[400000000]; // static allocation on stack
  printf("time to malloc         = %f s\n", t.toc());

  t.tic();
  for (long i = 0; i < n; i += skip) x[i] = i;
  printf("time to initialize     = %f s\n", t.toc());

  t.tic();
  for (long k = 0; k < repeat; k++) {
    double kk = k;
    //#pragma omp parallel for schedule (static)
    for (long i = 0; i < n; i += skip) {
      x[i] = kk;
    }
  }
  printf("time to write          = %f s    ", t.toc());
  if (skip == 1) printf("bandwidth = %f GB/s\n", n * repeat * sizeof(double) / 1e9 / t.toc());
  else printf("\n");

  t.tic();
  for (long k = 0; k < repeat; k++) {
    double kk = x[0] * 0.5;
    //#pragma omp parallel for schedule (static)
    for (long i = 0; i < n; i += skip) {
      x[i] = x[i] * 0.5 + kk;
    }
  }
  printf("time to read + write   = %f s    ", t.toc());
  if (skip == 1) printf("bandwidth = %f GB/s\n", 2 * n * repeat * sizeof(double) / 1e9 / t.toc());
  else printf("\n");

  t.tic();
  double sum = 0;
  for (long k = 0; k < repeat; k++) {
    //#pragma omp parallel for schedule (static) reduction(+:sum)
    for (long i = 0; i < n; i += skip) {
      sum += x[i];
    }
  }
  printf("time to read           = %f s    ", t.toc());
  if (skip == 1) printf("bandwidth = %f GB/s\n", n * repeat * sizeof(double) / 1e9 / t.toc());
  else printf("\n");

  t.tic();
  free(x);
  printf("time to free           = %f s\n", t.toc());

  return sum;
}
