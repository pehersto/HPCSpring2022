// OpenMP hello world
// $ gcc -fopenmp omp-hello.c && ./a.out
// # export OMP_NUM_THREADS = <number-of-threads>

#include <omp.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv) {
  double x[10];
  double s = 0.0;
  int n = 10;

  #pragma omp parallel // default(none) // shared(n, x, s)
  {
    double s2 = 0;
    for(int i = 0; i < n; ++i)
      s2 += x[i];
    s = s + s2;
  }

  return 0;
}
