// Timing memory operations and calculating bandwidth
// $ g++ -std=c++11 -O3 02-memory.cpp && ./a.out -n 400000000 -repeat 1 -skip 1
// $ cat /proc/cpuinfo
// $ cat /proc/meminfo
// $ htop
// $ getconf -a | grep CACHE

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

int main(int argc, char** argv) {
  Timer t;
  long n = read_option<long>("-n", argc, argv);
  long repeat = read_option<long>("-repeat", argc, argv, "1");
  long skip = read_option<long>("-skip", argc, argv, "1");

  t.tic();
  double* x = (double*) malloc(n * sizeof(double)); // sizeof returns size in number of bytes
  printf("time to malloc = %f s\n", t.toc());

  t.tic();
  for (long i = 0; i < n; i += skip) x[i] = i;
  printf("time to initialize = %f s\n", t.toc());

  t.tic();
  for (long k = 0; k < repeat; k++) {
    for (long i = 0; i < n; i += skip) {
      x[i] = 2 * i;
    }
  }
  printf("time to write = %f s\n", t.toc());
  printf("bandwidth = %f GB/s\n", n * repeat * sizeof(double) / 1e9 / t.toc());

  double sum = 0;
  for (long i = 0; i < n; i += skip) sum += x[i];
  printf("sum = %f\n", sum);

  t.tic();
  free(x);
  printf("time to free = %f s\n", t.toc());

  return 0;
}
