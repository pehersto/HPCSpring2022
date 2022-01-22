#include <omp.h>
#include <stdio.h>
#include "utils.h"

int fib(int n) {
  int i, j;
  if(n < 2) {
    return n;
  } else {
    #pragma omp task shared(i) firstprivate(n)
    i = fib(n - 1);
    #pragma omp task shared(j) firstprivate(n)
    j = fib(n - 2);
    #pragma omp taskwait
    return i + j;
  }
}

int main(int argc, char** argv) {
  Timer t;
  long n = read_option<long>("-n", argc, argv);
  int res = 0;

  t.tic();
  #pragma omp parallel
  {
    #pragma omp single
    res = fib(n);
  }
  printf("Fibonacci number: %d\n", res);
  printf("time elapsed = %f\n", t.toc());

  return 0;
}
