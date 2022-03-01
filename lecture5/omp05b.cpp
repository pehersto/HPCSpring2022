#include <omp.h>
#include <stdio.h>
#include "utils.h"

int main(int argc, char** argv) {
  long n = read_option<long>("n", argc, argv, "25");
  long p = read_option<long>("p", argc, argv, "4");
  omp_set_num_threads(p); // set number of threads

  #pragma omp parallel
  {
     //#pragma omp for schedule(static)
     //#pragma omp for schedule(static, 2)
     //#pragma omp for schedule(dynamic, 3) // default chunksize is 1
     //#pragma omp for schedule(guided)
     for (long i = 0; i < n; i++) {
       printf("thread %d executing iteration %ld\n", omp_get_thread_num(), i);
     }
  }

  return 0;
}
