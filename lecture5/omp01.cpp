// OpenMP hello world
// $ gcc -fopenmp omp-hello.c && ./a.out
// # export OMP_NUM_THREADS = <number-of-threads>

#include <omp.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv) {
  printf("maximum number of threads = %d\n", omp_get_max_threads());

  #pragma omp parallel // num_threads(32)
  {
    printf("hello world from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
    //std::cout << "hello world from thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << std::endl;
  }

  return 0;
}
