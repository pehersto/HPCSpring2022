// $ gcc -O3 -fopenmp nbody.c && ./a.out

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <iostream>

int main(int argc, char** argv) {

  int a = 0;
  int b = 0;
  #pragma omp parallel sections shared(a,b) num_threads(2)
  {

    #pragma omp section
    {
      a = 1;
      std::cout << b;
    }
    #pragma omp section
    {
      b = 2;
      std::cout << a;
    }

  }
  printf("\n");

  return 0;
}
