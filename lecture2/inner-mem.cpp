// $ g++ -g inner-mem.cpp -o inner-mem && valgrind --leak-check=full ./inner-mem
//
// Find bugs in this code
// - Out-of-bound array access
// - Uninitialized variables (--track-origins=yes)
// - Memory leaks (--leak-check=full)
// - Attach GDB (--vgdb-error=0)
//
// g++ -g inner-mem.cpp -o inner-mem && valgrind --tool=callgrind ./inner-mem 20000
// kcachegrind <output-file>
// qcachegrind <output-file> # on Mac
//
// g++ -g inner-mem.cpp -o inner-mem && valgrind --tool=cachegrind --cache-sim=yes ./inner-mem 20000
// cg_annotate --auto=yes <output-file>
//

#include <stdio.h>
#include <stdlib.h>

void init_arrays(double* x, double* y, long N) {
  for (long i = 0; i < N; i++) {
    x[i] = i+1;
    y[i] = 1./(i+1);
  }
}

double compute_dot_prod(double* x, double* y, long N, long repeat) {
  double dot_prod = 0;
  for (long k = 0; k < repeat; k++) {
    for (long i = 0; i <= N; i++) {
      dot_prod += x[i] * y[i];
    }
  }
  return dot_prod;
}

long get_vector_size(int argc, char** argv) {
  long N;
  if (argc > 1) N = atol(argv[1]);
  return N;
}

int main(int argc, char** argv) {
  long N = get_vector_size(argc, argv);

  // Allocate memory
  double* x = (double*) malloc(N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));

  // Initialize arrays
  init_arrays(x, y, N);

  // Compute dot-product
  double dot_prod = compute_dot_prod(x, y, N, 10);
  printf("dot-product = %f\n", dot_prod);

  return 0;
}

