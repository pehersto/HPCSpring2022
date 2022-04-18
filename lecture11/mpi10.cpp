#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

double inner(long n, double* x, double* y, MPI_Comm comm) {
  double local_prod = 0;
  for (long i = 0; i < n; i++) local_prod += x[i] * y[i];

  double prod = 0;
  MPI_Reduce(&local_prod, &prod, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  return prod;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int mpirank, mpisize;
  MPI_Comm_rank(comm, &mpirank);
  MPI_Comm_size(comm, &mpisize);

  long N = 100000000;
  long N_local = N / mpisize;
  long offset = N_local * mpirank;

  double* x = (double*)malloc(N_local * sizeof(double));
  double* y = (double*)malloc(N_local * sizeof(double));
  for (long i = 0; i < N_local; i++) {
    x[i] = 1.0 + offset + i;
    y[i] = 2.0 / (1.0 + offset + i);
  }

  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  double dot_prod = inner(N_local, x, y, comm);
  tt = MPI_Wtime() - tt;
  if (mpirank == 0) {
    printf("inner-product = %f\n", dot_prod);
    printf("time elapsed = %f\n", tt);

    printf("%f GB/s\n", 2 * N * sizeof(double) / 1e9 / tt);
    printf("%f Gflop/s\n", 2 * N / 1e9 / tt);
  }

  free(x);
  free(y);

  MPI_Finalize();
  return 0;
}

