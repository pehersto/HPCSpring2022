#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

void myprint(long* A, long N, MPI_Comm comm) {
  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);

  for (int i = 0; i < np; i++) {
    MPI_Barrier(comm);
    if (rank == i) {
      printf("process %d ==> ", rank);
      for (long k = 0; k < N; k++) {
        printf("%4ld ", A[k]);
      }
      printf("\n");
    }
    MPI_Barrier(comm);
  }

}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);

  long* A = (long*) malloc(np*sizeof(long));
  for (long i = 0; i < np; i++) A[i] = rank*np + i;

  if (rank == 0) printf("Input matrix:\n");
  myprint(A, np, comm);

  long* B = (long*) malloc(np*sizeof(long));
  MPI_Alltoall(A, 1, MPI_LONG, B, 1, MPI_LONG, comm);

  if (rank == 0) printf("\n\nOutput matrix:\n");
  myprint(B, np, comm);

  free(A);
  free(B);

  MPI_Finalize();
  return 0;
}

