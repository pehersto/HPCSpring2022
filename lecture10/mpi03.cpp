#include <stdio.h>
#include <mpi.h>

#define MLENGTH 8192 

int main(int argc, char *argv[]) {
  int rank;
  int messageA_out[MLENGTH];
  int messageA_in[MLENGTH];
  int messageB_out[MLENGTH];
  int messageB_in[MLENGTH];
  for(int i = 0; i < MLENGTH; i++) {
	messageA_out[i] = i;
	messageA_in[i] = 0;
	messageB_out[i] = i;
	messageB_in[i] = 0;
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    MPI_Status status;

    MPI_Send(&messageA_out, MLENGTH, MPI_INT, 2, 999, MPI_COMM_WORLD);
    MPI_Recv(&messageA_in,  MLENGTH, MPI_INT, 2, 999, MPI_COMM_WORLD, &status);

    printf("Rank %d received %d, ...\n", rank, messageA_in[0]);
  } else if (rank == 2) {
    MPI_Status status_2;

    MPI_Send(&messageB_out, MLENGTH, MPI_INT, 0, 999, MPI_COMM_WORLD);
    MPI_Recv(&messageB_in,  MLENGTH, MPI_INT, 0, 999, MPI_COMM_WORLD, &status_2);

    printf("Rank %d received %d, ...\n", rank, messageB_in[0]);
  }

  MPI_Finalize();

  return 0;
}
