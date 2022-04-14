/* Simple send-receive example */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int message_out = 42;
    // msg, count, type, destination rank, tag, communicator
    MPI_Send(&message_out, 1, MPI_INT, 2, 999, MPI_COMM_WORLD);
  } else if (rank == 2) {
    int message_in;
    MPI_Status status;
    // msg, count, type, source rank, tag, communicator, status
    MPI_Recv(&message_in, 1, MPI_INT, 0, 999, MPI_COMM_WORLD, &status);

    printf("The message is %d. Status is %d %d %d\n", message_in, status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR);
  }

  MPI_Finalize();

  return 0;
}
