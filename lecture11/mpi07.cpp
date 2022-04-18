#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int rank;
  MPI_Request request_out, request_in, request_out2, request_in2;;
  MPI_Status status, status2;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int message_out = 42;
    int message_in;

    MPI_Irecv(&message_in,  1, MPI_INT, 2, 999, MPI_COMM_WORLD, &request_in);
    MPI_Isend(&message_out, 1, MPI_INT, 2, 999, MPI_COMM_WORLD, &request_out);

    printf("Rank %d received %d\n", rank, message_in);

    MPI_Wait(&request_in, &status);
    MPI_Wait(&request_out, &status);
  } else if (rank == 2) {
    int message_out = 80;
    int message_in;

    MPI_Irecv(&message_in,  1, MPI_INT, 0, 999, MPI_COMM_WORLD, &request_in2);
    MPI_Isend(&message_out, 1, MPI_INT, 0, 999, MPI_COMM_WORLD, &request_out2);

    printf("Rank %d received %d\n", rank, message_in);
    MPI_Wait(&request_in2, &status2);
    MPI_Wait(&request_out2, &status2);
  }

  MPI_Finalize();

  return 0;
}

