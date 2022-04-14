#include <mpi.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
  int npes, myrank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  printf("From process %d out of %d, Hello World!\n",
          myrank, npes);
  long sum = 0;
  for(long i = 0; i < 1000000; i++) {
	  sum = sum + 1 + 1;
	  sleep(0.1);
  }
  MPI_Finalize();
  return 0;
}

