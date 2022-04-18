/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq) {
  int i;
  double tmp, gres = 0.0, lres = 0.0;

  for (i = 1; i <= lN; i++){
    tmp = ((2.0*lu[i] - lu[i-1] - lu[i+1]) * invhsq - 1);
    lres += tmp * tmp;
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]) {
  int mpirank, i, p, N, lN, iter, max_iters;
  MPI_Status status;
  MPI_Request request_out1, request_in1;
  MPI_Request request_out2, request_in2;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N / p;
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left and right ghost points */
  double * lu    = (double *) calloc(sizeof(double), lN + 2);
  double * lunew = (double *) calloc(sizeof(double), lN + 2);
  double * lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
    /* interleaf computation and communication: compute the first
     * and last value, which are communicated with non-blocking
     * send/recv. During that communication, do all the local work */

    /* Jacobi step for the left and right most points */
    lunew[1]  = 0.5 * (hsq + lu[0] + lu[2]);
    lunew[lN] = 0.5 * (hsq + lu[lN-1] + lu[lN+1]);

    if (mpirank < p - 1) {
      /* If not the last process, send/recv bdry values to the right */
      MPI_Irecv(&(lunew[lN+1]), 1, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &request_in1);
      MPI_Isend(&(lunew[lN]), 1, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD, &request_out1);
    }
    if (mpirank > 0) {
      /* If not the first process, send/recv bdry values to the left */
      MPI_Irecv(&(lunew[0]), 1, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &request_in2);
      MPI_Isend(&(lunew[1]), 1, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD, &request_out2);
    }

    /* Jacobi step for all the inner points */
    for (i = 2; i < lN; i++){
      lunew[i]  = 0.5 * (hsq + lu[i - 1] + lu[i + 1]);
    }

    /* check if Isend/Irecv are done */
    if (mpirank < p - 1) {
      MPI_Wait(&request_out1, &status);
      MPI_Wait(&request_in1, &status);
    }
    if (mpirank > 0) {
      MPI_Wait(&request_out2, &status);
      MPI_Wait(&request_in2, &status);
    }

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
