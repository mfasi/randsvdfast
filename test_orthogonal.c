/*
 * Copyright (c) 2020 Massimiliano Fasi
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

#include "debug_util.h"
#include "randsvd_large.h"

// Generate distributed orthogonal symmetric matrix.
void orthog_unroll(double *Q,
                   const MKL_INT* descQ,
                   const MKL_INT nQ,
                   const MKL_INT mympirank,
                   const MKL_INT nprows,
                   const MKL_INT npcols) {

  // Parse descriptor.
  MKL_INT ctxt = descQ[1]; // BLACS context.
  MKL_INT M = descQ[2];    // Rows of global matrix.
  /* MKL_INT N = descQ[3];    // Rows of global matrix. */
  MKL_INT Mb = descQ[4];   // Blocking factor for rows.
  MKL_INT Nb = descQ[5];   // Blocking factor for columns.
  MKL_INT mQ = descQ[8];

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);
  double outconst = sqrt(2./(M+1));
  double inconst = M_PI / (M+1);
  MKL_INT iloc, jloc, i, j;
  jloc = 0;
  j = pcol*Nb+1;
  MKL_INT jdisp;
  while (jloc<nQ) {
    jdisp = jloc*mQ;
    i = prow*Mb+1;
    iloc = 0;
    while (iloc<mQ) {
      Q[jdisp+iloc] = outconst * sin(inconst*i*j);
      iloc++;
      i++;
      Q[jdisp+iloc] = outconst * sin(inconst*i*j);
      iloc++;
      i+=Mb*(nprows-1)+1;
    }
    jloc++;
    j++;
    i = prow*Mb+1;
    iloc = 0;
    while (iloc<mQ) {
      Q[jdisp+iloc] = outconst * sin(inconst*i*j);
      iloc++;
      i++;
      Q[jdisp+iloc] = outconst * sin(inconst*i*j);
      iloc++;
      i+=Mb*(nprows-1)+1;
    }
    jloc++;
    j+=Nb*(npcols-1)+1;
  }
}

/* Compare the wall-clock time of the algorithms for generating orthogonal
   matrices using two different techniques:
   1. Householder reflectors
   2. Symmetric eigenvector matrix for second difference matrix. [orthog(n,1)]*/
int main(int argc, char **argv) {

  // Initialize MPI.
  MPI_Init(&argc, &argv);

  // Parse input arguments.
  MKL_INT M=100,
    N = 100,
    nprows = 1,
    npcols = 1;
  size_t nreflectors = 4;
  size_t nreps = 5;
  static struct
    option long_opts[] = {
                          {"cond", required_argument, 0, 'k'},
                          {"order", required_argument, 0, 'M'},
                          {"nprows", required_argument, 0, 'm'},
                          {"npcols", required_argument, 0, 'n'},
                          {"nreps", required_argument, 0, 'p'},
                          {"reflectors", required_argument, 0, 'r'},
                          {0, 0, 0, 0}
  };
  int option_ind = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, "M:m:n:p:r:", long_opts, &option_ind)) != -1)
    {
      switch (opt) {
      case 0:
        break;
      case 'M':
        M = atoi(optarg);
        N = M;
        assert(M>0);
        break;
      case 'm':
        nprows = atoi(optarg);
        assert(nprows>0);
        break;
      case 'n':
        npcols = atoi(optarg);
        assert(npcols>0);
        break;
      case 'p':
        nreps = atoi(optarg);
        assert(nreps>0);
        break;
      case 'r':
        nreflectors = atoi(optarg);
        assert(nreflectors>0);
        break;
      case '?':
        printf("?\n");
        printf("Unrecognized option %c\n", optopt);
        break;
      case ':':
        printf(":\n");
        printf("Option %c requires an argument\n", optopt);
        break;
      default:
        abort();
      }
    }

  // Initialize BLACS.
  MKL_INT mympirank, nmpiprocs;
  MKL_INT ctxt, prow, pcol;
  MKL_INT info, MINUSONE = -1, ZERO = 0;

  blacs_pinfo(&mympirank, &nmpiprocs);
  blacs_get(&MINUSONE, &ZERO, &ctxt);
  blacs_gridinit(&ctxt, "R", &nprows, &npcols);
  blacs_gridinfo(&ctxt, &nprows, &npcols, &prow, &pcol);

  size_t k;
  MKL_INT mA, nA;
  MKL_INT mb = 2;
  MKL_INT nb = 2;

  double t_start, t_end;
  double t_orthog, t_orthog_householder,
    gt_orthog, gt_orthog_householder,
    gt_orthog_acc, *gt_orthog_householder_acc;

  gt_orthog_acc = 0;
  gt_orthog_householder_acc = (double *)calloc(nreflectors, sizeof(double));

  int r;

  // Generate matrix with orthogonal columns locally.
  mA = numroc(&M, &mb, &prow, &ZERO, &nprows);
  nA = numroc(&N, &nb, &pcol, &ZERO, &npcols);

  double *A = (double *)malloc(mA*nA*sizeof(double));
  if (A == NULL)
    fprintf(stderr, "[A] Value of errno: %d\n", errno);
  double *workcol = (double *)malloc(mA*sizeof(double));
  if (workcol == NULL)
    fprintf(stderr, "[workcol] Value of errno: %d\n", errno);
  double *workrow = (double *)malloc(nA*sizeof(double));
  if (workrow == NULL)
    fprintf(stderr, "[workrow] Value of errno: %d\n", errno);

  MKL_INT descA[9];
  descinit(descA, &M, &N, &mb, &nb, &ZERO, &ZERO, &ctxt, &mA, &info);

  // Warm-up lap.
  /* orthog(A, descA, nA, mympirank, nprows, npcols); */
  /* orthog_householder(A, descA, nA, mympirank, nprows, npcols, 1, workcol); */

  for (k = 0; k < nreps; k++) {
    t_start = MPI_Wtime();
    orthog_unroll(A, descA, nA, mympirank, nprows, npcols);
    t_end = MPI_Wtime();
    t_orthog = t_end - t_start;
    MPI_Reduce(&t_orthog, &gt_orthog, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    gt_orthog_acc += gt_orthog;

    for (r=0; r < nreflectors; r++) {
      t_start = MPI_Wtime();
      orthog_householder(A, descA, nA, mympirank, nprows, npcols, r+1, workcol);
      t_end = MPI_Wtime();
      t_orthog_householder = t_end - t_start;
      MPI_Reduce(&t_orthog_householder, &gt_orthog_householder, 1, MPI_DOUBLE, MPI_MAX, 0,
                 MPI_COMM_WORLD);
      gt_orthog_householder_acc[r] += gt_orthog_householder;
    }
  }

  if (mympirank ==0) {
    // Open output file.
    char outfilename [50];
    sprintf(outfilename, "./orthog_%07lld_%.04lld.dat", M, nmpiprocs);

    FILE *outfile = fopen(outfilename, "w");
    if (outfile != NULL) {
      printf("%7lld %.5e ",
             M, gt_orthog/nreps);
      for (r=0; r<nreflectors; r++) {
        printf("%.5e ", gt_orthog_householder_acc[r]/nreps);
      }
      printf("\n");

      fprintf(outfile, "%7lld %.5e ",
              M, gt_orthog/nreps);
      for (r=0; r<nreflectors; r++) {
        fprintf(outfile, "%.5e ", gt_orthog_householder_acc[r]/nreps);
      }
      fprintf(outfile, "\n");
      fclose(outfile);
    }
  }

  free(workrow);
  free(workcol);
  free(A);

  blacs_gridexit(&ctxt);
  MPI_Finalize();
  return 0;
}
