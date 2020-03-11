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
#include <unistd.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <getopt.h>

#include "debug_util.h"
#include "randsvd_large.h"

#define ILOC2I(iloc,mb,myrow,nprows) mb*((iloc)/mb*nprows + prow) + (iloc) % mb
#define JLOC2J(jloc,nb,mycol,npcols) nb*((jloc)/nb*npcols + pcol) + (jloc) % nb
#define SWAP(a,b) {double temp = a; a = b; b = temp;}

double median (double *A, const size_t n) {
  size_t low = 0, high = n-1;
  size_t median = (low + high) / 2; // Take lower median if n is even.
  size_t lpoint, hpoint, midpoint;

  while (true) {
    // Return single element.
    if (high <= low)
      return A[median];
    // Return lower median of two elements.
    if (high == low + 1) {
      if (A[low] > A[high])
        SWAP(A[low], A[high]);
      return A[median];
    }
    // Swap A[low], A[midpoint], A[high] so that A[midpoint] <= A[low] <= A[high].
    midpoint = (low + high) / 2;
    if (A[midpoint] > A[high])
      SWAP(A[midpoint], A[high]);
    if (A[low] > A[high])
      SWAP(A[low], A[high]);
    if (A[midpoint] > A[low])
      SWAP(A[midpoint], A[low]);
    // Move low item from A[midpoint] to A[low+1].
    SWAP(A[midpoint], A[low+1]);
    // Converge towards the center of the interval [low,high] swapping if needed.
    lpoint = low + 1;
    hpoint = high;
    while (hpoint >= lpoint){
      do lpoint++; while (A[low] > A[lpoint]);
      do hpoint--; while (A[hpoint] > A[low]);
      if (hpoint >= lpoint)
        SWAP(A[lpoint], A[hpoint]);
    }
    // Move midpoint from A[low] to where it belongs.
    SWAP(A[low], A[hpoint]);
    // Recurr on the unbalanced partition.
    if (hpoint <= median)
      low = lpoint;
    if (hpoint >= median)
      high = hpoint - 1;
  }
  return A[median];
}

int main(int argc, char **argv) {

  // Parse input arguments.
  static int use_randsvd_int= false;
  static int compute_cond2_int = false;
  static int printmatrices_int = false;

  MKL_INT M = 100,  N = 100;
  MKL_INT nprows = 1, npcols = 1;
  size_t nreps = 5;
  double kappa = 1e4;
  static struct option long_opts[] = {
                                      {"full", no_argument, &use_randsvd_int, 1},
                                      {"new", no_argument, &use_randsvd_int, 0},
                                      {"debug", no_argument, &printmatrices_int, 1},
                                      {"cond", no_argument, &compute_cond2_int, 1},
                                      {"cond", required_argument, 0, 'k'},
                                      {"order", required_argument, 0, 'M'},
                                      {"nprows", required_argument, 0, 'm'},
                                      {"npcols", required_argument, 0, 'n'},
                                      {"nreps", required_argument, 0, 'p'},
                                      {0, 0, 0, 0}
  };
  int option_ind = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, "k:M:m:n:p:", long_opts, &option_ind)) != -1)
    {
      switch (opt) {
      case 0:
        break;
      case 'k':
        kappa = atof(optarg);
        assert (kappa>0);
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

  bool use_randsvd = (bool) use_randsvd_int;
  bool compute_cond2 = (bool) compute_cond2_int;
  bool printmatrices = (bool) printmatrices_int;

  size_t iloc, jloc, i, j;

  // Initialize BLACS.
  MKL_INT mympirank, nmpiprocs;
  MKL_INT ctxt, prow, pcol;
  MKL_INT info;
  MKL_INT MINUSONE = -1, ZERO=0;

  blacs_pinfo(&mympirank, &nmpiprocs);
  blacs_get(&MINUSONE, &ZERO, &ctxt);
  blacs_gridinit(&ctxt, "R", &nprows, &npcols);
  blacs_gridinfo(&ctxt, &nprows, &npcols, &prow, &pcol);

  // Generate matrix with orthogonal columns locally.
  MKL_INT mb = 2;
  MKL_INT nb = 2;
  MKL_INT mA = numroc(&M, &mb, &prow, &ZERO, &nprows);
  MKL_INT nA = numroc(&N, &nb, &pcol, &ZERO, &npcols);
  double *A = (double *)malloc(mA*nA*sizeof(double));
  double *S = (double *)malloc(mA*sizeof(double));
  double *B, *workcol2, *workcol3;
  MKL_INT *ipiv;
  double *workcol = (double *)malloc(mA*sizeof(double));
  double *workrow = (double *)malloc(nA*sizeof(double));
  MKL_INT descA[9];
  double cond2;
  descinit(descA, &M, &N, &mb, &nb, &ZERO, &ZERO, &ctxt, &mA, &info);

  if (compute_cond2) {
    B = (double *)malloc(mA*nA*sizeof(double));
    workcol2 = (double *)malloc(mA*sizeof(double));
    workcol3 = (double *)malloc(mA*sizeof(double));
    ipiv = (MKL_INT *)malloc((mA+nb)*sizeof(MKL_INT));
  }

  double t_start, t_end;
  double t_gencond_bwd, t_gencond_fwd,
    t_randsvd_bwd, t_randsvd_fwd, t_randsvd;
  double gt_gencond_bwd, gt_gencond_fwd,
    gt_randsvd_bwd, gt_randsvd_fwd, gt_randsvd;
  double gt_gencond_bwd_acc [nreps], gt_gencond_fwd_acc [nreps],
    gt_randsvd_bwd_acc [nreps], gt_randsvd_fwd_acc [nreps];
  double temp;
  size_t r;
  for (r=0; r<nreps; r++) {

    t_start = MPI_Wtime();
    gencond_bwd(A, descA, nA, mympirank, nprows, npcols, kappa, workrow, workcol);
    if (printmatrices)
      allocateandprint(A, descA, nA, mympirank, nprows, npcols, "gencond_bwd");
    t_end = MPI_Wtime();
    t_gencond_bwd = t_end - t_start;
    MPI_Reduce(&t_gencond_bwd, &temp, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    gt_gencond_bwd_acc[r] = temp;

    if (compute_cond2 && r == 0) {
      cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                            ipiv, workcol, workcol2, workcol3, B);
      if (mympirank == 0)
        printf("The 2-norm condition number should be %.15e\n", cond2 );
    }

    t_start = MPI_Wtime();
    gencond_fwd(A, descA, nA, mympirank, nprows, npcols, kappa, workrow, workcol);
    if (printmatrices)
      allocateandprint(A, descA, nA, mympirank, nprows, npcols, "gencond_fwd");
    t_end = MPI_Wtime();
    t_gencond_fwd = t_end - t_start;
    MPI_Reduce(&t_gencond_fwd, gt_gencond_fwd_acc+r, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (compute_cond2 && r == 0) {
      cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                            ipiv, workcol, workcol2, workcol3, B);
      if (mympirank == 0)
        printf("The 2-norm condition number should be %.15e\n", cond2 );
    }

    // Generate and distribute singular values for randsvd_bwd.
    t_start = MPI_Wtime();
    for (iloc=0; iloc<mA; iloc++) {
      i = ILOC2I(iloc,mb,prow,nprows);
      S[iloc] = pow(kappa, -((double)i)/(M-1.));
    }
    randsvd_bwd(A, descA, nA, S, mympirank, nprows, npcols,
                kappa, workrow, workcol);
    if (printmatrices)
      allocateandprint(A, descA, nA, mympirank, nprows, npcols, "randsvd_bwd");
    t_end = MPI_Wtime();
    t_randsvd_bwd = t_end - t_start;
    MPI_Reduce(&t_randsvd_bwd, gt_randsvd_bwd_acc+r, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (compute_cond2 && r == 0) {
      cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                            ipiv, workcol, workcol2, workcol3, B);
      if (mympirank == 0)
        printf("The 2-norm condition number should be %.15e\n", cond2 );
    }

    // Generate and distribute singular values for randsvd_fwd.
    free(S);
    t_start = MPI_Wtime();
    S = (double *)malloc(nA*sizeof(double));
    for (jloc=0; jloc<nA; jloc++) {
      j = JLOC2J(jloc,nb,pcol,npcols);
      S[jloc] = pow(kappa, -((double)j)/(M-1.));
    }
    randsvd_fwd(A, descA, nA, S, mympirank, nprows, npcols,
                kappa, workrow, workcol);
    if (printmatrices)
      allocateandprint(A, descA, nA, mympirank, nprows, npcols, "randsvd_fwd");
    t_end = MPI_Wtime();
    t_randsvd_fwd = t_end - t_start;
    MPI_Reduce(&t_randsvd_fwd, gt_randsvd_fwd_acc+r, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (compute_cond2 && r == 0) {
      cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                            ipiv, workcol, workcol2, workcol3, B);
      if (mympirank == 0)
        printf("The 2-norm condition number should be %.15e\n", cond2 );
    }
  }

  // Randsvd.

  /* printf("(%d,%d): M = %d, N = %d, mb =%d, nb = %d\n", prow, pcol, M, N, mb, nb); */
  MKL_INT mQl = numroc(&M, &mb, &prow, &ZERO, &nprows);
  MKL_INT nQl = numroc(&M, &mb, &pcol, &ZERO, &npcols);
  MKL_INT mQr = numroc(&N, &nb, &prow, &ZERO, &nprows);
  MKL_INT nQr = numroc(&N, &nb, &pcol, &ZERO, &npcols);
  MKL_INT mQ = mQl>mQr?mQl:mQr;
  MKL_INT nQ = nQl>nQr?nQl:nQr;
  double *Q = (double *)malloc(mQ*nQ*sizeof(double));

  t_start = MPI_Wtime();
  if (use_randsvd)
    randsvd(true, A, descA, nA, S, mympirank, nprows, npcols, Q, workrow);
  if (printmatrices)
    allocateandprint(A, descA, nA, mympirank, nprows, npcols, "randsvd");
  t_end = MPI_Wtime();
  t_randsvd = t_end - t_start;
  MPI_Reduce(&t_randsvd, &gt_randsvd, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (compute_cond2 && r == 0) {
    cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                          ipiv, workcol, workcol2, workcol3, B);
    if (mympirank == 0)
      printf("The 2-norm condition number should be %.15e\n", cond2 );
  }

  // Obtain timings.
  if (mympirank  == 0) {
    gt_gencond_bwd = median(gt_gencond_bwd_acc, nreps);
    gt_gencond_fwd = median(gt_gencond_fwd_acc, nreps);
    gt_randsvd_bwd = median(gt_randsvd_bwd_acc, nreps);
    gt_randsvd_fwd = median(gt_randsvd_fwd_acc, nreps);
  }

  if (mympirank == 0) {
    for (i = 0; i < nreps; ++i)
      printf("[%3ld] %e %e %e %e\n",
             i, gt_gencond_bwd_acc[i], gt_gencond_fwd_acc[i],
             gt_randsvd_bwd_acc[i], gt_randsvd_fwd_acc[i]);
    printf("[MED] %e %e %e %e\n",
           gt_gencond_bwd, gt_gencond_fwd,
           gt_randsvd_bwd, gt_randsvd_fwd);
  }


  if (mympirank == 0) {
    // Print to stdout.
    printf("%6lld %4lld %.5e %.5e %.5e %.5e %.5e\n",
           M, nprows * npcols,
           gt_randsvd, gt_randsvd_fwd, gt_randsvd_bwd,
           gt_gencond_fwd, gt_gencond_bwd);

    // Save to file.
    char outfilename [50];
    sprintf(outfilename, "./res_%07lld_%04lld%s.dat",
            M, nmpiprocs,use_randsvd?"_full":"_new");

    FILE *outfile = fopen(outfilename, "w");
    if (outfile != NULL) {
      if (use_randsvd)
        fprintf(outfile, "%6lld %4lld %.5e %.5e %.5e %.5e %.5e\n",
                M, nprows * npcols,
                gt_randsvd, gt_randsvd_fwd, gt_randsvd_bwd,
                gt_gencond_fwd, gt_gencond_bwd);
      else
        fprintf(outfile, "%6lld %4lld %.5e %.5e %.5e %.5e\n",
                M, nprows * npcols,
                gt_randsvd_fwd, gt_randsvd_bwd,
                gt_gencond_fwd, gt_gencond_bwd);
      fclose(outfile);
    }
  }

  free(workrow);
  free(workcol);
  free(A);
  free(S);
  free(Q);
  if (compute_cond2) {
    free(B);
    free(workcol2);
    free(workcol3);
    free(ipiv);
  }

  blacs_gridexit(&ctxt);
  MPI_Finalize();
  return 0;
}
