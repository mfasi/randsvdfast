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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

#include "debug_util.h"
#include "randsvd_large.h"

#define ILOC2I(iloc,mb,myrow,nprows) mb*((iloc)/mb*nprows + prow) + (iloc) % mb
#define JLOC2J(jloc,nb,mycol,npcols) nb*((jloc)/nb*npcols + pcol) + (jloc) % nb

int main(int argc, char **argv) {

  // Initialize MPI.
  MPI_Init(&argc, &argv);

  // Parse input arguments.
  static int use_randsvd_int= false;
  static int compute_cond2_int = false;
  static int printmatrices_int = false;

  // Parse input arguments.
  MKL_INT M=100,
    N = 100,
    nprows = 1,
    npcols = 1;
  double kappa = 1e4;
  static struct
    option long_opts[] = {
                          {"full", no_argument, &use_randsvd_int, 1},
                          {"new", no_argument, &use_randsvd_int, 0},
                          {"debug", no_argument, &printmatrices_int, 1},
                          {"cond", no_argument, &compute_cond2_int, 1},
                          {"order", required_argument, 0, 'M'},
                          {"nprows", required_argument, 0, 'm'},
                          {"npcols", required_argument, 0, 'n'},
                          {"cond", required_argument, 0, 'k'},
                          {0, 0, 0, 0}
  };
  int option_ind = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, "M:m:n:k:", long_opts, &option_ind)) != -1)
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
      case 'k':
        kappa = atof(optarg);
        assert (kappa>0);
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

  /* printf("M = %6lld, nprows = %2lld, npcols = %2lld, kappa = %.5e, use_randsvd  = %d\n", */
  /*        M, nprows, npcols, kappa, use_randsvd); */

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
  double *B = (double *)malloc(mA*nA*sizeof(double));
  double *workcol = (double *)malloc(mA*sizeof(double));
  double *workcol2, *workcol3;
  double *workrow = (double *)malloc(nA*sizeof(double));;
  if (compute_cond2) {
    workcol2 = (double *)malloc(mA*sizeof(double));
    workcol3 = (double *)malloc(mA*sizeof(double));
  }

  MKL_INT *ipiv = (MKL_INT *)malloc((mA+mb)*sizeof(MKL_INT));
  MKL_INT descA[9];
  descinit(descA, &M, &N, &mb, &nb, &ZERO, &ZERO, &ctxt, &mA, &info);

  double cond2;
  double condinf_gencond_bwd, condinf_gencond_fwd;
  double condinf_randsvd_bwd, condinf_randsvd_fwd;
  double condinf_randsvd;

  gencond_bwd(A, descA, nA, mympirank, nprows, npcols, kappa, workrow, workcol);
  condinf_gencond_bwd = condest_norminf(A, descA, nA, mympirank, nprows, npcols,
                                        ipiv, B);
  if (printmatrices)
    allocateandprint(A, descA, nA, mympirank, nprows, npcols, "gencond_bwd");
  if (compute_cond2) {
    cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                          ipiv, workcol, workcol2, workcol3, B);
    if (mympirank == 0)
      printf("The 2-norm condition number should be %.15e\n", cond2 );
  }

  gencond_fwd(A, descA, nA, mympirank, nprows, npcols, kappa, workrow, workcol);
  condinf_gencond_fwd = condest_norminf(A, descA, nA, mympirank, nprows, npcols,
                                        ipiv, B);
  if (printmatrices)
    allocateandprint(A, descA, nA, mympirank, nprows, npcols, "gencond_fwd");
  if (compute_cond2) {
    cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                          ipiv, workcol, workcol2, workcol3, B);
    if (mympirank == 0)
      printf("The 2-norm condition number should be %.15e\n", cond2 );
  }

  // Generate and distribute singular values for randsvd_bwd.
  double *S = (double *)malloc(mA*sizeof(double));
  for (iloc=0; iloc<mA; iloc++) {
    i = ILOC2I(iloc,mb,prow,nprows);
    S[iloc] = pow(kappa, -((double)i)/(M-1.));
  }
  randsvd_bwd(A, descA, nA, S, mympirank, nprows, npcols,
              kappa, workrow, workcol);
  condinf_randsvd_bwd = condest_norminf(A, descA, nA, mympirank, nprows, npcols,
                                        ipiv, B);
  if (printmatrices)
    allocateandprint(A, descA, nA, mympirank, nprows, npcols, "randsvd_bwd");
  if (compute_cond2) {
    cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                          ipiv, workcol, workcol2, workcol3, B);
    if (mympirank == 0)
      printf("The 2-norm condition number should be %.15e\n", cond2 );
  }

  // Generate and distribute singular values for randsvd_fwd.
  free(S);
  S = (double *)malloc(nA*sizeof(double));
  for (jloc=0; jloc<nA; jloc++) {
    j = JLOC2J(jloc,nb,pcol,npcols);
    S[jloc] = pow(kappa, -((double)j)/(M-1.));
  }
  randsvd_fwd(A, descA, nA, S, mympirank, nprows, npcols,
              kappa, workrow, workcol);
  condinf_randsvd_fwd = condest_norminf(A, descA, nA, mympirank, nprows, npcols,
                                        ipiv, B);
  if (printmatrices)
    allocateandprint(A, descA, nA, mympirank, nprows, npcols, "randsvd_fwd");
  if (compute_cond2) {
    cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                          ipiv, workcol, workcol2, workcol3, B);
    if (mympirank == 0)
      printf("The 2-norm condition number should be %.15e\n", cond2 );
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

  if (use_randsvd)
    randsvd(true, A, descA, nA, S, mympirank, nprows, npcols, Q, workrow);
  condinf_randsvd = condest_norminf(A, descA, nA, mympirank, nprows, npcols,
                                    ipiv, B);

  if (printmatrices)
    allocateandprint(A, descA, nA, mympirank, nprows, npcols, "randsvd");
  if (compute_cond2) {
    cond2 = condest_norm2(A, descA, nA, mympirank, nprows, npcols,
                          ipiv, workcol, workcol2, workcol3, B);
    if (mympirank == 0)
      printf("The 2-norm condition number should be %.15e\n", cond2 );
  }

  if (mympirank == 0) {
    // Print to stdout.
    printf("%6lld %4lld %.5e %.5e %.5e %.5e %.5e\n",
           M, nprows * npcols,
           condinf_randsvd, condinf_randsvd_fwd, condinf_randsvd_bwd,
           condinf_gencond_fwd, condinf_gencond_bwd);

    // Save to file.
    char outfilename [50];
    sprintf(outfilename, "./cond_%07lld_%.2e%s.dat",
            M, kappa, use_randsvd?"_full":"_new");

    FILE *outfile = fopen(outfilename, "w");
    if (outfile != NULL) {
      if (use_randsvd)
        fprintf(outfile, "%6lld %.5e %.5e %.5e %.5e %.5e %.5e\n",
                M, kappa,
                condinf_randsvd, condinf_randsvd_fwd, condinf_randsvd_bwd,
                condinf_gencond_fwd, condinf_gencond_bwd);
      else
        fprintf(outfile, "%6lld %.5e %.5e %.5e %.5e %.5e\n",
                M, kappa,
                condinf_randsvd_fwd, condinf_randsvd_bwd,
                condinf_gencond_fwd, condinf_gencond_bwd);
      fclose(outfile);
    }
  }

  free(workrow);
  free(workcol);
  free(A);
  free(S);
  free(Q);

  free(B);
  if (compute_cond2) {
    free(workcol2);
    free(workcol3);
  }
  free(ipiv);


  blacs_gridexit(&ctxt);
  MPI_Finalize();
  return 0;
}
