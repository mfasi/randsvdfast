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

#include "debug_util.h"

/* Esimate inf-norm condition number.  */
double condest_norminf(const double *A,
                       const MKL_INT* descA,
                       const MKL_INT nA,
                       const MKL_INT mympirank,
                       const MKL_INT nprows,
                       const MKL_INT npcols,
                       MKL_INT *ipiv,
                       double *B) {

  // Parse descriptor.
  MKL_INT ctxt = descA[1]; // BLACS context.
  MKL_INT M = descA[2];    // Rows of global matrix.
  MKL_INT N = descA[3];    // Columns of global matrix.
  /* MKL_INT Mb = descA[4];   // Blocking factor for rows. */
  /* MKL_INT mA = descA[8]; */
  MKL_INT ONE = 1;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  // Initialize normalized random vector.
  /* double *x = workcol1; */
  MKL_INT info;
  /* MKL_INT iloc, info; */
  /* mkl_INT descx [9]; */
  /* descinit(descx, &M, &ONE, &Mb, &ONE, &ZERO, &ZERO, &ctxt, &mA, &info); */
  /* if (pcol == 0) */
  /*   for (iloc=0; iloc < mA; iloc++) */
  /*     x[iloc] = rand() / (double)RAND_MAX; */
  /* double norm2; */
  /* pdnrm2(&M, &norm2, x, &ONE, &ONE, descx, &ONE); */
  /* norm2 = 1/norm2; */
  /* pdscal(&M, &norm2, x, &ONE, &ONE, descx, &ONE); */

  double anorm = pdlange("I", &M, &N,
                         A, &ONE, &ONE, descA, B);

  //Compute LU decoposition of A (with copy).
  pdgemr2d(&M, &N,
           A, &ONE, &ONE, descA,
           B, &ONE, &ONE, descA, &ctxt);
  pdgetrf(&M, &N,
          B, &ONE, &ONE, descA, ipiv, &info);

  /* allocateandprint(A, descA, nA, mympirank, nprows, npcols, "A"); */
  /* allocateandprint(B, descA, nA, mympirank, nprows, npcols, "B"); */

  double rcond;
  MKL_INT MINUSONE = -1;
  double lworkfloat;// This is in fact WORK!
  MKL_INT liwork;
  pdgecon("I", &M,
          B, &ONE, &ONE, descA,
          &anorm, &rcond,
          &lworkfloat, &MINUSONE, &liwork, &MINUSONE, &info);
  MKL_INT lwork = (MKL_INT)lworkfloat;
  double *work = (double *)malloc(lwork*sizeof(double));
  MKL_INT *iwork = (MKL_INT *)malloc(liwork*sizeof(MKL_INT));

  /* printf("lwork = %lld, liwork = %lld\n", lwork, liwork); */

  pdgecon("I", &M,
          B, &ONE, &ONE, descA,
          &anorm, &rcond,
          work, &lwork, iwork, &liwork, &info);

  /* printf("anorm = %.5e, condinf = %.5e\n", anorm, 1/rcond); */

  return 1/rcond;
}

/* Estimate 2-norm condition number. */
double condest_norm2(const double *A,
                     const MKL_INT* descA,
                     const MKL_INT nA,
                     const MKL_INT mympirank,
                     const MKL_INT nprows,
                     const MKL_INT npcols,
                     MKL_INT *ipiv,
                     double *workcol1,
                     double *workcol2,
                     double *workcol3,
                     double *B) {

  // Initialize tolerance parameter.
  double tolerance = 1e-12;

  // Parse descriptor.
  MKL_INT ctxt = descA[1]; // BLACS context.
  MKL_INT M = descA[2];    // Rows of global matrix.
  MKL_INT N = descA[3];    // Columns of global matrix.
  MKL_INT Mb = descA[4];   // Blocking factor for rows.
  /* MKL_INT Nb = descA[5];   // Blocking factor for columns. */
  MKL_INT mA = descA[8];
  MKL_INT ZERO = 0, ONE = 1;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  // Initialize normalized random vector.
  double *x = workcol1;
  MKL_INT descx [9];
  MKL_INT iloc, info;
  descinit(descx, &M, &ONE, &Mb, &ONE, &ZERO, &ZERO, &ctxt, &mA, &info);
  if (pcol == 0)
    for (iloc=0; iloc < mA; iloc++)
      x[iloc] = rand() / (double)RAND_MAX;
  double norm2;
  pdnrm2(&M, &norm2, x, &ONE, &ONE, descx, &ONE);
  norm2 = 1/norm2;
  pdscal(&M, &norm2, x, &ONE, &ONE, descx, &ONE);

  // Simple power iteration:
  // z <- A'x;
  // y <- Az;
  // s <- x'y
  // x <- y/norm(y)
  double *y = workcol2;
  double *z = workcol3;
  double alpha = 1;
  double beta = 0;
  double normA = 1;
  double normA_prev = INFINITY;
  MKL_INT counter = 0;
  /* printf("(%d,%d): normA = %.5e, normA_prev = %.5e\n", */
  /*        prow, pcol, normA, normA_prev); */
  /* printf("%.5e, %.5e\n", fabs(normA-normA_prev), tolerance); */
  while (fabs(normA-normA_prev) > tolerance && counter < 30) {
    pdgemv("T", &M, &N, &alpha,
           A, &ONE, &ONE, descA,
           x, &ONE, &ONE, descx, &ONE,
           &beta,
           z, &ONE, &ONE, descx, &ONE);
    pdgemv("N", &M, &N, &alpha,
           A, &ONE, &ONE, descA,
           z, &ONE, &ONE, descx, &ONE,
           &beta,
           y, &ONE, &ONE, descx, &ONE);
    normA_prev = normA;
    /* printf("(%d,%d): normA = %e, normA_prev = %e\n", prow, pcol, normA, normA_prev); */
    if (pcol == 0) {
      pddot(&M, &normA,
            x, &ONE, &ONE, descx, &ONE,
            y, &ONE, &ONE, descx, &ONE);
      dgebs2d(&ctxt, "Row", " ", &ONE, &ONE, &normA, &ONE);
      /* printf("(%d,%d): %.15e\n", prow, pcol, normA); */
      pdcopy(&M,
             y, &ONE, &ONE, descx, &ONE,
             x, &ONE, &ONE, descx, &ONE);
      pdnrm2(&M, &norm2, x, &ONE, &ONE, descx, &ONE);
      norm2 = 1/norm2;
      pdscal(&M, &norm2, x, &ONE, &ONE, descx, &ONE);
      /* printf("(%d,%d): %.15e\n", prow, pcol, normA); */
    } else {
      dgebr2d(&ctxt, "Row", " ", &ONE, &ONE, &normA, &ONE, &prow, &ZERO);
    }
    /* printf("(%d,%d): normA = %.5e, normA_prev = %.5e\n", */
    /*        prow, pcol, normA, normA_prev); */
    /* if (mympirank == 0) { */
    /*   printf("normA = %.15e\n", normA); */
    /* } */
    counter++;
  }
  normA = sqrt(normA);

  // Reinitialize normalized random vector.
  if (pcol == 0)
    for (iloc=0; iloc < mA; iloc++)
      x[iloc] = 1; //rand() / (double)RAND_MAX;
  pdnrm2(&M, &norm2, x, &ONE, &ONE, descx, &ONE);
  norm2 = 1/norm2;
  pdscal(&M, &norm2, x, &ONE, &ONE, descx, &ONE);
  pdcopy(&M,
         x, &ONE, &ONE, descx, &ONE,
         y, &ONE, &ONE, descx, &ONE);

  //Compute LU decoposition of A (with copy).
  pdgemr2d(&M, &N,
           A, &ONE, &ONE, descA,
           B, &ONE, &ONE, descA, &ctxt);
  /* MKL_INT *ipiv = (MKL_INT *)malloc((mA+Mb)*sizeof(MKL_INT)); */
  pdgetrf(&M, &N,
          B, &ONE, &ONE, descA, ipiv, &info);


  /* allocateandprint(x, descx, 1, mympirank, nprows, npcols, "x"); */
  /* alpha = 1; */
  /* beta = 0; */
  /* pdgemv("N", &M, &N, &alpha, */
  /*        A, &ONE, &ONE, descA, */
  /*        x, &ONE, &ONE, descx, &ONE, */
  /*        &beta, */
  /*        y, &ONE, &ONE, descx, &ONE); */
  /* allocateandprint(y, descx, 1, mympirank, nprows, npcols, "y"); */
  /* pdgetrs("N", &M, &ONE, */
  /*         B, &ONE, &ONE, descA, ipiv, */
  /*         y, &ONE, &ONE, descx, &info); */
  /* allocateandprint(y, descx, 1, mympirank, nprows, npcols, "y"); */

  // Simple inverse power iteration:
  // y <- A\y;
  // y <- A'\y;
  // s <- x'y
  // x <- y/norm(y)
  double normAinv = 1;
  double normAinv_prev = INFINITY;
  counter = 0;
  /* printf("(%d,%d): normAinv = %.5e, normAinv_prev = %.5e\n", */
  /*        prow, pcol, normAinv, normAinv_prev); */
  /* printf("%.5e, %.5e\n", fabs(normAinv-normAinv_prev), tolerance); */
  while (fabs(normAinv-normAinv_prev) > tolerance && counter < 30) {
    pdgetrs("N", &M, &ONE,
            B, &ONE, &ONE, descA, ipiv,
            y, &ONE, &ONE, descx, &info);
    pdgetrs("T", &M, &ONE,
            B, &ONE, &ONE, descA, ipiv,
            y, &ONE, &ONE, descx, &info);
    /* allocateandprint(y, descx, 1, mympirank, nprows, npcols, "y"); */
    normAinv_prev = normAinv;
    /* printf("(%d,%d): normAinv = %e, normAinv_prev = %e\n", prow, pcol, normAinv, normAinv_prev); */
    if (pcol == 0) {
      pddot(&M, &normAinv,
            x, &ONE, &ONE, descx, &ONE,
            y, &ONE, &ONE, descx, &ONE);
      dgebs2d(&ctxt, "Row", " ", &ONE, &ONE, &normAinv, &ONE);
      /* printf("(%d,%d): %.15e\n", prow, pcol, normAinv); */
      pdnrm2(&M, &norm2, y, &ONE, &ONE, descx, &ONE);
      norm2 = 1/norm2;
      pdscal(&M, &norm2, y, &ONE, &ONE, descx, &ONE);

      pdcopy(&M,
             y, &ONE, &ONE, descx, &ONE,
             x, &ONE, &ONE, descx, &ONE);
      /* printf("(%d,%d): %.15e\n", prow, pcol, normAinv); */
    } else {
      dgebr2d(&ctxt, "Row", " ", &ONE, &ONE, &normAinv, &ONE, &prow, &ZERO);
    }
    /* if (mympirank == 0) { */
    /*   printf("normAinv = %.5e\n", sqrt(normAinv) ); */
    /* } */
    /* printf("(%d,%d): normAinv = %.5e, normAinv_prev = %.5e\n", */
    /*        prow, pcol, normAinv, normAinv_prev); */
    counter++;
  }
  normAinv = sqrt(normAinv);
  /* if (mympirank == 0) */
  /*   printf("normA = %.5e, normAinv = %.5e, cond = %.5e\n", normA, normAinv, normA / normAinv); */
  return normA * normAinv;
}

/* Distribute a matrix from the root note (mympirank = 0) to the local
   submatrices. */
void distribute(double *Aglobal,
                double *Alocal,
                const MKL_INT* descA,
                const MKL_INT nA,
                const MKL_INT mympirank,
                const MKL_INT nprows,
                const MKL_INT npcols) {

  // Parse descriptor.
  MKL_INT ctxt = descA[1]; // BLACS context.
  MKL_INT M = descA[2];    // Rows of global matrix.
  MKL_INT N = descA[3];    // Columns of global matrix.
  MKL_INT Mb = descA[4];   // Blocking factor for rows.
  MKL_INT Nb = descA[5];   // Blocking factor for columns.
  MKL_INT mA = descA[8];

  MKL_INT ZERO = 0;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  // Distribute matrix.
  MKL_INT sendr = 0, sendc = 0, recvr = 0, recvc = 0, r, c;
  for (r = 0; r < M; r += Mb) {
    sendc = 0;
    MKL_INT nr = M-r>=Mb?Mb:M-r;     // Number of rows to be sent
    for (c = 0; c < N; c += Nb) {
      MKL_INT nc = N-c>=Nb?Nb:N-c;     // Number of cols to be sent
      if (mympirank == 0) { // Send small matrices around.
        dgesd2d(&ctxt, &nr, &nc, Aglobal+M*c+r, &M, &sendr, &sendc);
      }
      if (prow == sendr && pcol == sendc) { // Receive small matrices.
        dgerv2d(&ctxt, &nr, &nc, Alocal+mA*recvc+recvr, &mA, &ZERO, &ZERO);
        recvc = (recvc+nc)%nA;
      }
      sendc=(sendc+1)%npcols;
    }
    if (prow == sendr)
      recvr = (recvr+nr)%mA;
    sendr=(sendr+1)%nprows;
  }
}

/* Collect the local submatrices into a global matrix in the root note
   (mympirank = 0). */
void collect(double *Aglobal,
             double *Alocal,
             const MKL_INT* descA,
             const MKL_INT nA,
             const MKL_INT mympirank,
             const MKL_INT nprows,
             const MKL_INT npcols) {

  // Parse descriptor.
  MKL_INT ctxt = descA[1]; // BLACS context.
  MKL_INT M = descA[2];    // Rows of global matrix.
  MKL_INT N = descA[3];    // Columns of global matrix.
  MKL_INT Mb = descA[4];   // Blocking factor for rows.
  MKL_INT Nb = descA[5];   // Blocking factor for columns.
  MKL_INT mA = descA[8];
  MKL_INT ZERO = 0;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  // Collect matrix.
  MKL_INT sendr = 0, sendc = 0, recvr = 0, recvc = 0, r, c;
  for (r = 0; r < M; r += Mb) {
    sendc = 0;
    MKL_INT nr = M-r>=Mb?Mb:M-r;     // Number of rows to be sent
    for (c = 0; c < N; c += Nb) {
      MKL_INT nc = N-c>=Nb?Nb:N-c;     // Number of cols to be sent
      if (prow == sendr && pcol == sendc) { // Send small matrices.
        dgesd2d(&ctxt, &nr, &nc, Alocal+mA*recvc+recvr, &mA, &ZERO, &ZERO);
        recvc = (recvc+nc)%nA;
      }
      if (mympirank == 0) { // Receive small matrices around.
        dgerv2d(&ctxt, &nr, &nc, Aglobal+M*c+r, &M, &sendr, &sendc);
      }
      sendc=(sendc+1)%npcols;
    }
    if (prow == sendr)
      recvr = (recvr+nr)%mA;
    sendr=(sendr+1)%nprows;
  }
}

void printmatrix(const double* A,
                 const MKL_INT M,
                 const MKL_INT N) {
  MKL_INT i, j;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      printf("%12.5e ", A[j*M+i]);
    }
    printf("\n");
  }
  printf("\n");
}

void allocateandprint(double *A,
                      const MKL_INT *descA,
                      const MKL_INT nA,
                      const MKL_INT mympirank,
                      const MKL_INT nprows,
                      const MKL_INT npcols,
                      const char *title) {

  MKL_INT M = descA[2];    // Rows of global matrix.
  MKL_INT N = descA[3];    // Columns of global matrix.

  double *Aglobal = NULL;
  if (mympirank == 0)
    Aglobal = (double *) malloc(M*N*sizeof(double));
  collect(Aglobal, A, descA, nA,
          mympirank, nprows, npcols);
  if (mympirank == 0) {
    if (title != NULL)
      printf("%s:\n",title);
    printmatrix(Aglobal, M, N);
    free(Aglobal);
  }
}
