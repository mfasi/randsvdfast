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

#include "randsvd_large.h"

// Generate distributed orthogonal symmetric matrix.
void orthog(double *Q,
            const MKL_INT* descQ,
            const MKL_INT nQ,
            const MKL_INT mympirank,
            const MKL_INT nprows,
            const MKL_INT npcols) {

  // Parse descriptor.
  MKL_INT ctxt = descQ[1]; // BLACS context.
  MKL_INT M = descQ[2];    // Rows of global matrix.
  MKL_INT N = descQ[3];    // Rows of global matrix.
  MKL_INT Mb = descQ[4];   // Blocking factor for rows.
  MKL_INT Nb = descQ[5];   // Blocking factor for columns.
  MKL_INT mQ = descQ[8];

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);
  double outconst = sqrt(2./(M+1));
  double inconst = M_PI / (M+1);
  MKL_INT iloc, jloc, jdisp, i, j, k1, k2;
  jloc = 0;
  j = pcol*Nb+1;
  while (jloc<nQ){
    for (k1=0; k1<Nb && j<=N; k1++) {
      jdisp = jloc*mQ;
      i = prow*Mb+1;
      iloc = 0;
      while (iloc<mQ) {
        for (k2=0; k2<Mb && i<=M; k2++) {
          Q[jdisp+iloc] = outconst * sin(inconst*i*j);
          iloc++;
          i++;
        }
        i+=Mb*(nprows-1);
      }
      jloc++;
      j++;
    }
    j+=Nb*(npcols-1);
  }
}

void orthog_householder(double *Q,
                        const MKL_INT* descQ,
                        const MKL_INT nQ,
                        const MKL_INT mympirank,
                        const MKL_INT nprows,
                        const MKL_INT npcols,
                        const size_t r, // number of reflectors, min 1
                        double *workcol) {

  // Parse descriptor.
  MKL_INT ctxt = descQ[1]; // BLACS context.
  MKL_INT M = descQ[2];    // Rows of global matrix.
  MKL_INT N = descQ[3];    // Rows of global matrix.
  MKL_INT Mb = descQ[4];   // Blocking factor for rows.
  /* MKL_INT Nb = descQ[5];   // Blocking factor for columns. */
  MKL_INT mQ = descQ[8];
  MKL_INT ZERO = 0, ONE = 1;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  // Allocate identity matrix.
  double alpha = 0.;
  double beta = 1.;
  pdlaset("A", &M, &N, &alpha, &beta, Q, &ONE, &ONE, descQ);

  MKL_INT k, iloc;
  double *u = workcol;
  MKL_INT descu[9], info;
  descinit(descu, &M, &ONE, &Mb, &ONE, &ZERO, &ZERO, &ctxt, &mQ, &info);
  srand(890317*prow*pcol);
  for (k=0; k<r; k++) {
    // Generate and distribute random vector u.
    if (pcol == 0) {
      for (iloc = 0; iloc < mQ; iloc++)
        u[iloc] = rand() / (double)RAND_MAX;
    }

    // Compute and distribute alpha.
    if (pcol == 0) {
      pdnrm2(&M, &alpha, u, &ONE, &ONE, descu, &ONE);
      alpha = -2/pow(alpha,2);
      dgebs2d(&ctxt, "Row", " ", &ONE, &ONE, &alpha, &ONE);
    } else
      dgebr2d(&ctxt, "Row", " ", &ONE, &ONE, &alpha, &ONE, &prow, &ZERO);

    // Generate orthogonal matrix.
    pdger(&M, &N, &alpha,
          u, &ONE, &ONE, descu, &ONE,
          u, &ONE, &ONE, descu, &ONE,
          Q, &ONE, &ONE, descQ);
  }
}

// Generate matrix with pre-assigned 2-norm condition number.
void gencond_bwd(double *Q,
                 const MKL_INT* descQ,
                 const MKL_INT nQ,
                 const MKL_INT mympirank,
                 const MKL_INT nprows,
                 const MKL_INT npcols,
                 const double kappa,
                 double *workrow,
                 double *workcol) {

  // Parse descriptor.
  MKL_INT ctxt = descQ[1]; // BLACS context.
  MKL_INT M = descQ[2];    // Rows of global matrix.
  MKL_INT N = descQ[3];    // Columns of global matrix.
  MKL_INT Mb = descQ[4];   // Blocking factor for rows.
  MKL_INT Nb = descQ[5];   // Blocking factor for columns.
  MKL_INT mQ = descQ[8];
  MKL_INT ONE = 1;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  // Generate orthogonal matrix.
  orthog(Q, descQ, nQ, mympirank, nprows, npcols);

  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Q"); */

  double s1 = sqrt(kappa);
  double sm = 1/sqrt(kappa);

  // Generate elements of y corresponding to local columns.
  double outconst = sqrt(2./(M+1));
  double inconst = M_PI / (M+1);
  MKL_INT jstar = 1;  // Could use a random choice here.
  double q1jstar = outconst * sin(inconst*jstar);
  double qMjstar = outconst * sin(inconst*M*jstar);

  MKL_INT iloc, jloc, i, j;
  MKL_INT k1, k2;

  double *y = workrow;
  double alpha = -2.;
  jloc = 0;
  j = pcol*Nb+1;
  while (jloc<nQ){
    for (k1=0; k1<Nb && j<=N; k1++) {
      y[jloc] = alpha*(q1jstar * (outconst * sin(inconst*j)) * (s1-1) +
                       qMjstar * (outconst * sin(inconst*j*M)) * (sm-1) +
                       (double)(j == jstar));
      jloc++;
      j++;
    }
    j+=Nb*(npcols-1);
  }

  /* for (jloc=0; jloc<nQ; jloc++) { */
  /*   j = JLOC2J(jloc,Nb,pcol,npcols)+1; */
  /*   y[jloc] = alpha*(q1jstar * (outconst * sin(inconst*j)) * (s1-1) + */
  /*                    qMjstar * (outconst * sin(inconst*j*M)) * (sm-1) + */
/*                    (double)(j == jstar)); */
    /* } */

  double *q = workcol;
  i = prow*Mb+1;
  iloc = 0;
  while (iloc<mQ) {
    for (k2=0; k2<Mb && i<=M; k2++) {
      q[iloc] = outconst * sin(inconst*i);
      iloc++;
      i++;
    }
    i+=Mb*(nprows-1);
  }

  /* for (iloc=0; iloc<mQ; iloc++) { */
  /*   i = ILOC2I(iloc,Mb,prow,nprows)+1; */
  /*   q[iloc] = outconst * sin(inconst*i); */
  /* } */

  /* if (pcol == 0) */
  /*   for (iloc=0; iloc<mQ; iloc++) */
  /*     printf("(%d,%d): %5f\n", prow, pcol, q[iloc]); */

  /* MKL_INT *descrow [9], *desccol[9]; */
  /* descinit(descrow, &ONE, &N, &ONE, &Nb, &ZERO, &ZERO, &ctxt, &ONE, &info); */
  /* descinit(desccol, &M, &ONE, &Mb, &ONE, &ZERO, &ZERO, &ctxt, &mQ, &info); */
  /* allocateandprint(q, desccol, &ONE, mympirank, nprows, npcols, "q"); */
  /* allocateandprint(y, descrow, &nQ, mympirank, nprows, npcols, "y"); */

  /* if (prow == 0) */
  /*   for (j = 0; j < nQ; j++) */
  /*     Q[j*mQ] *= s1; */
  /* if (prow == ((M / Mb) % nprows)-1) */
  /*   for (j = 0; j < nQ; j++) */
  /*     Q[(j+1)*mQ-1] *= sm; */

  if (prow == 0)
    dscal(&nQ, &s1, Q, &mQ);
  MKL_INT nblocks = (MKL_INT)ceil(M / (double)Mb);
  MKL_INT frow = nblocks%nprows==0?nprows-1:nblocks%nprows-1;
  if (prow == frow)
    dscal(&nQ, &sm, Q+mQ-1, &mQ);

  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Qtilde"); */

  //  Rescale first and last row of Q.
  /* double *Q2 = (double *)malloc(mQ*nQ*sizeof(double)); */
  /* for (i = 0; i<nQ*mQ; i++) */
  /*   Q2[i] = Q[i]; */

  // Compute the matrix with pre-assigned singular values.
  /* for (jloc=0; jloc<nQ; jloc++) { */
  /*   for (iloc=0; iloc<mQ; iloc++) { */
  /*     /\* printf("(%d,%d): %.5e\n", prow, pcol, y[jloc] * q[iloc]); *\/ */
  /*     Q[jloc*mQ+iloc] += y[jloc] * q[iloc]; */
  /*   } */
  /* } */
  alpha = 1;
  dger(&mQ,&nQ,&alpha,q,&ONE,y,&ONE,Q,&mQ);

  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Q"); */
  /* allocateandprint(Q2, descQ, nQ, mympirank, nprows, npcols, "Q2"); */

}

// Generate matrix with pre-assigned 2-norm condition number.
void gencond_fwd(double *Q,
                 const MKL_INT* descQ,
                 const MKL_INT nQ,
                 const MKL_INT mympirank,
                 const MKL_INT nprows,
                 const MKL_INT npcols,
                 const double kappa,
                 double *workrow,
                 double *workcol) {

  // Parse descriptor.
  MKL_INT ctxt = descQ[1]; // BLACS context.
  MKL_INT M = descQ[2];    // Rows of global matrix.
  MKL_INT N = descQ[3];    // Columns of global matrix.
  MKL_INT Mb = descQ[4];   // Blocking factor for rows.
  MKL_INT Nb = descQ[5];   // Blocking factor for columns.
  MKL_INT mQ = descQ[8];
  MKL_INT ONE = 1;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  // Generate orthogonal matrix.
  orthog(Q, descQ, nQ, mympirank, nprows, npcols);

  double s1 = sqrt(kappa);
  double sm = 1/sqrt(kappa);

  // Generate elements of y corresponding to local columns.
  double outconst = sqrt(2./(M+1));
  double inconst = M_PI / (M+1);
  MKL_INT jstar = 1;  // Could use a random choice here.
  double q1jstar = outconst * sin(inconst*jstar);
  double qMjstar = outconst * sin(inconst*M*jstar);

  MKL_INT iloc, jloc, i, j, k;

  double *q = workrow;
  j = pcol*Nb+1;
  jloc = 0;
  while (jloc<nQ) {
    for (k=0; k<Nb && j<=N; k++) {
      q[jloc] = outconst * sin(inconst*j);
      jloc++;
      j++;
    }
    j+=Nb*(npcols-1);
  }

  /* for (jloc=0; jloc<nQ; jloc++) { */
  /*   j = JLOC2J(jloc,Nb,pcol,npcols)+1; */
  /*   q[jloc] = outconst * sin(inconst*j); */
  /* } */

  double *y = workcol;
  double alpha = -2.;
  iloc = 0;
  i = prow*Mb+1;
  while (iloc<mQ){
    for (k=0; k<Mb && i<=M; k++) {
      y[iloc] = alpha*(q1jstar * (outconst * sin(inconst*i)) * (s1-1) +
                       qMjstar * (outconst * sin(inconst*i*M)) * (sm-1) +
                       (double)(i == jstar));
      iloc++;
      i++;
    }
    i+=Mb*(nprows-1);
  }

  /* for (iloc=0; iloc<mQ; iloc++) { */
  /*   i = ILOC2I(iloc,Mb,prow,nprows)+1; */
  /*   y[iloc] = alpha*(q1jstar * (outconst * sin(inconst*i)) * (s1-1) + */
  /*                    qMjstar * (outconst * sin(inconst*i*M)) * (sm-1) + */
  /*                    (double)(i == jstar)); */
  /* } */

  if (pcol == 0)
    dscal(&mQ, &s1, Q, &ONE);
  MKL_INT nblocks = (MKL_INT)ceil(N / (double)Nb);
  MKL_INT fcol = nblocks%npcols==0?npcols-1:nblocks%npcols-1;
  if (pcol == fcol)
    dscal(&mQ, &sm, Q+(nQ-1)*mQ, &ONE);

  // Rescale first and last columns of Q.
  /* if (pcol == 0) */
  /*   for (i = 0; i < mQ; i++) */
  /*     Q[i] *= s1; */
  /* if (pcol == ((N / Nb) % npcols)-1) */
  /*   for (i = 0; i < mQ; i++) */
  /*     Q[mQ*(nQ-1)+i] *= sm; */

  /* double *Q2 = (double *)malloc(mQ*nQ*sizeof(double)); */
  /* for (i = 0; i<nQ*mQ; i++) */
  /*   Q2[i] = Q[i]; */

  // Compute the matrix with pre-assigned singular values.
  /* for (jloc=0; jloc<nQ; jloc++) { */
  /*   for (iloc=0; iloc<mQ; iloc++) { */
  /*     Q[jloc*mQ+iloc] += y[iloc] * q[jloc]; */
  /*   } */
  /* } */
  alpha = 1;
  dger(&mQ,&nQ,&alpha,y,&ONE,q,&ONE,Q,&mQ);

  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Q"); */
  /* allocateandprint(Q2, descQ, nQ, mympirank, nprows, npcols, "Q2"); */

}

/* Only square matrices for now. */
void randsvd_bwd(double *Q,
                 const MKL_INT* descQ,
                 const MKL_INT nQ,
                 const double* S,
                 const MKL_INT mympirank,
                 const MKL_INT nprows,
                 const MKL_INT npcols,
                 const double kappa,
                 double *workrow,
                 double *workcol) {

  // Parse descriptor.
  MKL_INT ctxt = descQ[1]; // BLACS context.
  MKL_INT M = descQ[2];    // Rows of global matrix.
  MKL_INT N = descQ[3];    // Columns of global matrix.
  MKL_INT Mb = descQ[4];   // Blocking factor for rows.
  MKL_INT Nb = descQ[5];   // Blocking factor for columns.
  MKL_INT mQ = descQ[8];
  MKL_INT ZERO = 0;
  MKL_INT ONE = 1;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  // Generate orthogonal matrix.
  orthog(Q, descQ, nQ, mympirank, nprows, npcols);

  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Q"); */

  // Scale rows of the matrix.
  MKL_INT iloc, jloc;
  for (jloc = 0; jloc < nQ; jloc++)
    for (iloc=0; iloc<mQ; iloc++)
      Q[jloc*mQ+iloc] *= S[iloc];

  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Qtilde"); */

  // Generate and distribute random vector u.
  double *u = workcol;
  MKL_INT descu[9], info;
  descinit(descu, &M, &ONE, &Mb, &ONE, &ZERO, &ZERO, &ctxt, &mQ, &info);
  if (pcol == 0) {
    srand(890317*prow*pcol);
    for (iloc = 0; iloc < mQ; iloc++)
      u[iloc] = 1; // rand() / (double)RAND_MAX;
    dgebs2d(&ctxt, "Row", " ", &ONE, &mQ, u, &ONE);
  } else
    dgebr2d(&ctxt, "Row", " ", &ONE, &mQ, u, &ONE, &prow, &ZERO);

  // Compute and distribute alpha.
  double alpha;
  if (pcol == 0) {
    pdnrm2(&M, &alpha, u, &ONE, &ONE, descu, &ONE);
    alpha = -2/pow(alpha,2);
    dgebs2d(&ctxt, "Row", " ", &ONE, &ONE, &alpha, &ONE);
  } else
    dgebr2d(&ctxt, "Row", " ", &ONE, &ONE, &alpha, &ONE, &prow, &ZERO);

  /* printf("alpha = %f\n\n", alpha ); */

  /* allocateandprint(u, descu, 1, mympirank, nprows, npcols, "u"); */

  // Compute and distribute y <- u*Q.
  double *y = workrow;
  double beta = 0.;
  MKL_INT descy [9];
  descinit(descy, &ONE, &N, &ONE, &Nb, &ZERO, &ZERO, &ctxt, &ONE, &info);
  pdgemm("T", "N", &ONE, &N, &M,
         &alpha,
         u,&ONE,&ONE,descu,
         Q,&ONE,&ONE,descQ,
         &beta,
         y,&ONE,&ONE,descy);
  if (prow == 0)
    dgebs2d(&ctxt, "Column", " ", &nQ, &ONE, y, &nQ);
  else
    dgebr2d(&ctxt, "Column", " ", &nQ, &ONE, y, &nQ, &ZERO, &pcol);

  /* allocateandprint(y, descy, nQ, mympirank, nprows, npcols, "y"); */

  // Compute output matrix.
  /* double *Q2 = (double *)malloc(mQ*nQ*sizeof(double)); */
  /* MKL_INT i; */
  /* for (i = 0; i<nQ*mQ; i++) */
  /*   Q2[i] = Q[i]; */

  /* double *Q2 = (double *)calloc(0, mQ*nQ*sizeof(double)); */

  alpha = 1;
  dger(&mQ,&nQ,&alpha,u,&ONE,y,&ONE,Q,&mQ);

  /* allocateandprint(Q2, descQ, nQ, mympirank, nprows, npcols, "Q2"); */

  /* for (jloc=0; jloc<nQ; jloc++) { */
  /*   for (iloc=0; iloc<mQ; iloc++) { */
  /*     Q[jloc*mQ+iloc] += u[iloc]*y[jloc]; */
  /*   } */
  /* } */
  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Q"); */
  /* allocateandprint(Q2, descQ, nQ, mympirank, nprows, npcols, "Q2"); */
}

/* Only square matrices for now. */
void randsvd_fwd(double *Q,
                 const MKL_INT* descQ,
                 const MKL_INT nQ,
                 const double* S,
                 const MKL_INT mympirank,
                 const MKL_INT nprows,
                 const MKL_INT npcols,
                 const double kappa,
                 double *workrow,
                 double *workcol) {

  // Parse descriptor.
  MKL_INT ctxt = descQ[1]; // BLACS context.
  MKL_INT M = descQ[2];    // Rows of global matrix.
  MKL_INT N = descQ[3];    // Columns of global matrix.
  MKL_INT Mb = descQ[4];   // Blocking factor for rows.
  MKL_INT Nb = descQ[5];   // Blocking factor for columns.
  MKL_INT mQ = descQ[8];
  MKL_INT ZERO = 0;
  MKL_INT ONE = 1;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  // Generate orthogonal matrix.
  orthog(Q, descQ, nQ, mympirank, nprows, npcols);

  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Q"); */

  // Scale columns of the matrix.
  MKL_INT iloc, jloc;
  for (jloc = 0; jloc < nQ; jloc++)
    for (iloc=0; iloc<mQ; iloc++)
      Q[jloc*mQ+iloc] *= S[jloc];

  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Qtilde"); */

  // Generate and distribute random vector u.
  double *u = workrow;
  MKL_INT descu[9], info;
  descinit(descu, &ONE, &N, &ONE, &Nb, &ZERO, &ZERO, &ctxt, &ONE, &info);
  if (prow == 0) {
    srand(920722*prow*pcol);
    for (jloc = 0; jloc < nQ; jloc++)
      u[jloc] = 1.; // rand() / (double)RAND_MAX;
    dgebs2d(&ctxt, "Column", " ", &ONE, &nQ, u, &ONE);
  } else
    dgebr2d(&ctxt, "Column", " ", &ONE, &nQ, u, &ONE, &ZERO, &pcol);

  /* allocateandprint(u, descu, nQ, mympirank, nprows, npcols, "u"); */

  // Compute and distribute alpha.
  double alpha;
  if (prow == 0) {
    pdnrm2(&N, &alpha, u, &ONE, &ONE, descu, &ONE);
    alpha = -2/pow(alpha,2);
    dgebs2d(&ctxt, "Column", " ", &ONE, &ONE, &alpha, &ONE);
  } else
    dgebr2d(&ctxt, "Column", " ", &ONE, &ONE, &alpha, &ONE, &ZERO, &pcol);

  /* printf("alpha = %.15e\n", alpha); */

  // Compute and distribute y <- Qu
  double *y = workcol;
  double beta = 0.;
  MKL_INT descy [9];
  descinit(descy, &M, &ONE, &Mb, &ONE, &ZERO, &ZERO, &ctxt, &mQ, &info);
  pdgemm("N", "T", &M, &ONE, &N,
         &alpha,
         Q,&ONE,&ONE,descQ,
         u,&ONE,&ONE,descu,
         &beta,
         y,&ONE,&ONE,descy);
  if (pcol == 0) {
    dgebs2d(&ctxt, "Row", " ", &mQ, &ONE, y, &mQ);
  } else
    dgebr2d(&ctxt, "Row", " ", &mQ, &ONE, y, &mQ, &prow, &ZERO);

  /* allocateandprint(y, descy, ONE, mympirank, nprows, npcols, "y"); */

  /* MKL_INT *descrow [9], *desccol[9]; */
  /* descinit(descrow, &ONE, &N, &ONE, &Nb, &ZERO, &ZERO, &ctxt, &ONE, &info); */
  /* descinit(desccol, &M, &ONE, &Mb, &ONE, &ZERO, &ZERO, &ctxt, &mQ, &info); */
  /* allocateandprint(y, descy, &ONE, mympirank, nprows, npcols, "y"); */
  /* allocateandprint(u, descrow, &nQ, mympirank, nprows, npcols, "u"); */
  /* double norm2; */
  /* pdnrm2(&N, &norm2, y, &ONE, &ONE, descy, &ONE); */
  /* printf("The norm of y is %.15e\n", norm2); */
  /* pdnrm2(&M, &norm2, u, &ONE, &ONE, descu, &ONE); */
  /* printf("The norm of u is %.15e\n", norm2); */

  // Compute output matrix.
  /* double *Q2 = (double *)malloc(mQ*nQ*sizeof(double)); */
  /* MKL_INT i; */
  /* for (i = 0; i<nQ*mQ; i++) */
  /*   Q2[i] = Q[i]; */
  alpha = 1;
  dger(&mQ,&nQ,&alpha,y,&ONE,u,&ONE,Q,&mQ);

  /* for (jloc=0; jloc<nQ; jloc++) { */
  /*   for (iloc=0; iloc<mQ; iloc++) { */
  /*     Q[jloc*mQ+iloc] += y[iloc]*u[jloc]; */
  /*     /\* printf("(%d,%d): %.15e\n", *\/ */
  /*     /\*        ILOC2I(iloc,Mb,prow,nprows), *\/ */
  /*     /\*        JLOC2J(jloc,Nb,pcol,npcols), *\/ */
  /*     /\*        y[iloc]*u[jloc]); *\/ */
  /*   } */
  /* } */
  /* allocateandprint(Q, descQ, nQ, mympirank, nprows, npcols, "Q"); */
  /* allocateandprint(Q2, descQ, nQ, mympirank, nprows, npcols, "Q2"); */
}

/* C rewriting of the pdlagge fortran function in TESTING/EIG/pdlagge.f. */
void randsvd(const bool init,
             double *A,
             const MKL_INT *descA,
             const MKL_INT nA,
             const double *S,
             const MKL_INT mympirank,
             const MKL_INT nprows,
             const MKL_INT npcols,
             double *Q,
             double *workrow) {

  // Parse descriptor.
  MKL_INT ctxt = descA[1]; // BLACS context.
  MKL_INT M = descA[2];    // Rows of global matrix.
  MKL_INT N = descA[3];    // Rows of global matrix.
  MKL_INT Mb = descA[4];   // Blocking factor for rows.
  MKL_INT Nb = descA[5];   // Blocking factor for columns.
  MKL_INT mA = descA[8];
  MKL_INT ZERO = 0;
  MKL_INT ONE = 1;
  MKL_INT info;

  MKL_INT prow, pcol;
  blacs_pcoord(&ctxt, &mympirank, &prow, &pcol);

  MKL_INT iloc, jloc;
  // Initialize A to identity matrix, if required.
  if (init) {
    double alpha = 0.;
    double beta = 1.;
    pdlaset("A", &M, &N, &alpha, &beta, A, &ONE, &ONE, descA);

    // Scale the elements of the diagonal matrix.
    MKL_INT iloc2;
    iloc = 0;
    jloc = 0;
    bool founddiag = false;
    while(jloc<nA) {
      founddiag = false;
      for (iloc2=iloc; iloc2<mA && !founddiag; iloc2++)
        if (A[jloc*mA+iloc]!=0.)
          founddiag = true;
      while (jloc<nA && iloc<mA && A[jloc*mA+iloc]!=0.) {
        A[jloc*mA+iloc] *= S[jloc];
        iloc++;
        jloc++;
      }
      jloc = founddiag?jloc+1:jloc+Nb;
    }
  }

  /* allocateandprint(A, descA, nA, mympirank, nprows, npcols, "Scaled matrix"); */

  MKL_INT mQl = numroc(&M, &Mb, &prow, &ZERO, &nprows);
  MKL_INT nQl = numroc(&M, &Mb, &pcol, &ZERO, &npcols);
  MKL_INT descQl [9];
  descinit(descQl, &M, &M, &Mb, &Mb, &ZERO, &ZERO, &ctxt, &mQl, &info);

  MKL_INT mQr = numroc(&N, &Nb, &prow, &ZERO, &nprows);
  MKL_INT nQr = numroc(&N, &Nb, &pcol, &ZERO, &npcols);
  MKL_INT descQr [9];
  descinit(descQr, &N, &N, &Nb, &Nb, &ZERO, &ZERO, &ctxt, &mQr, &info);

  // Allocate work vector for all subsequent operations.
  MKL_INT MINUSONE = -1;
  double lworkfloat;// This is in fact WORK!
  double *tau = (double *)malloc(M*sizeof(double)); // workrow;
  pdgeqrf(&M, &M, Q, &ONE, &ONE, descQl, tau, &lworkfloat, &MINUSONE, &info);
  MKL_INT lwork = (MKL_INT)lworkfloat;
  pdormqr("L", "N", &M, &N, &M,
          Q, &ONE, &ONE, descQl, tau,
          A, &ONE, &ONE, descA,
          &lworkfloat, &MINUSONE, &info);
  MKL_INT lworktemp = (MKL_INT)lworkfloat;
  lwork = lwork>lworktemp?lwork:lworktemp;
  pdgeqrf(&M, &M, Q, &ONE, &ONE, descQr, tau, &lworkfloat, &MINUSONE, &info);
  lworktemp = (MKL_INT)lworkfloat;
  lwork = lwork>lworktemp?lwork:lworktemp;
  pdormqr("R", "N", &M, &N, &M,
          Q, &ONE, &ONE, descQr, tau,
          A, &ONE, &ONE, descA,
          &lworkfloat, &MINUSONE, &info);
  lworktemp = (MKL_INT)lworkfloat;
  lwork = lwork>lworktemp?lwork:lworktemp;
  double *work = (double*)malloc(lwork*sizeof(double));

  // Compute QR decomposition of random matrix.
  for (jloc=0; jloc<nQl; jloc++)
    for (iloc=0; iloc<mQl; iloc++)
      Q[jloc*mQl+iloc] = rand() / (double)RAND_MAX;
  pdgeqrf(&M, &M, Q, &ONE, &ONE, descQl, tau, work, &lwork, &info);

  /* allocateandprint(Q, descQl, nQl, mympirank, nprows, npcols, "Scaled matrix"); */

  // A <- QA.
  pdormqr("L", "N", &M, &N, &M,
          Q, &ONE, &ONE, descQl, tau,
          A, &ONE, &ONE, descA,
          work, &lwork, &info);

  // Compute QR decomposition of another  random matrix.
  for (jloc=0; jloc<nQr; jloc++)
    for (iloc=0; iloc<mQr; iloc++)
      Q[jloc*mQr+iloc] = rand() / (double)RAND_MAX;

  // A <- AQ.
  pdgeqrf(&M, &M, Q, &ONE, &ONE, descQr, tau, work, &lwork, &info);
  pdormqr("R", "N", &M, &N, &M,
          Q, &ONE, &ONE, descQr, tau,
          A, &ONE, &ONE, descA,
          work, &lwork, &info);

  free(work);
}
