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

#ifndef RANDSVD_LARGE_H_
#define RANDSVD_LARGE_H_

#include <stdbool.h>
#include <math.h>

#include <mpi.h>
#include <mkl.h>
#include <mkl_blas.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>

void orthog(double *, const MKL_INT*, const MKL_INT, const MKL_INT,
            const MKL_INT, const MKL_INT);

void orthog_householder(double *, const MKL_INT*, const MKL_INT,
                        const MKL_INT, const MKL_INT, const MKL_INT,
                        const size_t, double *);

void gencond_bwd(double *, const MKL_INT*, const MKL_INT,
                 const MKL_INT, const MKL_INT, const MKL_INT,
                 const double, double *, double *);

void gencond_fwd(double *, const MKL_INT*, const MKL_INT,
                 const MKL_INT, const MKL_INT, const MKL_INT,
                 const double, double *, double *);

void randsvd_bwd(double *, const MKL_INT*, const MKL_INT, const double*,
                 const MKL_INT, const MKL_INT, const MKL_INT,
                 const double, double *, double *);

void randsvd_fwd(double *, const MKL_INT*, const MKL_INT, const double*,
                 const MKL_INT, const MKL_INT, const MKL_INT,
                 const double, double *, double *);

void randsvd(const bool, double *, const MKL_INT *, const MKL_INT,
             const double *,
             const MKL_INT, const MKL_INT, const MKL_INT,
             double *, double *);

#endif // RANDSVD_LARGE_H_
