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

#ifndef DEBUG_UTILS_H_
#define DEBUG_UTILS_H_

#include <stdio.h>
#include <math.h>

#include <mpi.h>
#include <mkl.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>

double condest_norminf(const double *, const MKL_INT*, const MKL_INT,
                       const MKL_INT, const MKL_INT, const MKL_INT,
                       MKL_INT *, double *);

double condest_norm2(const double *, const MKL_INT*, const MKL_INT,
                     const MKL_INT, const MKL_INT, const MKL_INT,
                     MKL_INT *, double *, double *, double *, double *);

void distribute(double *, double *,
                const MKL_INT *, const MKL_INT,
                const MKL_INT, const MKL_INT, const MKL_INT);

void collect(double *, double *,
             const MKL_INT*, const MKL_INT,
             const MKL_INT, const MKL_INT, const MKL_INT);

void printmatrix(const double *, const MKL_INT, const MKL_INT);

void allocateandprint(double *, const MKL_INT *, const MKL_INT,
                      const MKL_INT, const MKL_INT, const MKL_INT,
                      const char *);

#endif // DEBUG_UTILS_H_
