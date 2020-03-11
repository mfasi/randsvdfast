Extreme-scale matrices with specified 2-norm singular values
============================================================

This repository contains the code that produces the figures in Section 5.1.4 of:

M. Fasi and N. J. Higham. _Generating extreme-scale matrices with specified singular values or condition numbers_. Technical Report 2020.8, Manchester Institute for Mathematical Sciences, The University of Manchester, UK, Mar 2020.


Dependencies
------------

The code in this repository requires the Intel Math Kernel Library and OpenMPI. The Bash scripts to run the code are meant to be used in a queue system that supports the `qsub` command, such as the Portable Batch System (PBS) or the Oracle Grid Engine (SGE).


Compilation
-------------

Issuing `make all` should compile the source code and produce the three binaries `test_conditioning`, `test_orthogonal`, and `test_timing` used in the experiments. The Makefile that the code is being compiled in a Linux machine and that Intel MKL 2019 has been installed in the default location `/opt/intel/compilers_and_libraries_2019`. Should your system confirmation be different, you should change the variables `CMAKE_PATH` and `CMAKE_INCLUDE` in the Makefile accordingly.


Running the experiments
-----------------------

Once the three binaries have been generated successfully, in order to generate the data used for the figures in the preprint it suffices to run `run_tests.sh`.


License
-------

This software is distributed under the terms of the GNU GPL v. 2 software licence. All source code should include the following copyright notice and disclaimer:

~~~C
Copyright (c) 2020 Massimiliano Fasi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
~~~
