#!/bin/bash --login

#$ -cwd                         # Job runs in current directory
#$ -P hpc-nh-nahpnla

## Load the required modulefile
module load compilers/intel/18.0.3
module load mpi/intel-18.0/openmpi/3.1.4
module load mpi/nobind

## The variable NSLOTS sets the number of processes to match the pe core request
make test_orthogonal
mpirun -n $(($nprows * $npcols)) \
       ./test_orthogonal -M $matrixsize -m $nprows -n $npcols -p $nreps

##################$ -pe hpc.pe ${nproc}
