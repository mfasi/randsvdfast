#! /bin/bash
n=10

i=32
j=32
nproc=$(($i*$j))
nreps=5
cond=1e4

for k in `seq 1 $n`
do
    matrixsize=$(($k*50000))
    qsub -pe hpc.pe $nproc \
         -v nprows="$i",npcols="$j",matrixsize="$matrixsize",cond="$cond",nreps="$nreps" \
         test_orthogonal_sub.sh
done

while [[ $(qstat) ]]; do sleep 60; done

cat orthog_*_1024.dat > orthogonal_n_1024.dat
