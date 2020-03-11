#! /bin/bash
n=10

i=32
j=32
nproc=$(($i*$j))
cond=1e4

for k in `seq 1 $n`
do
    matrixsize=$(($k * 10000))
    qsub -pe hpc.pe $nproc \
         -v nprows="$i",npcols="$j", matrixsize="$matrixsize", cond="$cond",userandsvd="--full" \
         test_timing_sub.sh
done

while [[ $(qstat) ]]; do sleep 60; done

cat res_0*0000_1024_full.dat > timing_n_1024_full.dat
