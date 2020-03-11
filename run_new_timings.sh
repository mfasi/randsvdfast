#! /bin/bash
n=18

i=32
j=32
nproc=$(($i*$j))
cond=1e4


for k in `seq 1 $n`
do
    matrixsize=$(($k * 50000))
    qsub -pe hpc.pe $nproc \
         -v nprows="$i",npcols="$j",matrixsize="$matrixsize",cond="1e$k",userandsvd="--new" \
         test_timing_sub.sh
done

while [[ $(qstat) ]]; do sleep 60; done

cat res_0*0000_1024_new.dat > timing_n_1024_new.dat
