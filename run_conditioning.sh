#! /bin/bash
n=15

i=32
j=32
nproc=$(($i*$j))
matrixsize=50000

for k in `seq 0 $n`
do
    qsub -pe hpc.pe $nproc \
         -v nprows="$i",npcols="$j",matrixsize="$matrixsize",cond="1e$k",userandsvd="--full" \
         test_conditioning_sub.sh
done

while [[ $(qstat) ]]; do sleep 60; done

cat cond_0050000_*_full.dat > conditioning_0050000_k_full.dat
