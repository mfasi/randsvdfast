#! /bin/bash
matrixsize=50000
cond=1e4

# Do small matrices by hand on high memory nodes.
# ssh node785 -t "cd `pwd`; bash -c \"make test_timing; mpirun -n   1 -hostfile ./hostfile_himem ./test_timing $matrixsize 1 1\""
# ssh node785 -t "cd `pwd`; bash -c \"make test_timing; mpirun -n   2 -hostfile ./hostfile_himem ./test_timing $matrixsize 2 1\""
# ssh node785 -t "cd `pwd`; bash -c \"make test_timing; mpirun -n   4 -hostfile ./hostfile_himem ./test_timing $matrixsize 2 2\""
# ssh node785 -t "cd `pwd`; bash -c \"make test_timing; mpirun -n   8 -hostfile ./hostfile_himem ./test_timing $matrixsize 4 2\""
# ssh node785 -t "cd `pwd`; bash -c \"make test_timing; mpirun -n  16 -hostfile ./hostfile_himem ./test_timing $matrixsize 4 4\""
# ssh node785 -t "cd `pwd`; bash -c \"make test_timing; mpirun -n  32 -hostfile ./hostfile_himem ./test_timing $matrixsize 8 4\""
# ssh node785 -t "cd `pwd`; bash -c \"make test_timing; mpirun -n  64 -hostfile ./hostfile_himem ./test_timing $matrixsize 8 8\""

n=11
i=1
j=1
for k in `seq 1 $n`
do
    if [ $(($i * $j)) -lt 129 ]
    then
        nproc=128
    else
        nproc=$(($i * $j))
    fi

    qsub -pe hpc.pe $nproc \
         -v nprows="$i", npcols="$j",matrixsize="$matrixsize",cond="$cond",userandsvd="--full" \
         test_timing_sub.sh
    if [ $(($k % 2)) == 0 ]
    then
        i=$(($i * 2))
    else
        j=$(($j * 2))
    fi
done

while [[ $(qstat) ]]; do sleep 60; done

cat res_0020000_*_full.dat > timing_0020000_p_full.dat
