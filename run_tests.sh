#! /bin/bash

# Experiment 1
run_orthogonal.sh # Fig. 2

# Experiment 2
run_conditioning.sh # Fig. 3

# Experiment 3
nreps=21
cond=1e8
run_full_speedup.sh # Fig. 4
qsub run_full_timings_sub.sh -v nreps=$nreps,cond=$cond # Fig. 5

run_new_speedup.sh # Fig. 6
qsub run_new_timings_sub.sh -v nreps=$nreps,cond=$cond # Fig. 7
