#!/bin/bash

# Specify a name for the SLURM job
JOB_NAME="-J ${JOB_NAME:=run}"
# Hard time limit of the job, will forcefully be terminated if not done within
# this time
TIME_LIMIT="-t ${TIME_LIMIT:=03:00:00}"
# Number of CPUs to use per task
NUM_CPUS="--cpus-per-task=${NUM_CPUS:=16}"
# Amount of memory to allocate for the job
MEM="--mem ${MEM:=64G}"
# The partition to which the job is submitted
PARTITION="-p ${PARTITION:=gpu}"
# If using GPUS, specify which type of GPU and how many
#   Note: Hardcode this to 1, there is no need for more GPUs right now
if [[ "$PARTITION" = "-p gpu" ]] || [[ "$PARTITION" = "-p dgx" ]]; then
#   Note: Hardcode this to 1, there is no need for more GPUs right now
GPUS="--gres=gpu:h100:1"
fi;
# Group all sbatch command line arguments into one string
ARGS="$JOB_NAME $TIME_LIMIT $NUM_CPUS $MEM $PARTITION $GPUS $SBATCH_LOG"
# Forward all arguments following the shell script to be executed as the
# command line inside of another shell script which is executed via sbatch
#   Note: Waiting/Blocking sbatch with the -W option
# shellcheck disable=SC2086
echo $ARGS
sbatch $ARGS --verbose otus.sh "$@"
