#!/bin/bash
#SBATCH --job-name=v0
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --time=03:30:00

module rm gcc
module rm cuda

module load gcc
module load cuda

make clean
make v0

for d in 256; do
  for p in 7; do
    ./v0 ./test.txt $d $d $p $p
  done
done
