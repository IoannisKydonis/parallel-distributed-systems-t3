#!/bin/bash
#SBATCH --job-name=cuda-v2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

module rm gcc
module rm cuda

module load gcc
module load cuda

make clean
make v2

for d in 64 128 256; do
  for p in 3 5 7; do
    ./v2 ./test.txt $d $d $p $p
  done
done
