#!/bin/bash
#SBATCH --job-name=cuda-v1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

module rm gcc
module rm cuda

module load gcc
module load cuda

make clean
make v1

for d in 64 128 256; do
  for p in 3 5 7; do
    ./v1 ./test.txt $d $d $p $p
  done
done
