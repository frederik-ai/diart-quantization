#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=32G

quantization_types=("none" "static" "dynamic")

for smq in "${quantization_types[@]}"; do
 for emq in "${quantization_types[@]}"; do
   echo "Running benchmark with smq=$smq and emq=$emq"
   python diart_benchmark.py -smq "$smq" -emq "$emq"
 done
done

