#!/bin/bash
#SBATCH --job-name=Inat
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=Inat.log
#SBATCH --gres=gpu:4
#SBATCH -c 40 
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h 

PORT=$[$RANDOM + 10000]
#source activate py3.6pt1.7


python paco_binary.py \
  --dataset imagenet \
  --data data/imagenet \
  --alpha 0.05 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --mark paco_binary_R50_adam_mocot0.2_augrandcls_sim_400epochs_lr0.01_t1 \
  --lr 0.01 \
  --moco-t 0.2 \
  --aug randcls_sim \
  --randaug_m 10 \
  --randaug_n 2 \
  --dist-url "tcp://localhost:$PORT" \
  --epochs 400 \
  --batch-size 256 \
  --optimizer adam
