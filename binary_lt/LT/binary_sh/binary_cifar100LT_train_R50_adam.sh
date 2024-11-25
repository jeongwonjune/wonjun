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

# alpha 0.02 moco-t 0.05 for cifar100-lt
python paco_binary.py \
  --dataset cifar100 \
  --arch reactnet \
  --data data/ \
  --alpha 0.02 \
  --beta 1.0 \
  --gamma 1.0 \
  --wd 5e-4 \
  --lr 0.02 \
  --moco-t 0.05 \
  --epochs 400 \
  --batch-size 128 \
  --optimizer adam \
  --gpu 0

