#!/bin/bash
#SBATCH --partition=gpu-K80
#SBATCH --gres=gpu:1 
#SBATCH --mem=16000 
#SBATCH --output=cl_out.txt 
#SBATCH --error=cl_err.txt
source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate TimeSeries
echo .... Running on $(hostname) ....
echo $CUDA_VISIBLE_DEVICES

python mj_training.py

sleep 10
echo Job Done!
