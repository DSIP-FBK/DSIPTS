#!/bin/bash
#SBATCH --partition=gpu-K80
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --output=plot_out.txt
#SBATCH --error=plot_err.txt
source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate TimeSeries
echo .... Running on $(hostname) ....
echo $CUDA_VISIBLE_DEVICES

python mj_plot.py

sleep 10
echo Job Done!
