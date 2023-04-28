#!/bin/bash
#SBATCH --partition=gpu-V100
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --output=tft0_out.txt
#SBATCH --error=tft0_err.txt
source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate TimeSeries
echo .... Running on $(hostname) ....
echo $CUDA_VISIBLE_DEVICES

python /storage/DSIP/edison/anmartinelli/main_tft.py -cl -t 

sleep 10
echo Job Done!