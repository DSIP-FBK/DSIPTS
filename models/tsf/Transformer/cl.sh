#!/bin/bash
#SBATCH --partition=gpu-K80
#SBATCH --gres=gpu:1 
#SBATCH --mem=16000 
#SBATCH --output=4_2_04_out.txt 
#SBATCH --error=4_2_04_err.txt
source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate TimeSeries
echo .... Running on $(hostname) ....
echo $CUDA_VISIBLE_DEVICES
cd
cd tsf
python Transformer/main.py -t -l -p -r -E 500

sleep 10
echo Job Done!
