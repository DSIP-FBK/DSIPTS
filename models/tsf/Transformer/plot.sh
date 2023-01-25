#!/bin/bash
#SBATCH --partition=gpu-V100
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --output=plot_out.txt
#SBATCH --error=plot_err.txt
source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate TimeSeries
echo .... Running on $(hostname) ....
echo $CUDA_VISIBLE_DEVICES

cd
cd tsf
python Transformer/main.py -l -p -r -mod 16_5000_256_60_1e-07_0.0_4_2_4_4_16_2_2_0.0

sleep 10
echo Job Done!