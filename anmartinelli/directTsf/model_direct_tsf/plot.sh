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

python tsf_direct/main.py -l -p -r -mod 12_3000_256_60_0.0001_0.0_4_2_8_8_32_4_2_0.0

sleep 10
echo Job Done!
