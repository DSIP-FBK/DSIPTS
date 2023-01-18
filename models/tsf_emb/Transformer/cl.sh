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
cd tsf_emb
python Transformer/main.py -t -p -r -E 500 -bs 16 -bs_t 4 -enc 4 -dec 2 -xe 4 -ye 4 -fe 16 -he 2 -f 1 -lr 1e-04

sleep 10
echo Job Done!
