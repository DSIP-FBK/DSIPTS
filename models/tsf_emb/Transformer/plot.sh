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
cd tsf_emb
python Transformer/main.py -r -bs 8 -bs_t 1 -E 2000 -enc 4 -dec 2 -xe 12 -ye 12 -fe 256 -he 3 -f 2 -lr 1e-07

sleep 10
echo Job Done!