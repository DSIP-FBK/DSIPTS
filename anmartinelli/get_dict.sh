#!/bin/bash
#SBATCH --partition=gpu-V100
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --output=tftDicts.txt
#SBATCH --error=tftDicts.txt
source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate TimeSeries
echo .... Running on $(hostname) ....
echo $CUDA_VISIBLE_DEVICES

python /storage/DSIP/edison/anmartinelli/get_dictConf.py -p '/storage/DSIP/edison/anmartinelli/works/M0/M0.pkl' '/storage/DSIP/edison/anmartinelli/works/M1/M1.pkl' '/storage/DSIP/edison/anmartinelli/works/M2/M2.pkl'

sleep 10
echo Job Done!