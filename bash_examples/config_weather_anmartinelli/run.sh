#!/bin/bash
#SBATCH --job-name=comparing
#SBATCH --output=comparing.txt
#SBATCH --error=comparing.txt
#SBATCH --partition=gpu-V100
#SBATCH --mem=20000
#SBATCH --nice=0
#SBATCH --gres=gpu:1



echo .... Running on $(hostname) ....

cd /home/agobbi/Projects/timeseries/bash_examples
eval "$(conda shell.bash hook)"
conda activate tt

python compare.py -c config_weather/compare.yaml


echo Job Done!
