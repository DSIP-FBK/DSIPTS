import os
v = """#!/bin/bash
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

python Transformer/main.py -t -l -p -r -E 2500
## python run.py -c config.ini -t -s 1

sleep 10
echo Job Done!


"""
n = 3
start = 1
for i in range(start,n):
    tmp = v.replace("-t -s 1", f"-a -r {i}")
    tmp = tmp.replace("--job-name=TrainImage", f"--job-name=DAT{i}")

    tmp = tmp.replace("--output=sf_out.txt",f"--output=sf_out{i}.txt")
    tmp = tmp.replace("--error=sf_err.txt", f"--error=sf_err{i}.txt")
    
    with open(f"run{i}.sh", "w") as f:
        f.write(tmp)
    os.system(f"sbatch run{i}.sh")

for i in range(start, n):
    os.system(f"rm run{i}.sh")
