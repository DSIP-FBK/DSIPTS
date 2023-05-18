import os
import time
import subprocess
import pickle
v = """#!/bin/bash
#
##Note: in this file use two # symbols to comment a line!
#
## set the job name, the output files for stdout and stderr streams redirection
#SBATCH --job-name=TrainGNN
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#
## set desired queue and !MANDATORY! for GPUs: the number of GPUs used
#SBATCH --partition=gpu-K80
#SBATCH --gres=gpu:1
#
#SBATCH --mem=12000
#SBATCH --cores-per-socket=3
source ~/.bashrc

echo .... Running on $(hostname) ....
echo available cuda devices have IDs: $CUDA_VISIBLE_DEVICES
cd /home/achelini/Projects/project_edison/edison/models/GNN3/
eval "$(conda shell.bash hook)"
conda activate /storage/DSIP/edison_GNN/env

python run.py --config /home/achelini/Projects/project_edison/edison/models/GNN3/config.ini -t -hl 4 -ih 10 -oh 4

sleep 2
echo Job Done!
"""

max_jobs = 13
jobs_already_done = 0
run = []
lim_job = {
    "A40":13, 
    "V100": 7
}
gpus = list(lim_job.keys())
active_jobs = 0


def get_gpus(is_to_big = True):
    # devo controllare tutte le gpu che voglio controllare 
    # 
    job_on_gpu = 10000
    g = 0
    not_found = True
    out = None
    while not_found:
        if g<len(gpus):
            gpu = gpus[g]
            job_on_gpu = subprocess.check_output(f"squeue -p gpu-{gpu} -t R |  wc -l",  shell = True, text=True)
            job_on_gpu = int(job_on_gpu) -1
            if job_on_gpu < lim_job[gpu]:
                not_found = False
                out = gpu
            g += 1
        else:
            not_found = False
    if is_to_big:
        return "V100"
    else:
        if out == None:
            return "A40"
        else:
            return gpu

def is_executable() -> tuple:
    active_jobs = int(subprocess.check_output("squeue -u $USER | wc -l",  shell = True, text=True))
    if active_jobs < max_jobs:
        return True, active_jobs
    else:
        return False, active_jobs

models = os.listdir("/storage/DSIP/edison_GNN/outputs/GNN3/weight_train_saving")

for n, model in enumerate(models):
    print(f" we are doing the model {model}")
    tmp = v.replace("-t -hl 4 -ih 10 -oh 4", f"-p -m {model}")
    tmp = tmp.replace("--job-name=TrainGNN", f"--job-name=plot{n}")
    tmp = tmp.replace("--output=out.txt",f"--output=p_out{n}.txt")
    tmp = tmp.replace("--error=err.txt", f"--error=p_err{n}.txt")
    name = f"run{n}.sh"

    gpu = "1080"
  
    tmp = tmp.replace("--partition=gpu-K80", f"--partition=gpu-{gpu}")
    #if id not in saving:
    if (n+1) > jobs_already_done:
        executable, active_jobs = is_executable()
        while executable!=True:
            t = 5
            time.sleep(t)
            executable, active_jobs = is_executable()              
        
        with open(name, "w") as f:
            f.write(tmp)
        os.system(f"sbatch {name}")
        os.system(f"rm {name}")
