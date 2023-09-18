import os
import time
import subprocess
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
cd /home/achelini/Projects/project_edison/edison/models/GNN5/
eval "$(conda shell.bash hook)"
conda activate /storage/DSIP/edison_GNN/env

python run.py --config /home/achelini/Projects/project_edison/edison/models/GNN5/config.ini -t -hl 4 -ih 10 -oh 4

sleep 2
echo Job Done!
"""

gpu = "K80"

n_layer1 = [1, 2, 3]
n_layer2 = [1, 2, 3]
hidden_layers1 = [32, 64, 128, 256]
hidden_layers2 = [32, 64, 128, 256]
hidden_gnn = [8, 16, 32, 64]
lrs = [1e-3]


max_jobs = 8
jobs_already_done = 0
run = []
lim_job = {
    "A40":13, 
    "V100": 7
}
gpus = list(lim_job.keys())
active_jobs = 0


def is_executable() -> tuple:
    active_jobs = int(subprocess.check_output("squeue -u $USER | wc -l",  shell = True, text=True))
    if active_jobs < max_jobs:
        return True, active_jobs
    else:
        return False, active_jobs


parameters = {}
i=0
for nl1 in n_layer1:
    for nl2 in n_layer2:
        for hl1 in hidden_layers1:
            for hl2 in hidden_layers2:
                for hfg in hidden_gnn:
                    for lr in lrs:
                        i += 1
                        parameters[i] = (hl1, hl2, nl1, nl2, hfg, lr)

for n in parameters.keys():
    hl1, hl2, nl1, nl2, hfg, lr = parameters[n]
    tmp = v.replace("-t -hl 4 -ih 10 -oh 4", f"-t -hl1 {hl1} -hl2 {hl2} -nl1 {nl1} -nl2 {nl2} -hfg {hfg} -lr {lr}")
    #tmp = v.replace("-t -hl 4 -ih 10 -oh 4", f"-p")

    tmp = tmp.replace("--job-name=TrainGNN", f"--job-name=KNN{n}")
    tmp = tmp.replace("--output=out.txt",f"--output=out{n}.txt")
    tmp = tmp.replace("--error=err.txt", f"--error=err{n}.txt")
    name = f"run{n}.sh"
    
    tmp = tmp.replace("--partition=gpu-K80", f"--partition=gpu-{gpu}")
    #if id not in saving:
    if n>jobs_already_done:
        executable, active_jobs = is_executable()
        while executable is not True:
            t = 3600*0.5
            time.sleep(t)
            executable, active_jobs = is_executable()              
        
        with open(name, "w") as f:
            f.write(tmp)
        os.system(f"sbatch {name}")
        os.system(f"rm {name}")

