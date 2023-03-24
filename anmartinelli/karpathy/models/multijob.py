import os

v = """#!/bin/bash
#SBATCH --partition=gpu-K80
#SBATCH --gres=gpu:1 
#SBATCH --mem=16000 
#SBATCH --output=run_out.txt 
#SBATCH --error=run_err.txt
source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate TimeSeries
echo .... Running on $(hostname) ....
echo $CUDA_VISIBLE_DEVICES

python /storage/DSIP/edison/anmartinelli/models/main.py -cl -t change

sleep 10
echo Job Done!
"""
#               -mod: SeqLen_Lag_Enc_Dec_Embd_Head_HeadSize_FwExp_Dropout_LR_WD_E_BS_Hour_SchedStep
list_models = ['-m -tft -mod 265_65_4_2_64_4_16_3_0.3_1e-04_0.0_1000_32_24_100 -bs_t 2 -hr_t 24',
               '-m -tft -mod 265_65_4_2_64_4_16_3_0.3_1e-06_0.0_1000_32_24_100 -bs_t 2 -hr_t 24',
               '-m -tft -mod 265_65_4_2_128_4_32_3_0.3_1e-04_0.0_1000_32_24_100 -bs_t 2 -hr_t 24',
               '-m -tft -mod 265_65_4_2_128_4_32_3_0.3_1e-06_0.0_1000_32_24_100 -bs_t 2 -hr_t 24'
               ]
for k, model in enumerate(list_models):
    tmp = v.replace("change", model)
    tmp = tmp.replace("--output=run_out.txt",f"--output=tft_{k+5}_out.txt")
    tmp = tmp.replace("--error=run_err.txt", f"--error=tft_{k+5}_err.txt")
    
    with open(f"run{k}.sh", "w") as f:
        f.write(tmp)
    os.system(f"sbatch run{k}.sh")
    os.system(f"rm run{k}.sh")
