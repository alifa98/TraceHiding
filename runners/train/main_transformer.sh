#!/bin/bash
ORIGINAL_DIR=$(pwd)
cd /local/data1/users/faraji/unlearning_experiments_new || { echo "Failed to change directory"; exit 1; }

export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

export TORCH_COMPILE_DISABLE=0 # to disable the Triton optimization (the old gpus are not compatible with it)

export CUDA_VISIBLE_DEVICES=0
python experiment_scripts/02_5_modernBERT_trainer.py ModernBERT HO_Rome_Res8 &
export CUDA_VISIBLE_DEVICES=1
python experiment_scripts/02_5_modernBERT_trainer.py ModernBERT HO_Porto_Res8 &
export CUDA_VISIBLE_DEVICES=2
python experiment_scripts/02_5_modernBERT_trainer.py ModernBERT HO_Geolife_Res8 &
export CUDA_VISIBLE_DEVICES=6
python experiment_scripts/02_5_modernBERT_trainer.py ModernBERT HO_NYC_Res9 &

cd "$ORIGINAL_DIR" || { echo "Failed to return to the original directory"; exit 1; }
