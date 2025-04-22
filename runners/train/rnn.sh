#!/bin/bash
ORIGINAL_DIR=$(pwd)
cd /local/data1/users/faraji/unlearning_experiments_new || { echo "Failed to change directory"; exit 1; }

export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

export CUDA_VISIBLE_DEVICES=0
python experiment_scripts/02_1_rnn_trainer.py LSTM HO_Rome_Res8 &
export CUDA_VISIBLE_DEVICES=1
python experiment_scripts/02_1_rnn_trainer.py GRU  HO_Rome_Res8 &

export CUDA_VISIBLE_DEVICES=2
python experiment_scripts/02_1_rnn_trainer.py LSTM HO_Geolife_Res8 &
export CUDA_VISIBLE_DEVICES=4
python experiment_scripts/02_1_rnn_trainer.py GRU  HO_Geolife_Res8 & 

export CUDA_VISIBLE_DEVICES=6
python experiment_scripts/02_1_rnn_trainer.py LSTM HO_NYC_Res9 & 
export CUDA_VISIBLE_DEVICES=7
python experiment_scripts/02_1_rnn_trainer.py GRU  HO_NYC_Res9 &

export CUDA_VISIBLE_DEVICES=5
python experiment_scripts/02_1_rnn_trainer.py LSTM HO_Porto_Res8 &
python experiment_scripts/02_1_rnn_trainer.py GRU  HO_Porto_Res8 &


cd "$ORIGINAL_DIR" || { echo "Failed to return to the original directory"; exit 1; }
