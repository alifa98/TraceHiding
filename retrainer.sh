#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
# export  # default for all scripts in this folder

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

CUDA_VISIBLE_DEVICES=5
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 1 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 5 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 10 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 20 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 1 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 5 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 10 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 20 &
