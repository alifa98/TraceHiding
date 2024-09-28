#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=5 # default for all scripts in this folder


source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 1 & 
CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 5 & 
CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 10 & 
CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 20 & 
CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset Ho_Foursquare_NYC --scenario random --sampleSize 200 & 

# CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset Ho_Foursquare_NYC --scenario user --sampleSize 1 & 
# CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset Ho_Foursquare_NYC --scenario user --sampleSize 5 & 
# CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset Ho_Foursquare_NYC --scenario user --sampleSize 10 & 
# CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset Ho_Foursquare_NYC --scenario user --sampleSize 20 & 
# CUDA_VISIBLE_DEVICES=5 python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset Ho_Foursquare_NYC --scenario random --sampleSize 200 & 

