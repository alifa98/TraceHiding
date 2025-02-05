#!/bin/bash
ORIGINAL_DIR=$(pwd)
cd /local/data1/users/faraji/unlearning_experiments_new || { echo "Failed to change directory"; exit 1; }

export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

export CUDA_VISIBLE_DEVICES=5
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 1 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 2 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 3 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 5 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 1 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 2 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 3 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 5 &


python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Porto_Res8 --scenario user --biased entropy_max --sampleSize 4 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Porto_Res8 --scenario user --biased entropy_max --sampleSize 21 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Porto_Res8 --scenario user --biased entropy_max --sampleSize 43 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Porto_Res8 --scenario user --biased entropy_max --sampleSize 88 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Porto_Res8 --scenario user --biased entropy_max --sampleSize 4 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Porto_Res8 --scenario user --biased entropy_max --sampleSize 21 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Porto_Res8 --scenario user --biased entropy_max --sampleSize 43 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Porto_Res8 --scenario user --biased entropy_max --sampleSize 88 &


python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_NYC_Res9 --scenario user --biased entropy_max --sampleSize 2 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_NYC_Res9 --scenario user --biased entropy_max --sampleSize 11 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_NYC_Res9 --scenario user --biased entropy_max --sampleSize 23 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_NYC_Res9 --scenario user --biased entropy_max --sampleSize 46 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_NYC_Res9 --scenario user --biased entropy_max --sampleSize 2 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_NYC_Res9 --scenario user --biased entropy_max --sampleSize 11 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_NYC_Res9 --scenario user --biased entropy_max --sampleSize 23 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_NYC_Res9 --scenario user --biased entropy_max --sampleSize 46 &


python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 2 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 10 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 21 &
python experiment_scripts/04_1_model_retrainer.py --model GRU --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 43 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 2 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 10 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 21 &
python experiment_scripts/04_1_model_retrainer.py --model LSTM --dataset HO_Rome_Res8 --scenario user --biased entropy_max --sampleSize 43 &


cd "$ORIGINAL_DIR" || { echo "Failed to return to the original directory"; exit 1; }
