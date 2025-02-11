#!/bin/bash
ORIGINAL_DIR=$(pwd)
cd /local/data1/users/faraji/unlearning_experiments_new || { echo "Failed to change directory"; exit 1; }

export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

export CUDA_VISIBLE_DEVICES=0
python experiment_scripts/06_2_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 2 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_2_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_2_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 21 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_2_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 43 --batchSize 20 --biased entropy_max &

export CUDA_VISIBLE_DEVICES=1
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 2 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 21 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 43 --batchSize 20 --biased entropy_max &

export CUDA_VISIBLE_DEVICES=2
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 2 --batchSize 20 --biased entropy_max --plus &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --biased entropy_max --plus &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 21 --batchSize 20 --biased entropy_max --plus &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 43 --batchSize 20 --biased entropy_max --plus &

export CUDA_VISIBLE_DEVICES=5
python experiment_scripts/06_4_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 2 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_4_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_4_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 21 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_4_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 43 --batchSize 20 --biased entropy_max &

export CUDA_VISIBLE_DEVICES=4
python experiment_scripts/06_5_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 2 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_5_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_5_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 21 --batchSize 20 --biased entropy_max &
python experiment_scripts/06_5_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 43 --batchSize 20 --biased entropy_max &


cd "$ORIGINAL_DIR" || { echo "Failed to return to the original directory"; exit 1; }

# To not run and exit immediately, comment out the last line
wait