#!/bin/bash

# Function to kill all background processes
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p)
    wait
}

# Trap SIGINT (ctrl+c) and call the cleanup function
trap cleanup SIGINT

# Activate conda environment and run the commands in the background
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lit2

# Set CUDA_VISIBLE_DEVICES for each command
CUDA_VISIBLE_DEVICES=3 python experiment_scripts/02_1_model_trainer.py LSTM &
CUDA_VISIBLE_DEVICES=6 python experiment_scripts/02_1_model_trainer.py GRU &
CUDA_VISIBLE_DEVICES=3 python experiment_scripts/04_1_random_sample_remaining_data_retrainer.py LSTM &
CUDA_VISIBLE_DEVICES=6 python experiment_scripts/04_1_random_sample_remaining_data_retrainer.py GRU &
CUDA_VISIBLE_DEVICES=6 python experiment_scripts/04_2_random_user_remaining_data_retrainer.py LSTM &
CUDA_VISIBLE_DEVICES=3 python experiment_scripts/04_2_random_user_remaining_data_retrainer.py GRU &

# Wait for all background processes to finish
wait

echo "All commands have finished."
