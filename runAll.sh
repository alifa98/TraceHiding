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

python experiment_scripts/05_1_random_sample_our_method.py entropy &
python experiment_scripts/06_2_random_sample_baseline_finetune.py &
python experiment_scripts/06_3_random_sample_baseline_neggrad.py &
python experiment_scripts/06_3_random_sample_baseline_neggrad.py plus &
python experiment_scripts/06_4_random_sample_baseline_badt.py &
python experiment_scripts/06_5_random_sample_baseline_scrub.py &

# Wait for all background processes to finish
wait

echo "All commands have finished."
