#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd /local/data1/users/faraji/unlearning_experiments_new || { echo "Failed to change directory"; exit 1; }

declare -A sample_sizes
sample_sizes["HO_Rome_Res8"]="2 10 21 43"
sample_sizes["HO_NYC_Res9"]="2 11 23 46"
sample_sizes["HO_Geolife_Res8"]="1 2 3 5"
sample_sizes["HO_Porto_Res8"]="4 21 43 88"

biases=("entropy_max")
models=("LSTM" "GRU") # "LSTM" "GRU"
datasets=("HO_Rome_Res8" "HO_NYC_Res9" "HO_Geolife_Res8") # "HO_Rome_Res8" "HO_NYC_Res9" "HO_Geolife_Res8" "HO_Porto_Res8"
importances=("entropy" "coverage_diversity")

# Available GPUs
GPUs=(0 1 2 4 6 7)
num_gpus=${#GPUs[@]}
export GPUs_STR="${GPUs[*]}"

log_dir="cmd_logs"
mkdir -p "$log_dir"
timestamp=$(date +"%Y-%m-%d_%s")
# File to store commands
command_file="$log_dir/tracehiding_rnn_cmds_${timestamp}.txt"
> "$command_file"
# File to log failed commands
failed_commands_log="$log_dir/tracehiding_rnn_cmds_${timestamp}_failed.txt"
> "$failed_commands_log"

export FAILD_COMMAND_LIST_FILE="$failed_commands_log" # To be used in function (wasted 2 hours to find this bug)


execute_and_log_failure() {
    local slot_id=$1
    shift
    local cmd="$@"

    # Reconstruct GPUs array from exported string
    IFS=' ' read -r -a GPUs <<< "$GPUs_STR"

    gpu_id=${GPUs[$(( slot_id - 1 ))]}

    export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate unlearnF

    export TORCH_COMPILE_DISABLE=0 # to disable the Triton optimization (the old gpus are not compatible with it)

    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "Running on GPU $gpu_id (Slot ID: $slot_id): $cmd"

    eval "$cmd"
    status=$?
    if [ $status -ne 0 ]; then
        echo "Failed: $cmd"
        echo "$cmd" >> "$FAILD_COMMAND_LIST_FILE" 
    fi
}

export -f execute_and_log_failure

# Generate all possible combinations and store them in the command file
for dataset in "${datasets[@]}"; do
    for bias in "${biases[@]}"; do
        for sampleSize in ${sample_sizes[$dataset]}; do
            for model in "${models[@]}"; do
                for importance in "${importances[@]}"; do
                    cmd="python experiment_scripts/05_1_tracehiding_rnn.py --model $model --dataset $dataset --scenario user --sampleSize $sampleSize --biased $bias --batchSize 20 --importance $importance"
                    echo "$cmd" >> "$command_file"
                done
            done
        done
    done
done

cat "$command_file" | parallel -j$num_gpus --ungroup execute_and_log_failure {%} "{}"


cd "$ORIGINAL_DIR" || { echo "Failed to return to original directory"; exit 1; }