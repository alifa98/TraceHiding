#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd /local/data1/users/faraji/unlearning_experiments_new || { echo "Failed to change directory"; exit 1; }

# put desired commands in the file commands.txt (e.g. the faild commands from other runners)

# Available GPUs
GPUs=(0 1 2 3 4 6 7)
num_gpus=${#GPUs[@]}
export GPUs_STR="${GPUs[*]}"

command_file="runners/parallel_commands.txt"
failed_commands_log="runners/parallel_commands_failed.txt"
>> "$failed_commands_log"

export FAILD_COMMAND_LIST_FILE="$failed_commands_log"


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

    # export TORCH_COMPILE_DISABLE=0 # to disable the Triton optimization (the old gpus are not compatible with it)

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

cat "$command_file" | parallel -j$num_gpus --ungroup execute_and_log_failure {%} "{}"

cd "$ORIGINAL_DIR" || { echo "Failed to return to original directory"; exit 1; }
