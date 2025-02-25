#!/bin/bash

ORIGINAL_DIR=$(pwd)
cd /local/data1/users/faraji/unlearning_experiments_new || { echo "Failed to change directory"; exit 1; }

declare -A sample_sizes
sample_sizes["HO_Rome_Res8"]="2 10 21 43"
sample_sizes["HO_NYC_Res9"]="2 11 23 46"
sample_sizes["HO_Geolife_Res8"]="1 2 3 5"
sample_sizes["HO_Porto_Res8"]="4 21 43 88"

biases=("entropy_max")
models=("GRU" "LSTM")

datasets=("HO_Rome_Res8" "HO_NYC_Res9" "HO_Geolife_Res8") #"HO_Porto_Res8"
importances=("entropy" "coverage_diversity")
methods=("retraining" "finetune" "neg_grad" "neg_grad_plus" "bad-t" "scrub" "trace_hiding")

# Available GPUs
GPUs=(0 1 2 4 5 6 7)
num_gpus=${#GPUs[@]}
export GPUs_STR="${GPUs[*]}"

log_dir="cmd_logs"
mkdir -p "$log_dir"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# File to store commands
command_file="$log_dir/eval_rnn_commands_list_$timestamp.txt"
> "$command_file"
# File to log failed commands
failed_commands_log="$log_dir/eval_rnn_failed_commands_list_$timestamp.txt"
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
                for method in "${methods[@]}"; do
                    if [[ "$method" == "trace_hiding" ]]; then
                        # If method is trace_hiding, include importance
                        for importance in "${importances[@]}"; do
                            cmd="python experiment_scripts/07_1_metrics_eval.py \
                                --model $model --dataset $dataset --scenario user \
                                --method $method --sampleSize $sampleSize --biased $bias --batchSize 20 --importance $importance"
                            echo "$cmd" >> "$command_file"
                        done
                    else
                        # Otherwise, exclude the importance argument
                        cmd="python experiment_scripts/07_1_metrics_eval.py \
                            --model $model --dataset $dataset --scenario user \
                            --method $method --sampleSize $sampleSize --biased $bias --batchSize 20"
                        echo "$cmd" >> "$command_file"
                    fi
                done
            done
        done
    done
done

cat "$command_file" | parallel -j$num_gpus --ungroup execute_and_log_failure {%} "{}"


cd "$ORIGINAL_DIR" || { echo "Failed to return to original directory"; exit 1; }
