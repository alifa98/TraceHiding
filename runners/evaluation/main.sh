#!/bin/bash
ORIGINAL_DIR=$(pwd)
cd /local/data1/users/faraji/unlearning_experiments_new || { echo "Failed to change directory"; exit 1; }

export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF


declare -A sample_sizes
sample_sizes["HO_Rome_Res8"]="2 10 21 43"
sample_sizes["HO_Porto_Res8"]="4 21 43 88"
sample_sizes["HO_Geolife_Res8"]="1 2 3 5"
sample_sizes["HO_NYC_Res9"]="2 11 23 46"

models=("GRU" "LSTM")
datasets=("HO_Rome_Res8" "HO_Porto_Res8" "HO_Geolife_Res8" "HO_NYC_Res9")
importance_list=("entropy" "coverage_diversity")
methods=("retraining" "finetune" "neg_grad" "neg_grad_plus" "bad-t" "scrub" "trace_hiding")

# available GPUs
GPUs=(0 1 2 4 5 6 7)

# Create a temporary file to store commands
command_file="eval_commands_list.txt"
> "$command_file"  # Clear the file

# all possible combinations
for dataset in "${datasets[@]}"; do
  for sampleSize in ${sample_sizes[$dataset]}; do
    for model in "${models[@]}"; do
      for method in "${methods[@]}"; do
        if [[ "$method" == "trace_hiding" ]]; then
          # If method is trace_hiding, include importance
          for importance in "${importance_list[@]}"; do
            echo "CUDA_VISIBLE_DEVICES={%} python experiment_scripts/07_1_metrics_eval.py \
                  --model $model --dataset $dataset --scenario user \
                  --method $method --sampleSize $sampleSize --batchSize 20 --importance $importance" >> "$command_file"
          done
        else
          # Otherwise, exclude the importance argument
          echo "CUDA_VISIBLE_DEVICES={%} python experiment_scripts/07_1_metrics_eval.py \
                --model $model --dataset $dataset --scenario user \
                --method $method --sampleSize $sampleSize --batchSize 20" >> "$command_file"
        fi
      done
    done
  done
done

# run tasks on 7 available GPUs
cat "$command_file" | parallel -j7 --env CUDA_VISIBLE_DEVICES --ungroup --jobs 7 --lb --arg-sep '{%}' echo ${GPUs[@]}


cd "$ORIGINAL_DIR" || { echo "Failed to return to the original directory"; exit 1; }

# To not run and exit immediately, comment out the last line
wait