#!/bin/bash
ORIGINAL_DIR=$(pwd)
cd /local/data1/users/faraji/unlearning_experiments_new || { echo "Failed to change directory"; exit 1; }

export LD_LIBRARY_PATH=/usr/local/cuda-12/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

export CUDA_VISIBLE_DEVICES=0
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 1 --batchSize 10  --importance entropy & 
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 2 --batchSize 10  --importance entropy & 
export CUDA_VISIBLE_DEVICES=1
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 3 --batchSize 10  --importance entropy & 
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 5 --batchSize 10  --importance entropy &
export CUDA_VISIBLE_DEVICES=2
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 1 --batchSize 10 --importance coverage_diversity & 
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 2 --batchSize 10  --importance coverage_diversity & 
export CUDA_VISIBLE_DEVICES=4
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 3 --batchSize 10  --importance coverage_diversity & 
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 5 --batchSize 10  --importance coverage_diversity &
export CUDA_VISIBLE_DEVICES=5
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 1 --batchSize 10  --importance uuniqe &
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 2 --batchSize 10  --importance uuniqe &
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 3 --batchSize 10  --importance uuniqe &
python experiment_scripts/05_1_our_method.py --model GRU --dataset HO_Geolife_Res8 --scenario user --biased entropy_max --sampleSize 5 --batchSize 10  --importance  uuniqe &

cd "$ORIGINAL_DIR" || { echo "Failed to return to the original directory"; exit 1; }

# To not run and exit immediately, comment out the last line
wait