export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH


source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

# SAMPLE SIZES FOR EACH DATASET:
# Ho_Foursquare_NYC: 1, 5, 10, 20, 200
# HO_Rome_Res8: 1, 2, 3, 4, 113
# HO_Porto_Res8: 1, 5, 10, 21, 90000
# HO_Rome_Res8: 1, 5, 10, 20, 270

export CUDA_VISIBLE_DEVICES=5
python experiment_scripts/06_2_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 1 --batchSize 20 --biased entropy &
python experiment_scripts/06_2_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 5 --batchSize 20 --biased entropy &
python experiment_scripts/06_2_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --biased entropy &
python experiment_scripts/06_2_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 20 --batchSize 20 --biased entropy &
# python experiment_scripts/06_2_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --scenario random --sampleSize 570 --batchSize 20 --biased entropy &

# export CUDA_VISIBLE_DEVICES=1
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 1 --batchSize 20 --plus True --biased entropy &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 5 --batchSize 20 --plus True --biased entropy &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --plus True --biased entropy &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 20 --batchSize 20 --plus True --biased entropy &
# python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario random --sampleSize 570 --batchSize 20 --plus True --biased entropy &

# export CUDA_VISIBLE_DEVICES=2
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 1 --batchSize 20 --biased entropy &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 5 --batchSize 20 --biased entropy &
wait
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --biased entropy &
python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 20 --batchSize 20 --biased entropy &
# python experiment_scripts/06_3_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --scenario random --sampleSize 570 --batchSize 20 --biased entropy &

# export CUDA_VISIBLE_DEVICES=3
python experiment_scripts/06_4_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 1 --batchSize 20 --biased entropy &
python experiment_scripts/06_4_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 5 --batchSize 20 --biased entropy &
python experiment_scripts/06_4_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --biased entropy &
python experiment_scripts/06_4_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 20 --batchSize 20 --biased entropy &
# python experiment_scripts/06_4_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --scenario random --sampleSize 570 --batchSize 20 --biased entropy &

# export CUDA_VISIBLE_DEVICES=4
python experiment_scripts/06_5_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 1 --batchSize 20 --biased entropy &
python experiment_scripts/06_5_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 5 --batchSize 20 --biased entropy &
python experiment_scripts/06_5_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 10 --batchSize 20 --biased entropy &
python experiment_scripts/06_5_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --scenario user --sampleSize 20 --batchSize 20 --biased entropy &
# python experiment_scripts/06_5_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --scenario random --sampleSize 570 --batchSize 20 --biased entropy &

wait