export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH


source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

export CUDA_VISIBLE_DEVICES=5 

python experiment_scripts/06_2_baseline_finetune.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 1 --batchSize 20 &
python experiment_scripts/06_2_baseline_finetune.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 5 --batchSize 20 &
python experiment_scripts/06_2_baseline_finetune.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 10 --batchSize 20 &
python experiment_scripts/06_2_baseline_finetune.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 20 --batchSize 20 &
python experiment_scripts/06_2_baseline_finetune.py --model LSTM --dataset Ho_Foursquare_NYC --scenario random --sampleSize 200 --batchSize 20 &

python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 1 --batchSize 20 --plus True &
python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 5 --batchSize 20 --plus True &
python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 10 --batchSize 20 --plus True &
python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 20 --batchSize 20 --plus True &
python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario random --sampleSize 200 --batchSize 20 --plus True &

python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 1 --batchSize 20 &
python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 5 --batchSize 20 &
python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 10 --batchSize 20 &
python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 20 --batchSize 20 &
python experiment_scripts/06_3_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --scenario random --sampleSize 200 --batchSize 20 &

export CUDA_VISIBLE_DEVICES=1
python experiment_scripts/06_4_baseline_badt.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 1 --batchSize 20 &
python experiment_scripts/06_4_baseline_badt.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 5 --batchSize 20 &
python experiment_scripts/06_4_baseline_badt.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 10 --batchSize 20 &
python experiment_scripts/06_4_baseline_badt.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 20 --batchSize 20 &
python experiment_scripts/06_4_baseline_badt.py --model LSTM --dataset Ho_Foursquare_NYC --scenario random --sampleSize 200 --batchSize 20 &

export CUDA_VISIBLE_DEVICES=0
python experiment_scripts/06_5_baseline_scrub.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 1 --batchSize 20 &
python experiment_scripts/06_5_baseline_scrub.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 5 --batchSize 20 &
python experiment_scripts/06_5_baseline_scrub.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 10 --batchSize 20 &
python experiment_scripts/06_5_baseline_scrub.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 20 --batchSize 20 &
python experiment_scripts/06_5_baseline_scrub.py --model LSTM --dataset Ho_Foursquare_NYC --scenario random --sampleSize 200 --batchSize 20 &


