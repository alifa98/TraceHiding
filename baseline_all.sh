export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

python experiment_scripts/06_2_random_sample_baseline_finetune.py --model LSTM --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 &
python experiment_scripts/06_2_random_sample_baseline_finetune.py --model GRU --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 &


python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 --plus True &
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model GRU --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 --plus  True &


python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model LSTM --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 &
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model GRU --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 &


python experiment_scripts/06_4_random_sample_baseline_badt.py --model LSTM --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 &
python experiment_scripts/06_4_random_sample_baseline_badt.py --model GRU --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 &

python experiment_scripts/06_5_random_sample_baseline_scrub.py --model LSTM --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 &
python experiment_scripts/06_5_random_sample_baseline_scrub.py --model GRU --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 &

