export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

python experiment_scripts/05_1_random_sample_our_method.py --model GRU --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 --importance entropy
python experiment_scripts/05_1_random_sample_our_method.py --model LSTM --dataset Ho_Foursquare_NYC --sampleSize 200 --batchSize 20 --importance entropy