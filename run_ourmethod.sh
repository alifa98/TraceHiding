export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

python experiment_scripts/05_1_our_method.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 1 --batchSize 50  --importance entropy
python experiment_scripts/05_1_our_method.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 5 --batchSize 50  --importance entropy
python experiment_scripts/05_1_our_method.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 10 --batchSize 50  --importance entropy
python experiment_scripts/05_1_our_method.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 20 --batchSize 50  --importance entropy
python experiment_scripts/05_1_our_method.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 1 --batchSize 50  --importance coverage_diversity
python experiment_scripts/05_1_our_method.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 5 --batchSize 50  --importance coverage_diversity
python experiment_scripts/05_1_our_method.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 10 --batchSize 50  --importance coverage_diversity
python experiment_scripts/05_1_our_method.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --sampleSize 20 --batchSize 50  --importance coverage_diversity