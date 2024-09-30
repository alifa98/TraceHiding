export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=5

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF


python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method original --batchSize 20 &

python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method trace_hiding --sampleSize 1 --batchSize 50  --importance entropy & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method trace_hiding --sampleSize 5 --batchSize 50  --importance entropy & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method trace_hiding --sampleSize 10 --batchSize 50  --importance entropy & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method trace_hiding --sampleSize 20 --batchSize 50  --importance entropy & 

python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method trace_hiding --sampleSize 1 --batchSize 50  --importance coverage_diversity & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method trace_hiding --sampleSize 5 --batchSize 50  --importance coverage_diversity & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method trace_hiding --sampleSize 10 --batchSize 50  --importance coverage_diversity & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method trace_hiding --sampleSize 20 --batchSize 50  --importance coverage_diversity & 

python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method retraining --sampleSize 1 --batchSize 50 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method retraining --sampleSize 5 --batchSize 50 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method retraining --sampleSize 10 --batchSize 50 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method retraining --sampleSize 20 --batchSize 50  &

python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method finetune --sampleSize 1 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method finetune --sampleSize 5 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method finetune --sampleSize 10 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method finetune --sampleSize 20 --batchSize 20 &

python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method neg_grad --sampleSize 1 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method neg_grad --sampleSize 5 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method neg_grad --sampleSize 10 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method neg_grad --sampleSize 20 --batchSize 20 &

python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method neg_grad_plus --sampleSize 1 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method neg_grad_plus --sampleSize 5 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method neg_grad_plus --sampleSize 10 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method neg_grad_plus --sampleSize 20 --batchSize 20 &

python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method badt --sampleSize 1 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method badt --sampleSize 5 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method badt --sampleSize 10 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method badt --sampleSize 20 --batchSize 20 &

python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method scrub --sampleSize 1 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method scrub --sampleSize 5 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method scrub --sampleSize 10 --batchSize 20 & 
python experiment_scripts/07_2_mia_xgboost.py --model LSTM --dataset Ho_Foursquare_NYC --scenario user --method scrub --sampleSize 20 --batchSize 20 &

