export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=6

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

python experiment_scripts/06_2_random_sample_baseline_finetune.py --model LSTM --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 50
python experiment_scripts/06_2_random_sample_baseline_finetune.py --model LSTM --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25
python experiment_scripts/06_2_random_sample_baseline_finetune.py --model LSTM --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10 
python experiment_scripts/06_2_random_sample_baseline_finetune.py --model GRU --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 100
python experiment_scripts/06_2_random_sample_baseline_finetune.py --model GRU --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25
python experiment_scripts/06_2_random_sample_baseline_finetune.py --model GRU --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10


python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model LSTM --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 50 --plus True
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model LSTM --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25 --plus True
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model LSTM --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10 --plus True
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model GRU --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 100 --plus True
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25 --plus True
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model GRU --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10 --plus True


python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model LSTM --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 50 
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model LSTM --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25 
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model LSTM --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10 
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model GRU --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 100 
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model GRU --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25 
python experiment_scripts/06_3_random_sample_baseline_neggrad.py --model GRU --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10 


python experiment_scripts/06_4_random_sample_baseline_badt.py --model LSTM --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 50
python experiment_scripts/06_4_random_sample_baseline_badt.py --model LSTM --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25
python experiment_scripts/06_4_random_sample_baseline_badt.py --model LSTM --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10 
python experiment_scripts/06_4_random_sample_baseline_badt.py --model GRU --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 100
python experiment_scripts/06_4_random_sample_baseline_badt.py --model GRU --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25
python experiment_scripts/06_4_random_sample_baseline_badt.py --model GRU --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10

python experiment_scripts/06_5_random_sample_baseline_scrub.py --model LSTM --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 50
python experiment_scripts/06_5_random_sample_baseline_scrub.py --model LSTM --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25
python experiment_scripts/06_5_random_sample_baseline_scrub.py --model LSTM --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10 
python experiment_scripts/06_5_random_sample_baseline_scrub.py --model GRU --dataset HO_Porto_Res8 --sampleSize 45700 --batchSize 100
python experiment_scripts/06_5_random_sample_baseline_scrub.py --model GRU --dataset HO_Rome_Res8 --sampleSize 135 --batchSize 25
python experiment_scripts/06_5_random_sample_baseline_scrub.py --model GRU --dataset HO_Geolife_Res8 --sampleSize 50 --batchSize 10

