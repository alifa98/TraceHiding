export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearnF

export CUDA_VISIBLE_DEVICES=5
python experiment_scripts/02_1_model_trainer.py GRU & 
python experiment_scripts/02_1_model_trainer.py & 
