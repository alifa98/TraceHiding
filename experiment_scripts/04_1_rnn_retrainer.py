import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.ArguemntParser import get_args
from torch.utils.data import Subset
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utility.functions import custom_collate_fn
from models.HoLSTM import LitHigherOrderLSTM
from models.HoGRU import LitHigherOrderGRU
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import json
import torch

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#
args = get_args()
MODEL_NAME = args.model
DATASET_NAME = args.dataset
SAMPLE_SIZE = args.sampleSize
SCENARIO = args.scenario
BIASED_SAMPLE_IMPORTANCE_NAME = args.biased # if it is None, then the sample is not biased
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

model_params = json.load(open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "r"))

EMBEDDING_SIZE = model_params["embedding_size"]
HIDDEN_SIZE = model_params["hidden_size"]
NUMBER_OF_LAYERS = model_params["number_of_layers"]
DROPOUT = model_params["dropout"]
BATCH_SIZE = model_params["batch_size"]
MAX_EPOCHS = 150

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
    
    # base folder
    base_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample{f"_biased_{BIASED_SAMPLE_IMPORTANCE_NAME}" if BIASED_SAMPLE_IMPORTANCE_NAME else ""}/sample_size_{SAMPLE_SIZE}/sample_{i}"
    
    remaining_indexes = torch.load(f"{base_folder}/data/remaining.indexes.pt", weights_only=False)
    reamining_dataset = Subset(train_dataset, remaining_indexes)

    reamining_dloader = DataLoader(reamining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
    test_dloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, num_workers=24)

    if MODEL_NAME == "LSTM":
        model = LitHigherOrderLSTM(stats['vocab_size'],  stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
        model_class = LitHigherOrderLSTM
    elif MODEL_NAME == "GRU":
        model = LitHigherOrderGRU(stats['vocab_size'],  stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
        model_class = LitHigherOrderGRU
    else:
        raise Exception("Model name is not defined correctly")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=11,
        verbose=True,
        mode='min'
    )

    results_folder = f"{base_folder}/{MODEL_NAME}/retraining"
    os.makedirs(results_folder, exist_ok=True)
    
    CHECKPOINT_DIR = f"{results_folder}/checkpoints/"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=MAX_EPOCHS,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir=CHECKPOINT_DIR,
        callbacks=[
            early_stop_callback,
            checkpoint_callback
       ]
    )
    
    
    training_start_time = time.time()
    trainer.fit(model, reamining_dloader, test_dloader)
    training_end_time = time.time()
    logging.info("Retraining of the model is done.")
    
    logging.info(f"Training time: {training_end_time - training_start_time}")
    
    
    logging.info("Loading the best model ...")
    best_model_path = checkpoint_callback.best_model_path
    model = model_class.load_from_checkpoint(best_model_path)
    
    torch.save(model, f"{results_folder}/retrained_{MODEL_NAME}_model.pt")
    
    logging.info(f"Model is now saved in: {results_folder}/retrained_{MODEL_NAME}_model.pt")
    
    # model parameters
    retrained_stats = {
        "trained_epochs": trainer.current_epoch,
        "training_time": training_end_time - training_start_time
    }
    json.dump(retrained_stats, open(f"{results_folder}/retrained_{MODEL_NAME}_model.json", "w"))
    
    logging.info(f"Retraining stats are saved in: {results_folder}/retrained_{MODEL_NAME}_model.json")

    logging.info(f"The model For {SCENARIO} sample unlearning of sample {i} of size {SAMPLE_SIZE} retrained successfully.")
