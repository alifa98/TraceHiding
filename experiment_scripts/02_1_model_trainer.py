import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pytorch_lightning.callbacks import EarlyStopping
from models.HoLSTM import LitHigherOrderLSTM
from models.HoGRU import LitHigherOrderGRU
from utility.functions import custom_collate_fn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

MODEL_NAME = "GRU"
DATASET_NAME = "HO_NYC_Checkins"

# MODEL PARAMETERS
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 250
NUMBER_OF_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 80
MAX_EPOCHS = 300

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

os.makedirs(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/", exist_ok=True)

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")
cell_to_id = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_cell_to_id.pt")
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

# LOAD DATASET
train_dloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
test_dloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, num_workers=24)

# Create model
if MODEL_NAME == "LSTM":
    model = LitHigherOrderLSTM(stats['vocab_size'],  stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
elif MODEL_NAME == "GRU":
    model = LitHigherOrderGRU(stats['vocab_size'],  stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
else:
    raise ValueError("Model name is not valid")

# Configure the EarlyStopping callback
early_stop_callback = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    min_delta=0.00,  # Minimum change to qualify as an improvement
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min'  # Because we want to minimize validation loss
)

CHECKPOINT_DIR = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/checkpoints"
trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCHS, enable_progress_bar=True, enable_checkpointing=True, default_root_dir=CHECKPOINT_DIR, callbacks=[early_stop_callback])

# save initial model
torch.save(model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/initial_{MODEL_NAME}_model.pt")

# train the model
trainer.fit(model, train_dloader, test_dloader)

logging.info("Training completed")


logging.info("Saving model ...")

# save the model
torch.save(model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt")

# model parameters
model_params = {
    "embedding_size": EMBEDDING_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "number_of_layers": NUMBER_OF_LAYERS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "trained_epochs": trainer.current_epoch
}
json.dump(model_params, open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "w"))

logging.info("Training completed")
