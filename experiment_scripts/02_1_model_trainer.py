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

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "LSTM"
DATASET_NAME = "Ho_Foursquare_NYC"
# DATASET_NAME = "HO_Porto_Res8"
# DATASET_NAME = "HO_Geolife_Res8"

# MODEL PARAMETERS
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
NUMBER_OF_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 20
MAX_EPOCHS = 300
# {embedding_size': 231, 'hidden_size': 447, 'number_of_layers': 3, 'dropout': 0.29125920978558156, 'batch_size': 22} best params for LSTM on HO_Rome_Res8-v2
# ------------------------------------- END CONFIGURATIONS -------------------------------------#

os.makedirs(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/", exist_ok=True)

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)
cell_to_id = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_cell_to_id.pt", weights_only=False)
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
    patience=7,  # Number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min'  # Because we want to minimize validation loss
)

CHECKPOINT_DIR = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/checkpoints"
trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=MAX_EPOCHS,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir=CHECKPOINT_DIR,
        callbacks=[
            early_stop_callback
       ]
)

# save initial model
torch.save(model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/initial_{MODEL_NAME}_model.pt")


#debug: put just one batch in train_dloader to check if the model is working
# for batch in train_dloader:
#     train_dloader = DataLoader(train_dataset[:BATCH_SIZE], batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
#     test_dloader = DataLoader(train_dataset[:BATCH_SIZE], batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, num_workers=24)
#     break

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
