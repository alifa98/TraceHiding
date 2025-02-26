import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
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

# Check if the expected number of arguments is provided
if len(sys.argv) < 3:
    print("Usage: python experiment_scripts/02_1_model_trainer.py <model_type> <dataset_name>")
    sys.exit(1)
    
MODEL_NAME = sys.argv[1]
DATASET_NAME = sys.argv[2]

training_configs = {
    "HO_Rome_Res8": {
        "embedding_size": 256,
        "hidden_size": 128,
        "number_of_layers": 3,
        "dropout": 0.15,
        "batch_size": 10,
    },
    "HO_Porto_Res8": {
        "embedding_size": 1024,
        "hidden_size": 256,
        "number_of_layers": 3,
        "dropout": 0.1,
        "batch_size": 200,
    },
    "HO_Geolife_Res8": {
        "embedding_size": 256,
        "hidden_size": 64,
        "number_of_layers": 2,
        "dropout": 0.05,
        "batch_size": 10,
    },
    "HO_NYC_Res9": {
        "embedding_size": 256,
        "hidden_size": 128,
        "number_of_layers": 2,
        "dropout": 0.1,
        "batch_size": 50,
    }
}

# MODEL PARAMETERS
EMBEDDING_SIZE = training_configs[DATASET_NAME]["embedding_size"]
HIDDEN_SIZE = training_configs[DATASET_NAME]["hidden_size"]
NUMBER_OF_LAYERS = training_configs[DATASET_NAME]["number_of_layers"]
DROPOUT = training_configs[DATASET_NAME]["dropout"]
BATCH_SIZE = training_configs[DATASET_NAME]["batch_size"]
MAX_EPOCHS = 150
# ------------------------------------- END CONFIGURATIONS -------------------------------------#

os.makedirs(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/", exist_ok=True)

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)
cell_to_id = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_cell_to_id.pt", weights_only=False)
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

# LOAD DATASET
train_dloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
test_dloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, num_workers=24)

if MODEL_NAME == "LSTM":
    model = LitHigherOrderLSTM(stats['vocab_size'],  stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
    model_class = LitHigherOrderLSTM
elif MODEL_NAME == "GRU":
    model = LitHigherOrderGRU(stats['vocab_size'],  stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
    model_class = LitHigherOrderGRU
else:
    raise ValueError("Model name is not valid")

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=11,
    verbose=True,
    mode='min'
)

CHECKPOINT_DIR = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/checkpoints"

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

# save initial model
torch.save(model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/initial_{MODEL_NAME}_model.pt")

training_start_time = time.time()
trainer.fit(model, train_dloader, test_dloader)
training_end_time = time.time()
logging.info("Training completed")

logging.info(f"Training time: {training_end_time - training_start_time}")

logging.info("Loading the best model ...")
best_model_path = checkpoint_callback.best_model_path
print(f"Best model saved at: {best_model_path}")
model = model_class.load_from_checkpoint(best_model_path)

# test the model
logging.info("Testing the best model ...")
test_result = model.test_model(test_dloader)
logging.info(f"Test result: {test_result}")

logging.info("Saving the loaded model ...")

# save the model
torch.save(model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt")

# model parameters
model_params = {
    "embedding_size": EMBEDDING_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "number_of_layers": NUMBER_OF_LAYERS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "trained_epochs": trainer.current_epoch,
    "test_result": test_result,
    "training_time": training_end_time - training_start_time
}
json.dump(model_params, open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "w"))

logging.info("Model and its stats is saved.")
