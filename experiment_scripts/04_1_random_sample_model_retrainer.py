import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import Subset
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utility.functions import custom_collate_fn
from models.HoLSTM import LitHigherOrderLSTM
from models.HoGRU import LitHigherOrderGRU
from pytorch_lightning.callbacks import EarlyStopping
import json
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "LSTM"
# DATASET_NAME = "HO_Rome_Res8"
# DATASET_NAME = "HO_Porto_Res8"
DATASET_NAME = "HO_Geolife_Res8"
RANDOM_SAMPLE_UNLEARNING_SIZES =[50] # Rome:135, Porto: 45700, Geolife: 50
REPETITIONS_OF_EACH_SAMPLE_SIZE = 2

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

# load model parameters
model_params = json.load(open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "r"))

EMBEDDING_SIZE = model_params["embedding_size"]
HIDDEN_SIZE = model_params["hidden_size"]
NUMBER_OF_LAYERS = model_params["number_of_layers"]
DROPOUT = model_params["dropout"]
BATCH_SIZE = model_params["batch_size"]
MAX_EPOCHS = 300

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

for sample_size in RANDOM_SAMPLE_UNLEARNING_SIZES:
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        remaining_indexes = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")

        # LOAD DATASET
        reamining_dataset = Subset(train_dataset, remaining_indexes)

        reamining_dloader = DataLoader(reamining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        test_dloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, num_workers=24)

        # model
        if MODEL_NAME == "LSTM":
            model = LitHigherOrderLSTM(stats['vocab_size'],  stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
        elif MODEL_NAME == "GRU":
            model = LitHigherOrderGRU(stats['vocab_size'],  stats['users_size'], EMBEDDING_SIZE, HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)
        else:
            raise Exception("Model name is not defined correctly")

        # Configure the EarlyStopping callback
        early_stop_callback = EarlyStopping(
            monitor='val_loss',  # Metric to monitor
            min_delta=0.00,  # Minimum change to qualify as an improvement
            patience=30,  # Number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'  # Because we want to minimize validation loss
        )

        os.makedirs(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retraining", exist_ok=True)
        
        CHECKPOINT_DIR = f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retraining/checkpoints/"
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

        # train the model
        logging.info(f'Training the model for the remaining data, unlearning sample size: {sample_size}, repetition: {i}')
        trainer.fit(model, reamining_dloader, test_dloader)

        # save the model
        torch.save(model, f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retraining/retrained_{MODEL_NAME}_model.pt")
        logging.info(f'Model is saved for the remaining data, unlearning sample size: {sample_size}, sample: {i}')
