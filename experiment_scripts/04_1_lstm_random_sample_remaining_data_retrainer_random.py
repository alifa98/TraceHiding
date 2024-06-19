# This file train an LSTM model to predict the user of the sequence of trajectory data points
# basically, this file train the smart teacher or the orginal model
# model parameters will be saved next to the model in josn format

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import Subset
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utility.functions import custom_collate_fn
from models.HoLSTM import LitHigherOrderLSTM
from pytorch_lightning.callbacks import EarlyStopping
import json
import torch



os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# ------------------------------------- START CONFIGURATIONS -------------------------------------#

DATASET_NAME = "nyc_checkins"
# unlearning data sampler parameters
RANDOM_SAMPLE_UNLEARNING_SIZES = [10, 20, 50, 100, 200, 300, 600, 1000]
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

# ------------------------------------- END CONFIGURATIONS -------------------------------------#


MODEL_NAME = "LSTM"  # model name that we are training in this file

# load model parameters
model_params = json.load(
    open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_model.json", "r"))

EMBEDDING_SIZE = model_params["embedding_size"]
HIDDEN_SIZE = model_params["hidden_size"]
NUMBER_OF_LAYERS = model_params["number_of_layers"]
DROPOUT = model_params["dropout"]
BATCH_SIZE = model_params["batch_size"]
MAX_EPOCHS = 100

train_dataset = torch.load(
    f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
test_dataset = torch.load(
    f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")
stats = json.load(
    open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))


for sample_size in RANDOM_SAMPLE_UNLEARNING_SIZES:
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        remaining_indexes = torch.load(
            f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")

        # LOAD DATASET
        reamining_dataset = Subset(train_dataset, remaining_indexes)

        reamining_dloader = DataLoader(reamining_dataset, batch_size=BATCH_SIZE,
                                       collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        test_dloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                  collate_fn=custom_collate_fn, num_workers=24)

        # model
        model = LitHigherOrderLSTM(stats['vocab_size'],  stats['users_size'], EMBEDDING_SIZE,
                                   HIDDEN_SIZE, NUMBER_OF_LAYERS, DROPOUT)

        # Configure the EarlyStopping callback
        early_stop_callback = EarlyStopping(
            monitor='val_loss',  # Metric to monitor
            min_delta=0.00,  # Minimum change to qualify as an improvement
            patience=3,  # Number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'  # Because we want to minimize validation loss
        )

        CHECKPOINT_DIR = f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/checkpoints/"
        trainer = pl.Trainer(accelerator="gpu", devices=[
            0], max_epochs=MAX_EPOCHS, enable_progress_bar=True, enable_checkpointing=True, default_root_dir=CHECKPOINT_DIR, callbacks=[early_stop_callback])

        # train the model
        logging.info('Training the model for the remaining data, unlearning sample size: %d, sample: %d' % (
            sample_size, i))
        trainer.fit(model, reamining_dloader, test_dloader)

        os.makedirs(
            f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}", exist_ok=True)

        # save the model
        torch.save(model,
                   f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retrained_{MODEL_NAME}_model.pt")
        logging.info('Model is saved for the remaining data, unlearning sample size: %d, sample: %d' % (
            sample_size, i))
