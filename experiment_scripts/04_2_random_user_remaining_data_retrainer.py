import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import Subset
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utility.functions import custom_collate_fn
from utility.AccuracyCallback import EachBatchTester
from models.HoLSTM import LitHigherOrderLSTM
from models.HoGRU import LitHigherOrderGRU
from pytorch_lightning.callbacks import EarlyStopping
import json
import torch


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# ------------------------------------- START CONFIGURATIONS -------------------------------------#

MODEL_NAME = "GRU"
DATASET_NAME = "HO_NYC_Checkins"
UNLEANING_USERS = [23, 89, 164, 174, 200] # created folders by random sampler for users

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

# load model parameters
model_params = json.load(open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "r"))

EMBEDDING_SIZE = model_params["embedding_size"]
HIDDEN_SIZE = model_params["hidden_size"]
NUMBER_OF_LAYERS = model_params["number_of_layers"]
DROPOUT = model_params["dropout"]
BATCH_SIZE = model_params["batch_size"]
MAX_EPOCHS = 100

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))


for user_id in UNLEANING_USERS:
    remaining_indexes = torch.load(f"experiments/{DATASET_NAME}/unlearning/user_sample/user_{user_id}/data/remaining.indexes.pt")

    # Select the remaining data from the training dataset
    reamining_dataset = Subset(train_dataset, remaining_indexes)

    reamining_dloader = DataLoader(reamining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
    test_dloader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=custom_collate_fn, num_workers=24)

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
        patience=3,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'  # Because we want to minimize validation loss
    )
    
    os.makedirs(f"experiments/{DATASET_NAME}/unlearning/user_sample/user_{user_id}/{MODEL_NAME}/retraining", exist_ok=True)
    
    EACH_BATCH_STATS_FILE_PATH = f"experiments/{DATASET_NAME}/unlearning/user_sample/user_{user_id}/{MODEL_NAME}/retraining/retrained_each_batch_stats.json"
    batch_tester_callback = EachBatchTester(test_dloader, EACH_BATCH_STATS_FILE_PATH) # for AIN calculation (for class forgetting)

    CHECKPOINT_DIR = f"experiments/{DATASET_NAME}/unlearning/user_sample/user_{user_id}/{MODEL_NAME}/retraining/checkpoints/"
    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCHS, enable_progress_bar=True, enable_checkpointing=True, default_root_dir=CHECKPOINT_DIR, callbacks=[early_stop_callback, batch_tester_callback])

    # train the model
    logging.info(f'Training the model for the remaining data, user: {user_id}')
    trainer.fit(model, reamining_dloader, test_dloader)

    # save the model
    torch.save(model, f"experiments/{DATASET_NAME}/unlearning/user_sample/user_{user_id}/{MODEL_NAME}/retraining/retrained_{MODEL_NAME}_model.pt")
    logging.info(f'Model is saved for user: {user_id}')
