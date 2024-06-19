import os
from pathlib import Path
from models.HoLSTM import LitHigherOrderLSTM
import torch
import pytorch_lightning as pl
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from utility.CheckInDataset import HexagonCheckInUserDataset 
from pytorch_lightning.callbacks import EarlyStopping


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# DATASET PARAMETERS
SPLIT_RATIO = 0.8
DATASET_PATH = f"tul/TUL/dataset/nyc_HO_full.csv"

# MODEL PARAMETERS
RANDOM_STATE = 42
BATCH_SIZE = 50
MAX_EPOCHS = 300
EMBEDDING_SIZE = 80
HIDDEN_SIZE = 50
NUM_LAYERS = 1
DROPOUT = 0 # for one layer, dropout is not necessary

CHECKPOINT_DIR = f"checkpoints/LSTM/NYC_checkin_HO"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

dataset = HexagonCheckInUserDataset(
    DATASET_PATH,
    user_id_col_name="user",
    trajectory_col_name="poi",
    split_ratio=SPLIT_RATIO,
    random_state=RANDOM_STATE)

# Log dataset information
logging.info(f"Number of users: {dataset.users_size}")
logging.info(f"Number of cells: {dataset.vocab_size}")
logging.info(f"Number of training samples: {len(dataset)}")

# log model information
logging.info(f"Embedding size: {EMBEDDING_SIZE}")
logging.info(f"Hidden size: {HIDDEN_SIZE}")
logging.info(f"Number of layers: {NUM_LAYERS}")
logging.info(f"Dropout: {DROPOUT}")


# Create a custom collate function to pad the sequences
def custom_collate_fn(batch):
    #a batch is a list of tuples (sequence, label)
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq) for seq in sequences]
    labels = [torch.tensor(label) for label in labels]
    
    # Pad your sequences to the same length
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(labels)

train_dloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
test_dloader = DataLoader(dataset.get_test_data(), batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, num_workers=24)


model = LitHigherOrderLSTM(dataset.vocab_size,  dataset.users_size, EMBEDDING_SIZE,
                           HIDDEN_SIZE, NUM_LAYERS, DROPOUT)

#Configure the EarlyStopping callback
early_stop_callback = EarlyStopping(
   monitor='val_loss',  # Metric to monitor
   min_delta=0.00,  # Minimum change to qualify as an improvement
   patience=3,  # Number of epochs with no improvement after which training will be stopped
   verbose=True,
   mode='min'  # Because we want to minimize validation loss
)

trainer = pl.Trainer(accelerator="gpu", devices=[
                     0], max_epochs=MAX_EPOCHS, enable_progress_bar=True, enable_checkpointing=True, default_root_dir=CHECKPOINT_DIR, callbacks=[early_stop_callback])

# save initial model
torch.save(model, "saved_models/initial_model.pth")

trainer.fit(model, train_dloader, test_dloader)

# save trained model & cell to id mapping
torch.save(model, "saved_models/trained_model.pth")
torch.save(dataset.cell_to_id, "saved_models/cell_to_id.pth")
