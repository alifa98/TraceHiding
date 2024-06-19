# This file train a GRU model to predict the user of the sequence of trajectory data points
# basically, this file train the smart teacher or the orginal model
# model parameters will be saved next to the model in josn format

import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.functions import custom_collate_fn
from models.HoGRU import LitHigherOrderGRU
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import BertConfig, BertForSequenceClassification
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Dataset
import json
import torch



os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


MODEL_NAME = "BERT"  # model name that we are training in this file
# train on
DATASET_NAME = "nyc_checkins"

# MODEL PARAMETERS
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE = 300
MAX_POSITION_EMBEDDINGS = 512
BATCH_SIZE = 100
EPOCHS = 10


os.makedirs(
    f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/", exist_ok=True)

train_dataset = torch.load(
    f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
test_dataset = torch.load(
    f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")
cell_to_id = torch.load(
    f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_cell_to_id.pt")
stats = json.load(
    open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))


# LOAD DATASET
class HexagonDatasetForBert(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


train_dataset = HexagonDatasetForBert(train_dataset[0], train_dataset[1])
test_dataset = HexagonDatasetForBert(test_dataset[0], test_dataset[1])


# model
config = BertConfig(
    vocab_size=int(stats["vocab_size"]),
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    num_labels=int(stats["users_size"]),
)

model = BertForSequenceClassification(config)


CHECKPOINT_DIR = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/checkpoints"

# Define training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="epoch",  # Use eval_strategy instead of evaluation_strategy
    save_strategy="epoch",  # Ensure save_strategy matches eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# save initial model
torch.save(
    model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/initial_{MODEL_NAME}_model.pt")


# Train the model
trainer.train()
logging.info("Training Done!")


logging.info("Saving model ...")

# save the model
torch.save(
    model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt")


# model parameters
model_params = {
    "hidden_size": HIDDEN_SIZE,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "num_attention_heads": NUM_ATTENTION_HEADS,
    "intermediate_size": INTERMEDIATE_SIZE,
    "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
}

json.dump(model_params, open(
    f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "w"))

logging.info("Training completed")
