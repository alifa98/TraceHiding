import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import BertConfig, BertForSequenceClassification
from torch.utils.data import Dataset
import json
import torch
import wandb

os.environ["WANDB_MODE"] = "disabled"
# wandb.init(mode="dryrun")

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

MODEL_NAME = "BERT"
# DATASET_NAME = "HO_Rome_Res8"
DATASET_NAME = "Ho_Foursquare_NYC"
# DATASET_NAME = "HO_Geolife_Res8"

HIDDEN_SIZE = 100
NUM_HIDDEN_LAYERS = 3
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE = 128
BATCH_SIZE = 20
EPOCHS = 300

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

os.makedirs(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/", exist_ok=True)

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)
cell_to_id = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_cell_to_id.pt", weights_only=False)
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATASET
class HexagonDatasetForBert(Dataset):
    def __init__(self, sequences, lables):
        self.sequences = sequences
        self.labels = lables
        self.max_length = 0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.padded_sequences[idx], dtype=torch.long),
            'attention_mask': torch.tensor([1 if token != 0 else 0 for token in self.padded_sequences[idx]], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def pad_a_sequence(self, sequence):
        if len(sequence) < self.max_length:
            return sequence + [0] * (self.max_length - len(sequence))
        return sequence[:self.max_length]
    
    def pad_sequences(self):
        self.padded_sequences = [self.pad_a_sequence(seq) for seq in self.sequences]


sequences, labels = zip(*train_dataset)
train_dataset = HexagonDatasetForBert(sequences, labels)
sequences, labels = zip(*test_dataset)
test_dataset = HexagonDatasetForBert(sequences, labels)

max_sequence_length = max(max(len(seq) for seq in test_dataset.sequences), max(len(seq) for seq in train_dataset.sequences))
train_dataset.max_length = max_sequence_length
test_dataset.max_length = max_sequence_length

train_dataset.pad_sequences()
test_dataset.pad_sequences()

# model
config = BertConfig(
    vocab_size=stats["vocab_size"]+2,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=max_sequence_length,
    num_labels=int(stats["users_size"]),
)

model = BertForSequenceClassification(config)

# Move the model to the specified device
model.to(device)

CHECKPOINT_DIR = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/checkpoints"

# Define training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    run_name=f"{MODEL_NAME}_{DATASET_NAME}",
    eval_strategy="epoch",  # Use eval_strategy instead of evaluation_strategy
    save_strategy="epoch",  # Ensure save_strategy matches eval_strategy
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.02,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to=None,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
)

# Save initial model
torch.save(model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/initial_{MODEL_NAME}_model.pt")

# Train the model
trainer.train()
logging.info("Training Done!")

logging.info("Saving model ...")

# Save the model
torch.save(model, f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt")

# Model parameters
model_params = {
    "hidden_size": HIDDEN_SIZE,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "num_attention_heads": NUM_ATTENTION_HEADS,
    "intermediate_size": INTERMEDIATE_SIZE,
    "max_position_embeddings": max_sequence_length,
    "batch_size": BATCH_SIZE,
}

json.dump(model_params, open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "w"))

logging.info("Training completed")
