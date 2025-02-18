import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import logging
from transformers import PreTrainedTokenizerFast
from transformers import ModernBertConfig, ModernBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from utility.functions import compute_metrics_bert
from datasets import load_dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

# Check if the expected number of arguments is provided
if len(sys.argv) < 3:
    print("Usage: python experiment_scripts/02_5_modernBERT_trainer.py <model_type> <dataset_name>")
    sys.exit(1)
    
MODEL_NAME = sys.argv[1]
DATASET_NAME = sys.argv[2]

training_configs = {
    "HO_Rome_Res8": {
        "hidden_size": 128,
        "number_of_layers": 4,
        "number_of_heads": 4,
        "intermediate_size": 3072,
        "batch_size": 10,
    },
    "HO_Porto_Res8": {
    },
    "HO_Geolife_Res8": {
    },
    "HO_NYC_Res9": {
    }
}

# MODEL PARAMETERS
HIDDEN_SIZE = training_configs[DATASET_NAME]["hidden_size"]
NUMBER_OF_HIDDEN_LAYERS = training_configs[DATASET_NAME]["number_of_layers"]
NUMBER_OF_ATTEN_HEADS = training_configs[DATASET_NAME]["number_of_heads"]
INTERMEDIATE_SIZE = training_configs[DATASET_NAME]["intermediate_size"]
BATCH_SIZE = training_configs[DATASET_NAME]["batch_size"]
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 10
# ------------------------------------- END CONFIGURATIONS -------------------------------------#
torch.set_float32_matmul_precision('high')

tokenizer = PreTrainedTokenizerFast.from_pretrained(f"experiments/{DATASET_NAME}/saved_models/ho-sequence-tokenizer")

os.makedirs(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/", exist_ok=True)

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)
stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))
num_labels = stats["users_size"]

def tokenize_function(example):
    seq, label = example[0], example[1]
    tokens = tokenizer(
        seq,
        truncation=True,
        padding=True,
        max_length=300
    )
    tokens["labels"] = label
    return tokens

train_dataset = train_dataset.map(tokenize_function)
test_dataset = test_dataset.map(tokenize_function)

if MODEL_NAME == "ModernBERT":
    config = ModernBertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUMBER_OF_HIDDEN_LAYERS,
        num_attention_heads=NUMBER_OF_ATTEN_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=300, # defined when preprocessing the dataset and having max_length=300
        num_labels=num_labels
    )

    model = ModernBertForSequenceClassification(config)
elif MODEL_NAME == "Bert":
    raise NotImplementedError("Bert model is not implemented yet.")
else:
    raise ValueError("Unknown model name")


CHECKPOINT_DIR = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/checkpoints"

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    # batch_eval_metrics=True,
    # per_device_eval_batch_size=BATCH_SIZE, # compute metrics on the whole batch
    num_train_epochs=MAX_EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1, # best and the most recent one (might be the same)
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    compute_metrics=compute_metrics_bert
)

# save initial model
trainer.save_model(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/initial_{MODEL_NAME}_model.pt")

training_start_time = time.time()
trainer.train()
training_end_time = time.time()
logging.info("Training completed")

logging.info(f"Training time: {training_end_time - training_start_time}")

logging.info("Saving the loaded model ...")
trainer.save_model(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt")

# test the best model (it loads the best model automatically)
logging.info("Testing the best model ...")
test_results = trainer.evaluate(eval_dataset=test_dataset)
logging.info(f"Test result: {test_results}")

logging.info("Saving the model stats ...")

model_params = {
    "hidden_size": HIDDEN_SIZE,
    "number_of_layers": NUMBER_OF_HIDDEN_LAYERS,
    "number_of_heads": NUMBER_OF_ATTEN_HEADS,
    "batch_size": BATCH_SIZE,
    "trained_epochs": trainer.state.epoch,
    "test_result": test_results,
    "training_time": training_end_time - training_start_time
}
json.dump(model_params, open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "w"))

logging.info("Model and its stats is saved.")