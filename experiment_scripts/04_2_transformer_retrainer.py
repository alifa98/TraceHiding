import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.ArguemntParser import get_args
from utility.Dataset import CustomDataset
from utility.ArguemntParser import get_args
from utility.evaluationUtils import compute_metrics
from utility.functions import custom_collator_transformer
from transformers import ModernBertConfig, ModernBertForSequenceClassification, BertConfig, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import concurrent.futures
import logging
import torch
import time
import json

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

args = get_args()
MODEL_NAME = args.model
DATASET_NAME = args.dataset
SAMPLE_SIZE = args.sampleSize
SCENARIO = args.scenario
BIASED_SAMPLE_IMPORTANCE_NAME = args.biased # if it is None, then the sample is not biased
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

model_params = json.load(open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "r"))

HIDDEN_SIZE = model_params["hidden_size"]
NUMBER_OF_HIDDEN_LAYERS = model_params["number_of_layers"]
NUMBER_OF_ATTEN_HEADS = model_params["number_of_heads"]
INTERMEDIATE_SIZE = model_params["intermediate_size"]
BATCH_SIZE = model_params["batch_size"]
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 10

torch.set_float32_matmul_precision('high')

os.makedirs(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/", exist_ok=True)

stats = json.load(open(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_stats.json", "r"))
num_labels = stats["users_size"]
vocab_size = stats["vocab_size"]
pad_token_id = 0

# Load the dataset
train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

def tokenize_function(item):
        return {"input_ids": item[0], "attention_mask": [1] * len(item[0]), "labels": item[1]}
    
with concurrent.futures.ProcessPoolExecutor() as executor:
    logging.info("Tokenizing Training Data")
    train_data = list(executor.map(tokenize_function, train_data))
    logging.info("Tokenizing Test Data")
    test_data = list(executor.map(tokenize_function, test_data))

for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
    
    # base folder
    base_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample{f"_biased_{BIASED_SAMPLE_IMPORTANCE_NAME}" if BIASED_SAMPLE_IMPORTANCE_NAME else ""}/sample_size_{SAMPLE_SIZE}/sample_{i}"
    
    remaining_indexes = torch.load(f"{base_folder}/data/remaining.indexes.pt", weights_only=False)
    reamining_dataset = Subset(train_data, remaining_indexes)
    remaining_dataset = CustomDataset(reamining_dataset)
    test_dataset = CustomDataset(test_data)

    if MODEL_NAME == "ModernBERT":
        config = ModernBertConfig(
            vocab_size=vocab_size + 1, # for the padding token
            pad_token_id=pad_token_id,
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=NUMBER_OF_HIDDEN_LAYERS,
            num_attention_heads=NUMBER_OF_ATTEN_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            max_position_embeddings=300, # defined when preprocessing the dataset and having max_length=300
            num_labels=num_labels
        )
        model = ModernBertForSequenceClassification(config)
    elif MODEL_NAME == "BERT":
        config = BertConfig(
            vocab_size=vocab_size + 1, # for the padding token
            pad_token_id=pad_token_id,
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=NUMBER_OF_HIDDEN_LAYERS,
            num_attention_heads=NUMBER_OF_ATTEN_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            max_position_embeddings=300, # defined when preprocessing the dataset and having max_length=300
            num_labels=num_labels
        )
        model = BertForSequenceClassification(config)
    else:
        raise ValueError("Unknown model name")

    results_folder = f"{base_folder}/{MODEL_NAME}/retraining"
    os.makedirs(results_folder, exist_ok=True)
    
    CHECKPOINT_DIR = f"{results_folder}/checkpoints/"

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=MAX_EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=remaining_dataset,
        eval_dataset=test_dataset,
        data_collator=custom_collator_transformer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        compute_metrics=compute_metrics
    )

    training_start_time = time.time()
    trainer.train()
    training_end_time = time.time()
    logging.info("Retraining of the model is done.")

    logging.info(f"Training time: {training_end_time - training_start_time}")

    logging.info("Saving the loaded model ...")
    trainer.save_model(f"{results_folder}/retrained_{MODEL_NAME}_model.pt")

    logging.info(f"Model is now saved in: {results_folder}/retrained_{MODEL_NAME}_model.pt")

    # model parameters
    retrained_stats = {
        "trained_epochs": trainer.state.epoch,
        "training_time": training_end_time - training_start_time
    }
    json.dump(retrained_stats, open(f"{results_folder}/retrained_{MODEL_NAME}_model.json", "w"))
    
    logging.info(f"Retraining stats are saved in: {results_folder}/retrained_{MODEL_NAME}_model.json")

    logging.info(f"The model For {SCENARIO} sample unlearning of sample {i} of size {SAMPLE_SIZE} retrained successfully.")