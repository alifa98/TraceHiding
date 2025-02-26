import os
import sys

import concurrent

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility.Dataset import CustomDataset
from utility.evaluationUtils import compute_metrics, get_model_outputs
from utility.functions import check_stopping_criteria, custom_collator_transformer
from utility.ArguemntParser import get_args
from transformers import ModernBertForSequenceClassification, BertForSequenceClassification
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import concurrent.futures
import logging
import torch
import time
import json
import tqdm
import wandb

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# os.environ["WANDB_MODE"] = "offline" # then run wandb sync <path_to_your_wandb_run_directory> to sync the results

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

args = get_args()
MODEL_NAME = args.model
DATASET_NAME = args.dataset
SCENARIO = args.scenario
SAMPLE_SIZE =args.sampleSize
BATCH_SIZE = args.batchSize
FINE_TUNING_EPOCHS = args.epochs
BIASED_SAMPLE_IMPORTANCE_NAME = args.biased # if it is None, then the sample is not biased
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

PORTION_OF_FINE_TUNING_DATA = 0.3
FINE_TUNING_LEARNING_RATE = 5*1e-5

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    wandb.init(
        project="Thesis",
        job_type="baseline",
        name=f"Finetune-{DATASET_NAME}-{MODEL_NAME}-{SCENARIO}_unlearning-sample_size_{SAMPLE_SIZE}-repetition_{i}",
        config={
            "method_name": "Finetune",
            "dataset": DATASET_NAME,
            "model": MODEL_NAME,
            "scenario": SCENARIO + " deletion",
            "sample_size": SAMPLE_SIZE,
            "batch_size": BATCH_SIZE,
            "is_biased": BIASED_SAMPLE_IMPORTANCE_NAME is not None,
            "bias": BIASED_SAMPLE_IMPORTANCE_NAME,
            "repetition": i,
            "learning_rate": FINE_TUNING_LEARNING_RATE,
            "portion_of_fine_tuning_data": PORTION_OF_FINE_TUNING_DATA,
        }
    )
    
    # base folder
    base_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample{f"_biased_{BIASED_SAMPLE_IMPORTANCE_NAME}" if BIASED_SAMPLE_IMPORTANCE_NAME else ""}/sample_size_{SAMPLE_SIZE}/sample_{i}"
    
    # create results folder
    results_folder = f"{base_folder}/{MODEL_NAME}/finetune"
    os.makedirs(results_folder, exist_ok=True)

    unlearning_indices = torch.load(f"{base_folder}/data/unlearning.indexes.pt", weights_only=False)
    remaining_indices = torch.load(f"{base_folder}/data/remaining.indexes.pt", weights_only=False)

    unlearning_dataset = Subset(train_data, unlearning_indices)
    remaining_dataset = Subset(train_data, remaining_indices)

    # select the random portion of the remaining data for fine-tuning
    remaining_dataset = Subset(remaining_dataset, torch.randperm(len(remaining_dataset))[:int(PORTION_OF_FINE_TUNING_DATA * len(remaining_dataset))].tolist())
    
    unlearning_dataset = CustomDataset(unlearning_dataset)
    remaining_dataset = CustomDataset(remaining_dataset)
    test_dataset = CustomDataset(test_data)
    
    unlearning_dloader = DataLoader(unlearning_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collator_transformer, shuffle=True, num_workers=48)
    remaining_dloader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collator_transformer, shuffle=True, num_workers=48)
    test_dloader = DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=custom_collator_transformer, num_workers=48)

    # Load the original models stats
    original_model_stats = json.load(open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "r"))
    original_test_accuracy = original_model_stats['test_result']["eval_accuracy"]
    
    MODEL_PATH = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt"
    if MODEL_NAME == "ModernBERT":
        model = ModernBertForSequenceClassification.from_pretrained(MODEL_PATH)
    elif MODEL_NAME == "BERT":
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        raise ValueError("Model name is not correct")
    
    model.to(device)
    
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=FINE_TUNING_LEARNING_RATE)
    unlearning_stats = {}
    
    for unlearning_epoch in range(FINE_TUNING_EPOCHS):
        epoch_stats = {}
        start_epoch_time = time.time()
        total_epoch_loss = 0
        pbar = tqdm.tqdm(remaining_dloader, desc=f"Unlearning Epoch: {unlearning_epoch}")
        for remaining_batch in pbar:
            model.train()

            inputs_remaining = {k: v.to(device) for k, v in remaining_batch.items() if k != "labels"}
            labels_remaining = remaining_batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(**inputs_remaining, labels=labels_remaining)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
        end_epoch_time = time.time()

        model.eval()

        epoch_stats["epoch_time"] = end_epoch_time - start_epoch_time
        
        unlearning_eval_results = compute_metrics(get_model_outputs(model, DataLoader(unlearning_dataset, batch_size=700, collate_fn=custom_collator_transformer, shuffle=True, num_workers=48), device))
        test_eval_results = compute_metrics(get_model_outputs(model, DataLoader(test_dataset, batch_size=700, collate_fn=custom_collator_transformer, shuffle=True, num_workers=48), device))
        
        unlearning_data_accuracy = unlearning_eval_results['accuracy_1']
        
        wandb.log(epoch_stats | {"loss": total_epoch_loss} | {"unlearning_" + k: v for k, v in unlearning_eval_results.items() } | {"test_" + k: v for k, v in test_eval_results.items() })
        
        # Save unlearning model for this epoch
        model.save_pretrained(f"{results_folder}/unlearned_{MODEL_NAME}_epoch_{unlearning_epoch}_batch_{BATCH_SIZE}.pt")
        
        unlearning_stats[unlearning_epoch] = epoch_stats
        
        # Stop Unlearning at this epoch
        if check_stopping_criteria(original_test_accuracy, unlearning_data_accuracy, delta=0.001):
            logging.info(f"Unlearning stopped early at epoch {unlearning_epoch} for sample size {SAMPLE_SIZE}, no. {i}")
            break
        
    else:
        logging.info(f"Unlearning finished all epochs for sample size {SAMPLE_SIZE}, no. {i}")
        
    wandb.finish()
    
    json.dump(unlearning_stats, open(f"{results_folder}/unlearning_stats-batch_size_{BATCH_SIZE}.json", "w"))
    logging.info(f"Unlearning models for each epoch now are saved for sample size {SAMPLE_SIZE}, no. {i}")
