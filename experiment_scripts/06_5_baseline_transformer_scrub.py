import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility.Dataset import CustomDataset
from utility.ArguemntParser import get_args
from utility.evaluationUtils import compute_metrics, get_model_outputs
from utility.functions import check_stopping_criteria, custom_collator_transformer
from transformers import ModernBertForSequenceClassification, BertForSequenceClassification
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import concurrent.futures
import logging
import torch
import time
import json
import tqdm
import wandb

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
UNLEARNING_EPOCHS = args.epochs
BIASED_SAMPLE_IMPORTANCE_NAME = args.biased # if it is None, then the sample is not biased
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

LEARNING_RATE = 5e-5

# SCRUB parameters (see the paper for more details)
ALPHA = 0.9 # the cofficient for the forgetting loss term (student-teacher output)
GAMMA = 0.1 # the cofficient for the cross-entropy loss term (student-remaining data)

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):

    wandb.init(
        project="Thesis",
        job_type="baseline",
        name=f"Scrub-{DATASET_NAME}-{MODEL_NAME}-{SCENARIO}_unlearning-sample_size_{SAMPLE_SIZE}-repetition_{i}",
        config={
            "method_name": "SCRUB",
            "dataset": DATASET_NAME,
            "model": MODEL_NAME,
            "scenario": SCENARIO + " deletion",
            "sample_size": SAMPLE_SIZE,
            "batch_size": BATCH_SIZE,
            "bias": BIASED_SAMPLE_IMPORTANCE_NAME,
            "is_biased": BIASED_SAMPLE_IMPORTANCE_NAME is not None,
            "repetition": i,
            "learning_rate": LEARNING_RATE,
            "alpha": ALPHA,
            "gamma": GAMMA
        }
    )
    
    # base folder
    base_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample{f"_biased_{BIASED_SAMPLE_IMPORTANCE_NAME}" if BIASED_SAMPLE_IMPORTANCE_NAME else ""}/sample_size_{SAMPLE_SIZE}/sample_{i}"
    
    # results folder
    results_folder = f"{base_folder}/{MODEL_NAME}/scrub"
    os.makedirs(results_folder, exist_ok=True)
    
    unlearning_indices = torch.load(f"{base_folder}/data/unlearning.indexes.pt", weights_only=False)
    remaining_indices = torch.load(f"{base_folder}/data/remaining.indexes.pt", weights_only=False)

    train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
    test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

    unlearning_dataset = Subset(train_data, unlearning_indices)
    remaining_dataset = Subset(train_data, remaining_indices)

    def tokenize_function(item):
        return {"input_ids": item[0], "attention_mask": [1] * len(item[0]), "labels": item[1]}
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        unlearning_dataset = list(executor.map(tokenize_function, unlearning_dataset))
        remaining_dataset = list(executor.map(tokenize_function, remaining_dataset))
        test_data = list(executor.map(tokenize_function, test_data))

    unlearning_dataset = CustomDataset(unlearning_dataset)
    remaining_dataset = CustomDataset(remaining_dataset)
    test_data = CustomDataset(test_data)
    
    unlearning_dloader = DataLoader(unlearning_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collator_transformer, shuffle=True, num_workers=24)
    remaining_dloader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collator_transformer, shuffle=True, num_workers=24)
    test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collator_transformer, num_workers=24)

    # Load the original models stats
    original_model_stats = json.load(open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "r"))
    original_test_accuracy = original_model_stats['test_result']["eval_accuracy"]

    # Load initial model as the bad teacher
    MODEL_PATH = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt"
    INITIAL_MODEL_PATH = f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/initial_{MODEL_NAME}_model.pt"
    if MODEL_NAME == "ModernBERT":
        smart_teacher = ModernBertForSequenceClassification.from_pretrained(MODEL_PATH)
        student = ModernBertForSequenceClassification.from_pretrained(MODEL_PATH)
    elif MODEL_NAME == "BERT":
        smart_teacher = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        student = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        raise ValueError("Model name is not correct")
    
    smart_teacher.to(device)
    student.to(device)
    
    smart_teacher.eval()
    student.train()
    
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE)
    unlearning_stats = {}
    
    # Unlearning process
    for unlearning_epoch in range(UNLEARNING_EPOCHS):
        
        #re-shuffle the remaining data
        remaining_dloader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collator_transformer, shuffle=True, num_workers=24)
        
        epoch_stats = {}
        start_epoch_time = time.time()
        total_epoch_loss = 0
        pbar = tqdm.tqdm(zip(unlearning_dloader, remaining_dloader), desc=f"Unlearning Epoch: {unlearning_epoch}")
        for unlearning_batch, remaining_batch in pbar:
            smart_teacher.eval()
            student.train()

            inputs_unlearning = {k: v.to(device) for k, v in unlearning_batch.items() if k != "labels"}
            labels_unlearning = unlearning_batch["labels"].to(device)
            
            inputs_remaining = {k: v.to(device) for k, v in remaining_batch.items() if k != "labels"}
            labels_remaining = remaining_batch["labels"].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            with torch.no_grad():
                y_hat_remaining = smart_teacher(**inputs_remaining)
                y_hat_unlearning =  smart_teacher(**inputs_unlearning)
                
            # doing the max step (forgetting) [minimizing on the negative of the unlearning loss]
            student_forget_output = student(**inputs_unlearning)

            unlearning_loss = F.kl_div(
                F.log_softmax(y_hat_unlearning.logits, dim=1),
                F.log_softmax(student_forget_output.logits, dim=1), reduction='none', log_target=True
            ).sum(dim=1)
            
            loss = -1 * torch.clamp(unlearning_loss.mean(), min=0.0) # negative of the unlearning loss
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            
            optimizer.zero_grad()
            
            # doing the min step
            student_remember_output = student(**inputs_remaining)
            remembering_loss = F.kl_div(
                F.log_softmax(y_hat_remaining.logits, dim=1),
                F.log_softmax(student_remember_output.logits, dim=1), reduction='none', log_target=True
            ).sum(dim=1)

            cross_entropy_loss = F.cross_entropy(student_remember_output.logits, labels_remaining)
            
            loss = ALPHA * torch.clamp(remembering_loss.mean(), min=0.0) + GAMMA * cross_entropy_loss

            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

        end_epoch_time = time.time()

        student.eval()

        epoch_stats["epoch_time"] = end_epoch_time - start_epoch_time
        
        unlearning_eval_results = compute_metrics(get_model_outputs(student, unlearning_dloader, device))
        test_eval_results = compute_metrics(get_model_outputs(student, test_dloader, device))
        
        unlearning_data_accuracy = unlearning_eval_results['accuracy_1']
        
        wandb.log(epoch_stats | {"loss": total_epoch_loss} | {"unlearning_" + k: v for k, v in unlearning_eval_results.items() } | {"test_" + k: v for k, v in test_eval_results.items() })
        
        # Save unlearning model for this epoch
        student.save_pretrained(f"{results_folder}/unlearned_{MODEL_NAME}_epoch_{unlearning_epoch}_batch_{BATCH_SIZE}.pt")
        
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
