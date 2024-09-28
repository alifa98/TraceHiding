import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility.functions import custom_collate_fn
from utility.ArguemntParser import get_args
from torch.nn import functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import logging
import torch
import time
import json
import tqdm
import wandb

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

args = get_args()
MODEL_NAME = args.model
DATASET_NAME = args.dataset
SCENARIO = args.scenario
SAMPLE_SIZE =args.sampleSize
BATCH_SIZE = args.batchSize
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5
PORTION_OF_FINE_TUNING_DATA = 0.3
FINE_TUNING_EPOCHS = 15
FINE_TUNING_LEARNING_RATE = 5*1e-5

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            "repetition": i,
            "learning_rate": FINE_TUNING_LEARNING_RATE,
            "portion_of_fine_tuning_data": PORTION_OF_FINE_TUNING_DATA,
        }
    )

    # results folder
    results_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample/sample_size_{SAMPLE_SIZE}/sample_{i}/{MODEL_NAME}/finetune"
    os.makedirs(results_folder, exist_ok=True)

    unlearning_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample/sample_size_{SAMPLE_SIZE}/sample_{i}/data/unlearning.indexes.pt", weights_only=False)
    remaining_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample/sample_size_{SAMPLE_SIZE}/sample_{i}/data/remaining.indexes.pt", weights_only=False)

    train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
    test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

    remaining_dataset = Subset(train_data, remaining_indices)
    unlearning_dataset = Subset(train_data, unlearning_indices)

    # select the random portion of the remaining data for fine-tuning
    remaining_dataset = Subset(remaining_dataset, torch.randperm(len(remaining_dataset))[:int(PORTION_OF_FINE_TUNING_DATA*len(remaining_dataset))])
    
    remaining_dloader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
    test_dloader = DataLoader(test_data, batch_size=len(test_data)//10, collate_fn=custom_collate_fn, num_workers=24)

    model = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt", weights_only=False).to(device)
    
    model.train()
    
    model.config_lr(FINE_TUNING_LEARNING_RATE)
    optimizer = model.configure_optimizers()
    
    unlearning_stats = {}
    
    for unlearning_epoch in range(FINE_TUNING_EPOCHS):
        epoch_stats = {}
        start_epoch_time = time.time()
        total_epoch_loss = 0
        pbar = tqdm.tqdm(remaining_dloader, desc=f"Unlearning Epoch: {unlearning_epoch}")
        for remaining_batch in pbar:
            model.train()

            x_remaining, y_remaining = remaining_batch
            x_remaining, y_remaining = x_remaining.to(device), y_remaining.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            model_output = model(x_remaining)
            loss = F.cross_entropy(model_output, y_remaining)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            
        end_epoch_time = time.time()

        model.eval()

        model_accuracy_1, model_accuracy_3, model_accuracy_5, model_precision, model_recall, model_f1 = model.test_model(unlearning_dloader)
        epoch_stats["unlearning_student_accuracy_1"] = model_accuracy_1.item()
        epoch_stats["unlearning_student_accuracy_3"] = model_accuracy_3.item()
        epoch_stats["unlearning_student_accuracy_5"] = model_accuracy_5.item()
        epoch_stats["unlearning_student_precision"] = model_precision.item()
        epoch_stats["unlearning_student_recall"] = model_recall.item()
        epoch_stats["unlearning_student_f1"] = model_f1.item()
        logging.info(f"Unlearning Dataset Accuracy@1: {model_accuracy_1}")
        
        model_accuracy_1, model_accuracy_3, model_accuracy_5, model_precision, model_recall, model_f1 = model.test_model(test_dloader)
        epoch_stats["test_student_accuracy_1"] = model_accuracy_1.item()
        epoch_stats["test_student_accuracy_3"] = model_accuracy_3.item()
        epoch_stats["test_student_accuracy_5"] = model_accuracy_5.item()
        epoch_stats["test_student_precision"] = model_precision.item()
        epoch_stats["test_student_recall"] = model_recall.item()
        epoch_stats["test_student_f1"] = model_f1.item()
        logging.info(f"Test Dataset Accuracy@1: {model_accuracy_1}")
        
        epoch_stats["unlearning_epoch_time"] = end_epoch_time - start_epoch_time
        
        #wand logging
        wandb.log(epoch_stats | {"loss": total_epoch_loss})
        
        # save unlearning model for this epoch
        torch.save(model, f"{results_folder}/unlearned_{MODEL_NAME}_epoch_{unlearning_epoch}_batch_{BATCH_SIZE}.pt")
        
        unlearning_stats[unlearning_epoch] = epoch_stats
        
    wandb.finish()
    
    json.dump(unlearning_stats, open(f"{results_folder}/unlearning_stats-batch_size_{BATCH_SIZE}.json", "w"))
    logging.info(f"Unlearning models for each epoch now are saved for sample size {SAMPLE_SIZE}, no. {i}")
