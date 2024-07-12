import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility.functions import custom_collate_fn
from torch.nn import functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import logging
import torch
import time
import json
import tqdm
import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

DATASET_NAME = "nyc_checkins"
MODEL_NAME = "LSTM"

RANDOM_SAMPLE_UNLEARNING_SIZES = [10, 20, 50, 100, 200, 300, 600, 1000]
FINETUNING_BATCH_SIZES = [5, 10, 20, 30, 40, 40, 50, 100]
PORTION_OF_FINE_TUNING_DATA = 0.5
FINE_TUNING_EPOCHS = 25
FINE_TUNING_LEARNING_RATE = 5*1e-5

# ------------------------------------- END CONFIGURATIONS -------------------------------------#
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for sample_size, batch_size in zip(RANDOM_SAMPLE_UNLEARNING_SIZES, FINETUNING_BATCH_SIZES):
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):

        ## create a new wandb run
        wandb.init(
            project="thesis_unlearning",
            job_type="baseline",
            name=f"finetune-{DATASET_NAME}-{MODEL_NAME}-sample_size_{sample_size}-repetition_{i}",
            config={
                "method_name": "Finetune",
                "dataset": DATASET_NAME,
                "model": MODEL_NAME,
                "scenario": "Sample Deletion",
                "sample_size": sample_size,
                "batch_size": batch_size,
                "repetition": i,
                "learning_rate": FINE_TUNING_LEARNING_RATE,
                "portion_of_fine_tuning_data": PORTION_OF_FINE_TUNING_DATA,
            }
        )

        # results folder
        os.makedirs(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/finetune/", exist_ok=True)

        remaining_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")
        unlearning_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/unlearning.indexes.pt")
        
        train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
        test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")

        # select the remaining data
        remaining_dataset = Subset(train_data, remaining_indices)
        unlearning_dataset = Subset(train_data, unlearning_indices)

        # select the random portion of the remaining data for fine-tuning
        remaining_dataset = Subset(remaining_dataset, torch.randperm(len(remaining_dataset))[:int(PORTION_OF_FINE_TUNING_DATA*len(remaining_dataset))])
        
        remaining_dloader = DataLoader(remaining_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        unlearning_dloader = DataLoader(unlearning_dataset, batch_size=len(unlearning_dataset), collate_fn=custom_collate_fn, num_workers=24)
        test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate_fn, num_workers=24)

        model = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt").to(device)
        
        model.train()
        
        unlearning_stats = {}
        # Unlearning process
        for unlearning_epoch in range(FINE_TUNING_EPOCHS):
            epoch_stats = {}
            start_epoch_time = time.time()
            total_epoch_loss = 0
            pbar = tqdm.tqdm(remaining_dloader, desc=f"Unlearning Epoch: {unlearning_epoch}")
            for remaining_batch in pbar:
                model.train()

                x_remaining, y_remaining = remaining_batch
                x_remaining, y_remaining = x_remaining.to(device), y_remaining.to(device)

                # define optimizer
                optimizer = model.configure_optimizers()

                # set the learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = FINE_TUNING_LEARNING_RATE

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
            wandb.log(epoch_stats | {"unlearning_loss": total_epoch_loss})
            
            # save unlearning model for this epoch
            torch.save(model, f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/finetune/unlearned_epoch{unlearning_epoch}_{MODEL_NAME}_model.pt")
            
            unlearning_stats[unlearning_epoch] = epoch_stats
            
        wandb.finish()
        
        json.dump(unlearning_stats, open(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/finetune/unlearning_stats-batch_size_{batch_size}.json", "w"))
        logging.info(f"Unlearning models for each epoch now are saved for sample size {sample_size}, no. {i}")
