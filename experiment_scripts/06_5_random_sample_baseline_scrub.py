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
UNLEARNING_BATCH_SIZE_FOR_EACH_SAMPLE_SIZE = [5, 10, 20, 50, 100, 150, 300, 500] # /2
LEARNING_RATE = 5e-5
UNLEARNING_EPOCHS = 15

# SCRUB parameters (see the paper for more details)
ALPHA = 0.5 # the cofficient for the forgetting loss term (student-teacher output)
GAMMA = 0.5 # the cofficient for the cross-entropy loss term (student-remaining data)

# ------------------------------------- END CONFIGURATIONS -------------------------------------#
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for sample_size, batch_size in zip(RANDOM_SAMPLE_UNLEARNING_SIZES, UNLEARNING_BATCH_SIZE_FOR_EACH_SAMPLE_SIZE):
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):

        wandb.init(
            project="thesis_unlearning",
            job_type="baseline",
            name=f"scrub-{DATASET_NAME}-{MODEL_NAME}-sample_size_{sample_size}-repetition_{i}",
            config={
                "method_name": "scrub",
                "dataset": DATASET_NAME,
                "model": MODEL_NAME,
                "sample_size": sample_size,
                "batch_size": batch_size,
                "repetition": i,
                "learning_rate": LEARNING_RATE,
                "epochs": UNLEARNING_EPOCHS,
                "alpha": ALPHA,
                "gamma": GAMMA
            }
        )

        # results folder
        os.makedirs(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/scrub/", exist_ok=True)

        unlearning_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/unlearning.indexes.pt")
        remaining_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")

        train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
        test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")

        # creating training, unlearning and testing data loaders
        unlearning_dataset = Subset(train_data, unlearning_indices)
        remaining_dataset = Subset(train_data, remaining_indices)

        unlearning_dloader = DataLoader(unlearning_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        remaining_dloader = DataLoader(remaining_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate_fn, num_workers=24)

        # Load initial model as the bad teacher
        smart_teacher = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt").to(device)
        student = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt").to(device)

        smart_teacher.eval()
        student.train()
        

        unlearning_stats = {}
        # Unlearning process
        for unlearning_epoch in range(UNLEARNING_EPOCHS):
            
            #re-shuffle the remaining data
            remaining_dloader = DataLoader(remaining_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
            
            epoch_stats = {}
            start_epoch_time = time.time()
            total_epoch_loss = 0
            pbar = tqdm.tqdm(zip(unlearning_dloader, remaining_dloader), desc=f"Unlearning Epoch: {unlearning_epoch}")
            for unlearning_batch, remaining_batch in pbar:
                smart_teacher.eval()
                student.train()

                x_unlearning, y_unlearning = unlearning_batch
                x_remaining, y_remaining = remaining_batch

                x_unlearning, y_unlearning = x_unlearning.to(device), y_unlearning.to(device)
                x_remaining, y_remaining = x_remaining.to(device), y_remaining.to(device)

                # define optimizer
                optimizer = student.configure_optimizers()

                # set the learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LEARNING_RATE

                # Zero the parameter gradients
                optimizer.zero_grad()

                with torch.no_grad():
                    y_hat_remaining = smart_teacher(x_remaining)
                    y_hat_unlearning = smart_teacher(x_unlearning)
                    
                # doing the max step (forgetting) [minimizing on the negative of the unlearning loss]
                student_forget_output = student(x_unlearning)

                unlearning_loss = F.kl_div(
                    F.log_softmax(y_hat_unlearning, dim=1),
                    F.log_softmax(student_forget_output, dim=1), reduction='none', log_target=True
                ).sum(dim=1)
                
                loss = -1 * torch.clamp(unlearning_loss.mean(), min=0.0) # negative of the unlearning loss
                loss.backward()
                optimizer.step()
                total_epoch_loss += loss.item()
                
                optimizer.zero_grad()
                
                # doing the min step
                student_remember_output = student(x_remaining)
                remembering_loss = F.kl_div(
                    F.log_softmax(y_hat_remaining, dim=1),
                    F.log_softmax(student_remember_output, dim=1), reduction='none', log_target=True
                ).sum(dim=1)

                cross_entropy_loss = F.cross_entropy(student_remember_output, y_remaining)
                
                loss = ALPHA * torch.clamp(remembering_loss.mean(), min=0.0) + GAMMA * cross_entropy_loss

                loss.backward()
                optimizer.step()
                total_epoch_loss += loss.item()

            end_epoch_time = time.time()

            student.eval()

            model_accuracy_1, model_accuracy_3, model_accuracy_5, model_precision, model_recall, model_f1 = student.test_model(unlearning_dloader)
            epoch_stats["unlearning_student_accuracy_1"] = model_accuracy_1.item()
            epoch_stats["unlearning_student_accuracy_3"] = model_accuracy_3.item()
            epoch_stats["unlearning_student_accuracy_5"] = model_accuracy_5.item()
            epoch_stats["unlearning_student_precision"] = model_precision.item()
            epoch_stats["unlearning_student_recall"] = model_recall.item()
            epoch_stats["unlearning_student_f1"] = model_f1.item()
            logging.info(f"Unlearning Dataset Accuracy@1: {model_accuracy_1}")
            
            model_accuracy_1, model_accuracy_3, model_accuracy_5, model_precision, model_recall, model_f1 = student.test_model(test_dloader)
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
            torch.save(student, f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/scrub/unlearned_epoch{unlearning_epoch}_{MODEL_NAME}_model.pt")
            
            unlearning_stats[unlearning_epoch] = epoch_stats
            
        wandb.finish()
        
        json.dump(unlearning_stats, open(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/scrub/unlearning_stats-batch_size_{batch_size}.json", "w"))
        logging.info(f"Unlearning models for each epoch now are saved for sample size {sample_size}, no. {i}")
