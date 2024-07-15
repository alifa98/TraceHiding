import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.functions import custom_collate_fn
from utility.EntropyImportance import EntropyImportance
from utility.ImportanceCalculator import ImportanceCalculator
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
os.environ["WANDB_MODE"] = "disabled" 

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

DATASET_NAME = "HO_NYC_Checkins"
MODEL_NAME = "LSTM"
IMPORTANCE_NAME = "entropy"

RANDOM_SAMPLE_UNLEARNING_SIZES = [600] #[10, 50, 100, 200, 300, 600, 1000]
UNLEARNING_BATCH_SIZE_FOR_EACH_SAMPLE_SIZE = [100] #[10, 25, 50, 100, 100, 120, 125] # /2

MAX_UNLEARNING_EPOCHS = 25
LEARNING_RATE = 5e-5
ALPHA = 0.9
BETA = 3.5
GAMMA = 0.1

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for sample_size, batch_size in zip(RANDOM_SAMPLE_UNLEARNING_SIZES, UNLEARNING_BATCH_SIZE_FOR_EACH_SAMPLE_SIZE):
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        
        ## create a new wandb run
        wandb.init(
            project="thesis_unlearning",
            job_type="unlearning",
            name=f"unlearning-{DATASET_NAME}-{MODEL_NAME}-{IMPORTANCE_NAME}-sample_size_{sample_size}-repetition_{i}",
            config={
                "method_name": "TraceHiding",
                "dataset": DATASET_NAME,
                "model": MODEL_NAME,
                "scenario": "Sample Deletion",
                "importance": IMPORTANCE_NAME,
                "sample_size": sample_size,
                "batch_size": batch_size,
                "repetition": i,
                "learning_rate": LEARNING_RATE,
                "alpha": ALPHA,
                "gamma": GAMMA
            }
        )
        
        # results folder
        os.makedirs(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/our_method/{IMPORTANCE_NAME}", exist_ok=True)

        unlearning_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/unlearning.indexes.pt")
        remaining_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")

        train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
        test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")

        # prepare importance calculator
        if IMPORTANCE_NAME == "entropy":
            importance_calculator = EntropyImportance()
        else:
            importance_calculator = ImportanceCalculator()
        importance_calculator.prepare(train_data + test_data)

        # creating training, unlearning and testing data loaders
        unlearning_dataset = Subset(train_data, unlearning_indices)
        remaining_dataset = Subset(train_data, remaining_indices)

        unlearning_dloader = DataLoader(unlearning_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        remaining_dloader = DataLoader(remaining_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate_fn, num_workers=24)

        # Load teacher and student models
        smart_teacher = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt").to(device)
        student = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt").to(device)

        
        retrained_model = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retraining/retrained_{MODEL_NAME}_model.pt").to(device)

        smart_teacher.eval()
        student.train()
        
        retrained_model.eval()
        

        unlearning_stats = {}
        # Unlearning process
        for unlearning_epoch in range(MAX_UNLEARNING_EPOCHS):
            
            #re-shuffle the remaining data
            remaining_dloader = DataLoader(remaining_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
            
            epoch_stats = {}
            start_epoch_time = time.time()
            total_epoch_remaining_loss = 0
            total_epoch_unlearning_loss = 0
            for unlearning_batch, remaining_batch in tqdm.tqdm(zip(unlearning_dloader, remaining_dloader)):
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
                
                unlearning_loss = BETA * unlearning_loss * torch.tensor(importance_calculator.calculate_importance(unlearning_batch)).to(device)
                
                loss = -1 * torch.clamp(unlearning_loss.mean(), min=0.0) # negative of the unlearning loss
                loss.backward()
                optimizer.step()
                total_epoch_unlearning_loss += abs(loss.item())
                
                # Zero the parameter gradients again
                optimizer.zero_grad()
                
                # doing the min step (remembering) [minimizing on the actual loss]
                student_remember_output = student(x_remaining)
                remembering_loss = F.kl_div(
                    F.log_softmax(y_hat_remaining, dim=1),
                    F.log_softmax(student_remember_output, dim=1), reduction='none', log_target=True
                ).sum(dim=1)

                cross_entropy_loss = F.cross_entropy(student_remember_output, y_remaining)
                
                loss = ALPHA * torch.clamp(remembering_loss.mean(), min=0.0) + GAMMA * cross_entropy_loss

                loss.backward()
                optimizer.step()
                total_epoch_remaining_loss += loss.item()
            

            end_epoch_time = time.time()

            student.eval()

            logging.info("-------------------------------------------------")
            logging.info(f"Epoch: {unlearning_epoch}")
            logging.info(f"Total Unlearning Loss: {total_epoch_unlearning_loss}, Total Remaining Loss: {total_epoch_remaining_loss}, Total Loss: {total_epoch_remaining_loss + total_epoch_unlearning_loss}") 
            student_accuracy_1, student_accuracy_3, student_accuracy_5, student_precision, student_recall, student_f1 = student.test_model(unlearning_dloader)
            retrained_accuracy_1, retrained_accuracy_3, retrained_accuracy_5, retrained_precision, retrained_recall, retrained_f1 = retrained_model.test_model(unlearning_dloader)
            epoch_stats["unlearning_student_accuracy_1"] = student_accuracy_1.item()
            epoch_stats["unlearning_student_accuracy_3"] = student_accuracy_3.item()
            epoch_stats["unlearning_student_accuracy_5"] = student_accuracy_5.item()
            epoch_stats["unlearning_student_precision"] = student_precision.item()
            epoch_stats["unlearning_student_recall"] = student_recall.item()
            epoch_stats["unlearning_student_f1"] = student_f1.item()
            epoch_stats["unlearning_retrained_accuracy_1"] = retrained_accuracy_1.item()
            epoch_stats["unlearning_retrained_accuracy_3"] = retrained_accuracy_3.item()
            epoch_stats["unlearning_retrained_accuracy_5"] = retrained_accuracy_5.item()
            epoch_stats["unlearning_retrained_precision"] = retrained_precision.item()
            epoch_stats["unlearning_retrained_recall"] = retrained_recall.item()
            epoch_stats["unlearning_retrained_f1"] = retrained_f1.item()
           
            logging.info(f"Student Unlearning Accuracy@1: {student_accuracy_1}")
            logging.info(f"Retrained Unlearning Accuracy@1: {retrained_accuracy_1}")
            
            student_accuracy_1, student_accuracy_3, student_accuracy_5, student_precision, student_recall, student_f1 = student.test_model(test_dloader)
            retrained_accuracy_1, retrained_accuracy_3, retrained_accuracy_5, retrained_precision, retrained_recall, retrained_f1 = retrained_model.test_model(test_dloader)
            epoch_stats["test_student_accuracy_1"] = student_accuracy_1.item()
            epoch_stats["test_student_accuracy_3"] = student_accuracy_3.item()
            epoch_stats["test_student_accuracy_5"] = student_accuracy_5.item()
            epoch_stats["test_student_precision"] = student_precision.item()
            epoch_stats["test_student_recall"] = student_recall.item()
            epoch_stats["test_student_f1"] = student_f1.item()
            epoch_stats["test_retrained_accuracy_1"] = retrained_accuracy_1.item()
            epoch_stats["test_retrained_accuracy_3"] = retrained_accuracy_3.item()
            epoch_stats["test_retrained_accuracy_5"] = retrained_accuracy_5.item()
            epoch_stats["test_retrained_precision"] = retrained_precision.item()
            epoch_stats["test_retrained_recall"] = retrained_recall.item()
            epoch_stats["test_retrained_f1"] = retrained_f1.item()
            
            logging.info(f"Student Test Accuracy@1: {student_accuracy_1}")
            logging.info(f"Retrained Test Accuracy@1: {retrained_accuracy_1}")
            
            epoch_stats["epoch_time"] = end_epoch_time - start_epoch_time
            
            #wand logging
            wandb.log(epoch_stats | {"loss": total_epoch_remaining_loss + total_epoch_unlearning_loss})
            
            # save unlearning model for this epoch
            torch.save(student, f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/our_method/{IMPORTANCE_NAME}/unlearned_epoch{unlearning_epoch}_{MODEL_NAME}_model.pt")
            
            unlearning_stats[unlearning_epoch] = epoch_stats
            
        wandb.finish()
        
        json.dump(unlearning_stats, open(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/our_method/{IMPORTANCE_NAME}/unlearning_stats-batch_size_{batch_size}.json", "w"))
        logging.info(f"Unlearning models for each epoch now are saved for sample size {sample_size}, no. {i}")
