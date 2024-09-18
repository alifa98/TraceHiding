import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.ArguemntParser import get_args
from utility.functions import custom_collate_fn
from utility.ImportanceCalculator import ImportanceCalculator
from utility.EntropyImportance import EntropyImportance
from utility.CoverageDiversityImportance import CoverageDiversityImportance
from torch.nn import functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import logging
import torch
import time
import json
import tqdm
import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# os.environ["WANDB_MODE"] = "disabled"

# ------------------------------------- START CONFIGURATIONS -------------------------------------#
args = get_args()
MODEL_NAME = args.model
DATASET_NAME = args.dataset
IMPORTANCE_NAME = args.importance

RANDOM_SAMPLE_UNLEARNING_SIZES =[args.sampleSize] # Rome:135, Porto: 45700, Geolife: 50
UNLEARNING_BATCH_SIZE_FOR_EACH_SAMPLE_SIZE = [args.batchSize]
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

MAX_UNLEARNING_EPOCHS = 15
INITIAL_LEARNING_RATE = 1e-4
ALPHA = 0.9
BETA = 1
GAMMA = 0.1
FORGETTING_INITIAL_POWER = 0 # gives tendency to forget loss in lamda calculation (it can be dynamic maybe)

# ------------------------------------- END CONFIGURATIONS -------------------------------------#


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for sample_size, batch_size in zip(RANDOM_SAMPLE_UNLEARNING_SIZES, UNLEARNING_BATCH_SIZE_FOR_EACH_SAMPLE_SIZE):
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
        
        ## create a new wandb run
        wandb.init(
            project="Thesis",
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
                "learning_rate": INITIAL_LEARNING_RATE,
                "alpha": ALPHA,
                "gamma": GAMMA,
                "beta": BETA,
                "forgetting_initial_power": FORGETTING_INITIAL_POWER,
            }
        )
        
        # results folder
        results_folder = f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/our_method/{IMPORTANCE_NAME}"
        os.makedirs(results_folder, exist_ok=True)
        
        unlearning_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/unlearning.indexes.pt", weights_only=False)
        remaining_indices = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt", weights_only=False)

        train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
        test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

        # prepare importance calculator
        if IMPORTANCE_NAME == "entropy":
            importance_calculator = EntropyImportance()
        if IMPORTANCE_NAME == "coverage_diversity":
            importance_calculator = CoverageDiversityImportance()
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
        smart_teacher = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt", weights_only=False).to(device)
        student = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt", weights_only=False).to(device)

        # Perturb the student model
        # for param in student.parameters():
        #     param.data += 0.03 * torch.randn_like(param)

        retrained_model = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retraining/retrained_{MODEL_NAME}_model.pt", weights_only=False).to(device)

        smart_teacher.eval()
        student.train()
        
        retrained_model.eval()

        # Calculate initial unlearning and remembering losses without updating the model
        total_initial_unlearning_loss = 0
        total_initial_remaining_loss = 0

        with torch.no_grad():
            for unlearning_batch, remaining_batch in zip(unlearning_dloader, remaining_dloader):
                x_unlearning, _ = unlearning_batch
                x_remaining, y_remaining = remaining_batch

                x_unlearning = x_unlearning.to(device)
                x_remaining, y_remaining = x_remaining.to(device), y_remaining.to(device)

                y_hat_remaining = smart_teacher(x_remaining)
                y_hat_unlearning = smart_teacher(x_unlearning)

                student_forget_output = student(x_unlearning)
                student_remember_output = student(x_remaining)

                unlearning_loss = F.kl_div(
                    F.log_softmax(y_hat_unlearning, dim=1),
                    F.log_softmax(student_forget_output, dim=1), reduction='none', log_target=True
                ).sum(dim=1)
                
                total_initial_unlearning_loss += abs(unlearning_loss.mean().item())

                remembering_loss = F.kl_div(
                    F.log_softmax(y_hat_remaining, dim=1),
                    F.log_softmax(student_remember_output, dim=1), reduction='none', log_target=True
                ).sum(dim=1)
                
                cross_entropy_loss = F.cross_entropy(student_remember_output, y_remaining)
                
                total_loss = ALPHA * remembering_loss.mean().item() + GAMMA * cross_entropy_loss.item()
                total_initial_remaining_loss += total_loss

        lamda_dynamic = total_initial_unlearning_loss / (total_initial_remaining_loss + total_initial_unlearning_loss + FORGETTING_INITIAL_POWER)
        
        student.config_lr(INITIAL_LEARNING_RATE)
        optimizer = student.configure_optimizers()

        unlearning_stats = {}
        
        
        test_accuracy = 0 # initial to do the unlearning phase
        unlearning_accuracy = 100 # initial to do the unlearning phase
        
        # Unlearning process
        for unlearning_epoch in range(MAX_UNLEARNING_EPOCHS):
            
            # Adjust the scalers dynamically
            # current_unlearning_loss_scaler = BETA * unlearning_loss_scaler * (1 - unlearning_epoch / MAX_UNLEARNING_EPOCHS)
            # current_remaining_loss_scaler = BETA * remaining_loss_scaler * (unlearning_epoch / MAX_UNLEARNING_EPOCHS)
            
            # Re-shuffle the remaining data
            remaining_dloader = DataLoader(remaining_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
            
            epoch_stats = {}
            start_epoch_time = time.time()
            total_epoch_loss = 0
            for unlearning_batch, remaining_batch in tqdm.tqdm(zip(unlearning_dloader, remaining_dloader)):
                smart_teacher.eval()
                student.train()

                x_unlearning, y_unlearning = unlearning_batch
                x_remaining, y_remaining = remaining_batch

                x_unlearning, y_unlearning = x_unlearning.to(device), y_unlearning.to(device)
                x_remaining, y_remaining = x_remaining.to(device), y_remaining.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    y_hat_remaining = smart_teacher(x_remaining)
                    y_hat_unlearning = smart_teacher(x_unlearning)
                    
                y_unlearning_output = student(x_unlearning)
                y_remaining_output = student(x_remaining)

                unlearning_loss = F.kl_div(
                    F.log_softmax(y_hat_unlearning, dim=1),
                    F.log_softmax(y_unlearning_output, dim=1), reduction='none', log_target=True
                ).sum(dim=1)
                
                unlearning_loss = unlearning_loss * torch.tensor(importance_calculator.calculate_importance(unlearning_batch)).to(device)

                remembering_loss = F.kl_div(
                    F.log_softmax(y_hat_remaining, dim=1),
                    F.log_softmax(y_remaining_output, dim=1), reduction='none', log_target=True
                ).sum(dim=1)

                cross_entropy_loss = F.cross_entropy(y_remaining_output, y_remaining)
                
                loss_remember = (ALPHA * torch.clamp(remembering_loss.mean(), min=0.0) + GAMMA * cross_entropy_loss)
                loss_forget = torch.clamp(unlearning_loss.mean(), min=0.0)
                
                loss = BETA * (lamda_dynamic * loss_remember - (1 - lamda_dynamic) * loss_forget)
                
                loss.backward()
                optimizer.step()
                
                total_epoch_loss += loss.item()
                
                # if unlearning_epoch < 2 * MAX_UNLEARNING_EPOCHS /3 :
                if test_accuracy < unlearning_accuracy:
                    # Consider Forgetting with power
                    lamda_dynamic = loss_forget.item() / (loss_remember.item() + loss_forget.item() + (FORGETTING_INITIAL_POWER / (unlearning_epoch + 1)))
                else:
                    # Focus on retaining the knowledge
                    lamda_dynamic = 1
                

            end_epoch_time = time.time()

            student.eval()

            logging.info("-------------------------------------------------")
            logging.info(f"Epoch: {unlearning_epoch}")
            student_accuracy_1, student_accuracy_3, student_accuracy_5, student_precision, student_recall, student_f1 = student.test_model(unlearning_dloader)
            retrained_accuracy_1, retrained_accuracy_3, retrained_accuracy_5, retrained_precision, retrained_recall, retrained_f1 = retrained_model.test_model(unlearning_dloader)
            epoch_stats["unlearning_student_accuracy_1"] = student_accuracy_1.item()
            epoch_stats["unlearning_student_accuracy_3"] = student_accuracy_3.item()
            epoch_stats["unlearning_student_accuracy_5"] = student_accuracy_5.item()
            epoch_stats["unlearning_student_precision"] = student_precision.item()
            epoch_stats["unlearning_student_recall"] = student_recall.item()
            epoch_stats["unlearning_student_f1"] = student_f1.item()
            # epoch_stats["unlearning_retrained_accuracy_1"] = retrained_accuracy_1.item()
            # epoch_stats["unlearning_retrained_accuracy_3"] = retrained_accuracy_3.item()
            # epoch_stats["unlearning_retrained_accuracy_5"] = retrained_accuracy_5.item()
            # epoch_stats["unlearning_retrained_precision"] = retrained_precision.item()
            # epoch_stats["unlearning_retrained_recall"] = retrained_recall.item()
            # epoch_stats["unlearning_retrained_f1"] = retrained_f1.item()
            
            unlearning_accuracy = student_accuracy_1.item()
           
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
            # epoch_stats["test_retrained_accuracy_1"] = retrained_accuracy_1.item()
            # epoch_stats["test_retrained_accuracy_3"] = retrained_accuracy_3.item()
            # epoch_stats["test_retrained_accuracy_5"] = retrained_accuracy_5.item()
            # epoch_stats["test_retrained_precision"] = retrained_precision.item()
            # epoch_stats["test_retrained_recall"] = retrained_recall.item()
            # epoch_stats["test_retrained_f1"] = retrained_f1.item()
            
            test_accuracy = student_accuracy_1.item()
            
            logging.info(f"Student Test Accuracy@1: {student_accuracy_1}")
            logging.info(f"Retrained Test Accuracy@1: {retrained_accuracy_1}")
            
            epoch_stats["epoch_time"] = end_epoch_time - start_epoch_time
            
            # WandB logging
            wandb.log(epoch_stats | {"loss": total_epoch_loss} | {"lamda_dynamic": lamda_dynamic})
            
            # Save unlearning model for this epoch
            torch.save(student, f"{results_folder}/unlearned_{MODEL_NAME}_epoch_{unlearning_epoch}_batch_{batch_size}.pt")
            
            unlearning_stats[unlearning_epoch] = epoch_stats
            
        wandb.finish()
        
        json.dump(unlearning_stats, open(f"{results_folder}/unlearning_stats-batch_size_{batch_size}.json", "w"))
        logging.info(f"Unlearning models for each epoch now are saved for sample size {sample_size}, no. {i}")
