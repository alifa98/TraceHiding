import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.ArguemntParser import get_args
from utility.functions import check_stopping_criteria, custom_collate_fn
from utility.ImportanceCalculator import ImportanceCalculator
from utility.EntropyImportance import EntropyImportance
from utility.CoverageDiversityImportance import CoverageDiversityImportance

from torch.nn import functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
import torch
import time
import json
import tqdm
import wandb

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# os.environ["WANDB_MODE"] = "offline" # then run wandb sync <path_to_your_wandb_run_directory> to sync the results

# ------------------------------------- START CONFIGURATIONS -------------------------------------#
args = get_args()
MODEL_NAME = args.model
DATASET_NAME = args.dataset
IMPORTANCE_NAME = args.importance
SCENARIO = args.scenario
SAMPLE_SIZE =args.sampleSize
BATCH_SIZE = args.batchSize
MAX_UNLEARNING_EPOCHS = args.epochs
BIASED_SAMPLE_IMPORTANCE_NAME = args.biased # if it is None, then the sample is not biased
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

INITIAL_LEARNING_RATE = 1e-4
ALPHA = 0.9
GAMMA = 0.1

# the order is coverage, entropy, length
unified_importance_scores = {
    "HO_Geolife_Res8": [0.8, 0.1, 0.1],
    "HO_NYC_Res9": [0.1, 0.8, 0.1],
    "HO_Rome_Res8": [0.64, 0.26, 0.1],
}

# ------------------------------------- END CONFIGURATIONS -------------------------------------#


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):
    
    wandb.init(
        project="Thesis",
        job_type="unlearning",
        name=f"TraceHiding-{DATASET_NAME}-{MODEL_NAME}-{IMPORTANCE_NAME}-{SCENARIO}_unlearning-sample_size_{SAMPLE_SIZE}-repetition_{i}",
        config={
            "method_name": "TraceHiding",
            "dataset": DATASET_NAME,
            "model": MODEL_NAME,
            "scenario": SCENARIO + " deletion",
            "importance": IMPORTANCE_NAME,
            "sample_size": SAMPLE_SIZE,
            "batch_size": BATCH_SIZE,
            "is_biased": BIASED_SAMPLE_IMPORTANCE_NAME is not None,
            "bias": BIASED_SAMPLE_IMPORTANCE_NAME,
            "repetition": i,
            "learning_rate": INITIAL_LEARNING_RATE,
            "alpha": ALPHA,
            "gamma": GAMMA,
        }
    )
    
    # base folder
    base_folder = f"experiments/{DATASET_NAME}/unlearning/{SCENARIO}_sample{f"_biased_{BIASED_SAMPLE_IMPORTANCE_NAME}" if BIASED_SAMPLE_IMPORTANCE_NAME else ""}/sample_size_{SAMPLE_SIZE}/sample_{i}"
    
    # create results folder
    results_folder = f"{base_folder}/{MODEL_NAME}/trace_hiding/{IMPORTANCE_NAME}"
    os.makedirs(results_folder, exist_ok=True)
    
    unlearning_indices = torch.load(f"{base_folder}/data/unlearning.indexes.pt", weights_only=False)
    remaining_indices = torch.load(f"{base_folder}/data/remaining.indexes.pt", weights_only=False)

    train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
    test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

    # prepare importance calculator
    if IMPORTANCE_NAME == "entropy":
        importance_calculator = EntropyImportance()
    elif IMPORTANCE_NAME == "coverage_diversity":
        importance_calculator = CoverageDiversityImportance()
    elif IMPORTANCE_NAME == "unified":
        importance_calculator = UnifiedImportance([CoverageDiversityImportance(), EntropyImportance(), TrajectoryLengthImportance()], unified_importance_scores[DATASET_NAME])
    else:
        importance_calculator = ImportanceCalculator()
    importance_calculator.prepare(train_data + test_data)

    unlearning_dataset = Subset(train_data, unlearning_indices)
    remaining_dataset = Subset(train_data, remaining_indices)

    unlearning_dloader = DataLoader(unlearning_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
    test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate_fn, num_workers=24)
    
    # Load the original models stats
    original_model_stats = json.load(open(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.json", "r"))
    original_test_accuracy = original_model_stats['test_result']["accuracy_1"]

    # Load teacher and student models
    smart_teacher = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt", weights_only=False).to(device)
    student = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt", weights_only=False).to(device)

    # Perturb the student model
    # for param in student.parameters():
    #     param.data += 0.03 * torch.randn_like(param)

    smart_teacher.eval()
    student.train()
    
    optimizer = optim.Adam(student.parameters(), lr=INITIAL_LEARNING_RATE)

    unlearning_stats = {}
    
    # Unlearning process
    for unlearning_epoch in range(MAX_UNLEARNING_EPOCHS):
        
        # Re-shuffle the remaining data
        remaining_dloader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        
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
            
            batch_importance = torch.Tensor(importance_calculator.calculate_importance(unlearning_batch)).to(device)
            exp_weighted_importance = torch.exp(batch_importance) - 1

            # Weight the loss by the exponentially weighted importance
            weighted_loss = unlearning_loss * exp_weighted_importance
            
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

        epoch_stats["epoch_time"] = end_epoch_time - start_epoch_time
        
        unlearning_data_eval = student.test_model(unlearning_dloader)
        test_data_eval = student.test_model(test_dloader)
        
        unlearning_data_accuracy = unlearning_data_eval['accuracy_1']
        
        wandb.log(epoch_stats | {"loss": total_epoch_loss} | {"unlearning_" + k: v for k, v in unlearning_data_eval.items() } | {"test_" + k: v for k, v in test_data_eval.items() })
        
        # Save unlearning model for this epoch
        torch.save(student, f"{results_folder}/unlearned_{MODEL_NAME}_epoch_{unlearning_epoch}_batch_{BATCH_SIZE}.pt")
        
        unlearning_stats[unlearning_epoch] = epoch_stats
        
        # Stop Unlearning at this epoch
        if check_stopping_criteria(original_test_accuracy, unlearning_data_accuracy, delta=0):
            logging.info(f"Unlearning stopped early at epoch {unlearning_epoch} for sample size {SAMPLE_SIZE}, no. {i}")
            break
        
    else:
        logging.info(f"Unlearning finished all epochs for sample size {SAMPLE_SIZE}, no. {i}")
        
    wandb.finish()
    
    json.dump(unlearning_stats, open(f"{results_folder}/unlearning_stats-batch_size_{BATCH_SIZE}.json", "w"))
    logging.info(f"Unlearning models for each epoch now are saved for sample size {SAMPLE_SIZE}, no. {i}")
