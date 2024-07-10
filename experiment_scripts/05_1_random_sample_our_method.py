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

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ------------------------------------- START CONFIGURATIONS -------------------------------------#

DATASET_NAME = "nyc_checkins"
MODEL_NAME = "LSTM"
IMPORTANCE_NAME = "entropy"
# unlearning data sampler parameters
RANDOM_SAMPLE_UNLEARNING_SIZES = [10, 20, 50, 100, 200, 300, 600, 1000]
REPETITIONS_OF_EACH_SAMPLE_SIZE = 5

UNLEARNING_BATCH_SIZE_FOR_EACH_SAMPLE_SIZE = [5, 10, 20, 50, 100, 150, 300, 500] # /2

MAX_UNLEARNING_EPOCHS = 15
UNLEANING_LOSS_SCALING = 5

# ------------------------------------- END CONFIGURATIONS -------------------------------------#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for sample_size, batch_size in zip(RANDOM_SAMPLE_UNLEARNING_SIZES, UNLEARNING_BATCH_SIZE_FOR_EACH_SAMPLE_SIZE):
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):

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

        # Load initial model as the bad teacher
        stupid_teacher = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/initial_{MODEL_NAME}_model.pt").to(device)
        smart_teacher = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt").to(device)
        student = torch.load(f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt").to(device)
        
        retrained_model = torch.load(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retrained_{MODEL_NAME}_model.pt").to(device)

        smart_teacher.eval()
        stupid_teacher.eval()
        student.train()
        
        retrained_model.eval()
        

        unlearning_stats = {}
        # Unlearning process
        for unlearning_epoch in range(MAX_UNLEARNING_EPOCHS):
            
            #re-shuffle the remaining data
            remaining_dloader = DataLoader(remaining_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
            
            epoch_stats = {}
            start_epoch_time = time.time()
            for unlearning_batch, remaining_batch in tqdm.tqdm(zip(unlearning_dloader, remaining_dloader)):
                smart_teacher.eval()
                stupid_teacher.eval()
                student.train()

                x_unlearning, y_unlearning = unlearning_batch
                x_remaining, y_remaining = remaining_batch

                x_unlearning, y_unlearning = x_unlearning.to(device), y_unlearning.to(device)
                x_remaining, y_remaining = x_remaining.to(device), y_remaining.to(device)

                # define optimizer
                optimizer = student.configure_optimizers()

                # set the learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5*1e-5

                # Zero the parameter gradients
                optimizer.zero_grad()

                with torch.no_grad():
                    y_hat_unlearning = stupid_teacher(x_unlearning)
                    y_hat_remaining = smart_teacher(x_remaining)
                    
                student_forget_output = student(x_unlearning)
                student_remember_output = student(x_remaining)

                unlearning_loss = F.kl_div(
                    F.log_softmax(y_hat_unlearning, dim=1),
                    F.log_softmax(student_forget_output, dim=1), reduction='none', log_target=True
                ).sum(dim=1)

                remembering_loss = F.kl_div(
                    F.log_softmax(y_hat_remaining, dim=1),
                    F.log_softmax(student_remember_output, dim=1), reduction='none', log_target=True
                ).sum(dim=1)

                unlearning_loss = UNLEANING_LOSS_SCALING * unlearning_loss * \
                    torch.tensor(
                        importance_calculator.calculate_importance(unlearning_batch)).to(device)

                loss = torch.clamp(unlearning_loss.mean(), min=0.0) + \
                    torch.clamp(remembering_loss.mean(), min=0.0)

                loss.backward()
                optimizer.step()

                logging.info(f"Unlearning Epoch: {unlearning_epoch}, Loss: {loss.item()}")

            end_epoch_time = time.time()

            student.eval()

            logging.info("-------------------------------------------------")
            logging.info(f"Unlearning Epoch: {unlearning_epoch}")

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
            
            epoch_stats["unlearning_epoch_time"] = end_epoch_time - start_epoch_time
            
            # save unlearning model for this epoch
            torch.save(student, f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/our_method/{IMPORTANCE_NAME}/unlearned_epoch{unlearning_epoch}_{MODEL_NAME}_model.pt")
            
            unlearning_stats[unlearning_epoch] = epoch_stats
        json.dump(unlearning_stats, open(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/our_method/{IMPORTANCE_NAME}/unlearning_stats-batch_size_{batch_size}.json", "w"))
        logging.info(f"Unlearning models for each epoch now are saved for sample size {sample_size}, no. {i}")
