import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utility.functions import custom_collate_fn
from utility.EntropyImportance import EntropyImportance
from torch.nn import functional as F
from torch.utils.data import Subset
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import logging
import torch



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


for sample_size, batch_size in zip(RANDOM_SAMPLE_UNLEARNING_SIZES, UNLEARNING_BATCH_SIZE_FOR_EACH_SAMPLE_SIZE):
    for i in range(REPETITIONS_OF_EACH_SAMPLE_SIZE):

        #results folder
        os.makedirs(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/our_unlearning/{IMPORTANCE_NAME}", exist_ok=True)

        unlearning_indices = torch.load(
            f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/unlearning.indexes.pt")
        remaining_indices = torch.load(
            f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/data/remaining.indexes.pt")

        train_data = torch.load(
            f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt")
        test_data = torch.load(
            f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt")

        # prepare importance calculator
        importance_calculator = EntropyImportance()
        importance_calculator.prepare(train_data + test_data)

        # creating trainig, unlearning and testing data loaders
        unlearning_dataset = Subset(train_data, unlearning_indices)
        reamining_dataset = Subset(train_data, remaining_indices)

        unlearning_dloader = DataLoader(
            unlearning_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        remaining_dloader = DataLoader(
            reamining_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
        test_dloader = DataLoader(
            test_data, batch_size=1000, collate_fn=custom_collate_fn, num_workers=24)

        # Load initial model as the bad teacher
        stupid_teacher = torch.load(
            f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/initial_{MODEL_NAME}_model.pt")
        smart_teacher = torch.load(
            f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt")
        student = torch.load(
            f"experiments/{DATASET_NAME}/saved_models/{MODEL_NAME}/full_trained_{MODEL_NAME}_model.pt")
        
        retrained_model = torch.load(
            f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/retrained_{MODEL_NAME}_model.pt")

        smart_teacher.eval()
        stupid_teacher.eval()
        retrained_model.eval()
        student.train()

        unlearning_stats = {}
        # Unlearning process
        for unlearning_epoch in range(MAX_UNLEARNING_EPOCHS):
            epoch_stats = {}
            start_epoch_time = time.time()
            for unlearning_batch, remaining_batch in zip(unlearning_dloader, remaining_dloader):
                student.train()

                x_unlearning, y_unlearning = unlearning_batch
                x_remaining, y_remaining = remaining_batch

                # we do not need the labels be cause we are using the teacher models to generate the labels for the student

                # define optimizer
                optimizer = student.configure_optimizers()

                # set the learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 5*1e-5

                # Zero the parameter gradients
                optimizer.zero_grad()

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
                        importance_calculator.calculate_importance(unlearning_batch))

                loss = torch.clamp(unlearning_loss.mean(), min=0.0) + \
                    torch.clamp(remembering_loss.mean(), min=0.0)

                loss.backward()
                optimizer.step()

                print(f"Unlearning Epoch: {unlearning_epoch}, Loss: {loss.item()}")

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
           
            logging.info(f"Student unlearning Accuracy@1: {student_accuracy_1}")
            # logging.info(f"Student unlearning Accuracy@3: {student_accuracy_3}")
            # logging.info(f"Student unlearning Accuracy@5: {student_accuracy_5}")
            # logging.info(f"Student unlearning Precision: {student_precision}")
            # logging.info(f"Student unlearning Recall: {student_recall}")
            logging.info(f"Student unlearning F1: {student_f1}")
            logging.info(f"Retrained unlearning Accuracy@1: {retrained_accuracy_1}")
            # logging.info(f"Retrained unlearning Accuracy@3: {retrained_accuracy_3}")
            # logging.info(f"Retrained unlearning Accuracy@5: {retrained_accuracy_5}")
            # logging.info(f"Retrained unlearning Precision: {retrained_precision}")
            # logging.info(f"Retrained unlearning Recall: {retrained_recall}")
            logging.info(f"Retrained unlearning F1: {retrained_f1}")
            
           
            logging.info("++++++++++++++++++++++++++++++++++++++++++++++++")
            
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
            # logging.info(f"Student Test Accuracy@3: {student_accuracy_3}")
            # logging.info(f"Student Test Accuracy@5: {student_accuracy_5}")
            # logging.info(f"Student Test Precision: {student_precision}")
            # logging.info(f"Student Test Recall: {student_recall}")
            logging.info(f"Student Test F1: {student_f1}")
            logging.info(f"Retrained Test Accuracy@1: {retrained_accuracy_1}")
            # logging.info(f"Retrained Test Accuracy@3: {retrained_accuracy_3}")
            # logging.info(f"Retrained Test Accuracy@5: {retrained_accuracy_5}")
            # logging.info(f"Retrained Test Precision: {retrained_precision}")
            # logging.info(f"Retrained Test Recall: {retrained_recall}")
            logging.info(f"Retrained Test F1: {retrained_f1}")
            
            
            epoch_stats["unlearning_epoch_time"] = end_epoch_time - start_epoch_time
            # save unlerning model for this epoch
            torch.save(student, f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/our_unlearning/{IMPORTANCE_NAME}/unlearned_epoch{unlearning_epoch}_{MODEL_NAME}_model.pt")
            logging.info(f"Unlearning model saved for sampe size {sample_size}, no. {i}")
            
            unlearning_stats[unlearning_epoch] = epoch_stats
        json.dump(unlearning_stats, open(f"experiments/{DATASET_NAME}/unlearning/sample_size_{sample_size}/sample_{i}/{MODEL_NAME}/our_unlearning/{IMPORTANCE_NAME}/unlearning_stats-batch_size_{batch_size}.json", "w"))
            
        # break
    # break

