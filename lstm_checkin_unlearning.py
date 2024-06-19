import math
import os
import torch
import pytorch_lightning as pl
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from utility.CheckInDataset import HexagonCheckInUserDataset 
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Subset
from torch.nn import functional as F
from collections import Counter
from scipy.stats import entropy
from utility.EntropyImportance import EntropyImportance

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

#random unlearning samples size
UNLEARNING_SAMPLE_SIZE = 100


# DATASET PARAMETERS
SPLIT_RATIO = 0.8
DATASET_PATH = f"tul/TUL/dataset/nyc_HO_full.csv"

# MODEL PARAMETERS
RANDOM_STATE = 42
BATCH_SIZE = 20
MAX_UNLEARNING_EPOCH = 5

cell_to_id = torch.load("saved_models/cell_to_id.pth")

dataset = HexagonCheckInUserDataset(
    DATASET_PATH,
    user_id_col_name="user",
    trajectory_col_name="poi",
    cell_to_id=cell_to_id,
    split_ratio=SPLIT_RATIO,
    random_state=RANDOM_STATE)

# prepare importance calculator
importance_calculator = EntropyImportance()
importance_calculator.prepare(dataset)

# Create a custom collate function to pad the sequences
def custom_collate_fn(batch):
    #a batch is a list of tuples (sequence, label)
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq) for seq in sequences]
    labels = [torch.tensor(label) for label in labels]
    
    # Pad your sequences to the same length
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(labels)


# creating trainig, unlearning and testing data loaders
dataset_size = len(dataset)
dataset_shuffled_indices = torch.randperm(dataset_size).tolist()

unlearning_dataset = Subset(dataset, dataset_shuffled_indices[:UNLEARNING_SAMPLE_SIZE])
reamining_dataset = Subset(dataset, dataset_shuffled_indices[UNLEARNING_SAMPLE_SIZE:])

unlearning_dloader = DataLoader(unlearning_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
remaining_dloader = DataLoader(reamining_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True, num_workers=24)
test_dloader = DataLoader(dataset.get_test_data(), batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, num_workers=24)

# Log dataset information
logging.info(f"Number of users: {dataset.users_size}")
logging.info(f"Number of cells: {dataset.vocab_size}")
logging.info(f"Number of test samples: {len(dataset.get_test_data())}")
logging.info(f"Number of unlearning samples: {len(unlearning_dataset)}")
logging.info(f"Number of remaining samples: {len(reamining_dataset)}")


#Load initial model as the bad teacher
stupid_teacher = torch.load("saved_models/initial_model.pth")
smart_teacher = torch.load("saved_models/trained_model.pth")
retrained_model = torch.load("saved_models/initial_model.pth")
student = torch.load("saved_models/trained_model.pth")



## retraining the model with the remaining data
early_stop_callback = EarlyStopping(
   monitor='val_loss', 
   min_delta=0.00,
   patience=3,
   verbose=True,
   mode='min'
)
trainer = pl.Trainer(accelerator="gpu", devices=[
                     0], max_epochs=300, enable_progress_bar=True, callbacks=[early_stop_callback])
# trainer.fit(retrained_model, remaining_dloader, test_dloader)

smart_teacher.eval()
stupid_teacher.eval()
retrained_model.eval()
student.train()


# Unlearning process
for unlearning_epoch in range(MAX_UNLEARNING_EPOCH):
    for unlearning_batch in unlearning_dloader:
        
        x_unlearning, y_unlearning = unlearning_batch
        
        for remaining_batch in remaining_dloader:
            x_remaining, y_remaining = remaining_batch
            break
        
        # we do not need the labels be cause we are using the teacher models to generate the labels for the student
        
        student.train()
        
        #define optimizer
        optimizer = student.configure_optimizers()
        
        #set the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-6
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        y_hat_unlearning = stupid_teacher(x_unlearning)
        y_hat_remaining = smart_teacher(x_remaining)
        student_forget_output = student(x_unlearning)
        student_remember_output = student(x_remaining)
        
        unlearning_loss = F.kl_div(F.log_softmax(y_hat_unlearning, dim=1), F.log_softmax(student_forget_output, dim=1), reduction='none', log_target=True).sum(dim=1)
        remembering_loss = F.kl_div(F.log_softmax(y_hat_remaining, dim=1), F.log_softmax(student_remember_output, dim=1), reduction='none', log_target=True).sum(dim=1)
        
        
        unlearning_loss = unlearning_loss * torch.tensor(importance_calculator.calculate_importance(unlearning_batch))
        
        loss = torch.clamp(unlearning_loss.mean(), min=0.0) + torch.clamp(remembering_loss.mean(), min=0.0)

        loss.backward()
        optimizer.step()
        
        print(f"Unlearning Epoch: {unlearning_epoch}, Loss: {loss.item()}")
        

    # Test the student model
    print("="*20)
    print("Test Data Evaluation:")
    student_accuracy_1, student_accuracy_3, student_accuracy_5, student_precision, student_recall, student_f1 = student.test_model(test_dloader)
    retrained_accuracy_1, retrained_accuracy_3, retrained_accuracy_5, retrained_precision, retrained_recall, retrained_f1 = retrained_model.test_model(test_dloader)
    smart_teacher_accuracy_1, smart_teacher_accuracy_3, smart_teacher_accuracy_5, smart_teacher_precision, smart_teacher_recall, smart_teacher_f1 = smart_teacher.test_model(test_dloader)
    
    print(f"Student Model Accuracy@1: {student_accuracy_1}")
    print(f"Student Model Accuracy@3: {student_accuracy_3}")
    print(f"Student Model Accuracy@5: {student_accuracy_5}")
    print(f"Rtrained Model Accuracy@1: {retrained_accuracy_1}")
    print(f"Rtrained Model Accuracy@3: {retrained_accuracy_3}")
    print(f"Rtrained Model Accuracy@5: {retrained_accuracy_5}")
    print(f"Smart Teacher Model Accuracy@1: {smart_teacher_accuracy_1}")
    print(f"Smart Teacher Model Accuracy@3: {smart_teacher_accuracy_3}")
    print(f"Smart Teacher Model Accuracy@5: {smart_teacher_accuracy_5}")
    
    
    print("Unlearning Data Evaluation:")
    student_accuracy_1, student_accuracy_3, student_accuracy_5, student_precision, student_recall, student_f1 = student.test_model(unlearning_dloader)
    retrained_accuracy_1, retrained_accuracy_3, retrained_accuracy_5, retrained_precision, retrained_recall, retrained_f1 = retrained_model.test_model(unlearning_dloader)
    smart_teacher_accuracy_1, smart_teacher_accuracy_3, smart_teacher_accuracy_5, smart_teacher_precision, smart_teacher_recall, smart_teacher_f1 = smart_teacher.test_model(unlearning_dloader)
    
    print(f"Student Model Accuracy@1: {student_accuracy_1}")
    print(f"Student Model Accuracy@3: {student_accuracy_3}")
    print(f"Student Model Accuracy@5: {student_accuracy_5}")
    print(f"Rtrained Model Accuracy@1: {retrained_accuracy_1}")
    print(f"Rtrained Model Accuracy@3: {retrained_accuracy_3}")
    print(f"Rtrained Model Accuracy@5: {retrained_accuracy_5}")
    print(f"Smart Teacher Model Accuracy@1: {smart_teacher_accuracy_1}")
    print(f"Smart Teacher Model Accuracy@3: {smart_teacher_accuracy_3}")
    print(f"Smart Teacher Model Accuracy@5: {smart_teacher_accuracy_5}")
    
    
    print("Remaining Data Evaluation:")
    student_accuracy_1, student_accuracy_3, student_accuracy_5, student_precision, student_recall, student_f1 = student.test_model(remaining_dloader)
    retrained_accuracy_1, retrained_accuracy_3, retrained_accuracy_5, retrained_precision, retrained_recall, retrained_f1 = retrained_model.test_model(remaining_dloader)
    smart_teacher_accuracy_1, smart_teacher_accuracy_3, smart_teacher_accuracy_5, smart_teacher_precision, smart_teacher_recall, smart_teacher_f1 = smart_teacher.test_model(remaining_dloader)
    
    print(f"Student Model Accuracy@1: {student_accuracy_1}")
    print(f"Student Model Accuracy@3: {student_accuracy_3}")
    print(f"Student Model Accuracy@5: {student_accuracy_5}")
    print(f"Rtrained Model Accuracy@1: {retrained_accuracy_1}")
    print(f"Rtrained Model Accuracy@3: {retrained_accuracy_3}")
    print(f"Rtrained Model Accuracy@5: {retrained_accuracy_5}")
    print(f"Smart Teacher Model Accuracy@1: {smart_teacher_accuracy_1}")
    print(f"Smart Teacher Model Accuracy@3: {smart_teacher_accuracy_3}")
    print(f"Smart Teacher Model Accuracy@5: {smart_teacher_accuracy_5}")
    
    print("="*20)
        


    
    
    
    
    


