
import logging
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import evaluate

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def custom_collate_fn(batch):
    # a batch is a list of tuples (sequence, label)
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq) for seq in sequences]
    labels = [torch.tensor(label) for label in labels]

    # Pad your sequences to the same length
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(labels)

def custom_collator_transformer(batch):
    # Extract `input_ids`, `attention_mask`, and `token_type_ids`
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_masks = [torch.tensor(item["attention_mask"]) for item in batch]

    # Pad sequences to the longest in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "labels": torch.tensor([item["labels"] for item in batch])
        # "token_type_ids": token_type_ids_padded
    }
    
import evaluate
import numpy as np
import logging

def compute_metrics_bert(eval_pred):
    """
    eval_pred is (logits, labels). We'll compute several metrics:
    Accuracy, Precision, Recall, and F1-score.
    """
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # logging.info("Prediction Evaluation Metrics Initialized")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro", zero_division=1)
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"] if accuracy else None,
        "precision": precision["precision"] if precision else None,
        "recall": recall["recall"] if recall else None,
        "f1": f1["f1"] if f1 else None
    }



import numpy as np
import scipy.stats as stats

def compute_confidence_interval(values, confidence_level=0.95):
    # Calculate the mean
    values = np.array(values)
    mean = np.mean(values)
    
    # Check if all values are the same (constant data)
    if np.all(values == values[0]):
        # If all values are constant, the standard deviation is 0,
        # so the confidence interval would be the mean (because there is no variability)
        return mean, 0 # Return the mean and 0 as the confidence interval

    # Number of data points
    n = len(values)

    # Calculate the standard deviation (sample standard deviation)
    std_dev = np.std(values, ddof=1)

    # Calculate the standard error of the mean
    standard_error = std_dev / np.sqrt(n)

    # Compute the t-interval for the given confidence level
    ci_lower, ci_upper = stats.t.interval(confidence_level, n - 1, loc=mean, scale=standard_error)

    return mean , (ci_upper - mean)

def check_stopping_criteria(target_accuracy, unlearning_accuracy, delta=0.01):
    if unlearning_accuracy - target_accuracy < delta:
        return True
    return False