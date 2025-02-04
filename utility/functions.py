
# Create a custom collate function to pad the sequences
import torch
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch):
    # a batch is a list of tuples (sequence, label)
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq) for seq in sequences]
    labels = [torch.tensor(label) for label in labels]

    # Pad your sequences to the same length
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(labels)



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