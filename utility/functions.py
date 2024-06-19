
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
