import os
import matplotlib.pyplot as plt
import torch

DATASET_NAME = "HO_Rome_Res8"

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

os.makedirs(f"analysis/{DATASET_NAME}/plots", exist_ok=True)

def analyze_and_visualize(dataset, dataset_name="Dataset"):
    print(f"Analyzing {dataset_name}...")

    # Sequence Length Analysis
    sequence_lengths = [len(seq) for seq, label in dataset]
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of Sequence Lengths in {dataset_name}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')

    # Save the plot
    plt.savefig(f"analysis/{DATASET_NAME}/plots/{dataset_name}_sequence_lengths.png")
    


analyze_and_visualize(train_dataset, "Train Dataset")
analyze_and_visualize(test_dataset, "Test Dataset")