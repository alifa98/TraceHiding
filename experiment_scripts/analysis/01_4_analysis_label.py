from collections import Counter
import os
import matplotlib.pyplot as plt
import torch

DATASET_NAME = "Ho_Foursquare_NYC"

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

os.makedirs(f"analysis/{DATASET_NAME}/plots", exist_ok=True)

def analyze_and_visualize(dataset, dataset_name="Dataset"):
   print(f"Analyzing {dataset_name}...")
   
   # Label Distribution Analysis
   labels = [label for _, label in dataset]
   label_counts = Counter(labels)
   print(f"Label distribution in {dataset_name}:")
   for label, count in label_counts.items():
      print(f"Label {label}: {count} samples")

   # Plot the distribution
   plt.figure(figsize=(8, 5))
   plt.bar(label_counts.keys(), label_counts.values(), color='skyblue', edgecolor='black')
   plt.title(f'Label Distribution in {dataset_name}')
   plt.xlabel('Label')
   plt.ylabel('Frequency')
   
   # Save the plot
   plt.savefig(f"analysis/{DATASET_NAME}/plots/{dataset_name}_label_distribution.png")


analyze_and_visualize(train_dataset, "Train Dataset")
analyze_and_visualize(test_dataset, "Test Dataset")