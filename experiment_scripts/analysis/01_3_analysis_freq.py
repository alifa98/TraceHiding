import os
from typing import Counter
import matplotlib.pyplot as plt
import torch

DATASET_NAME = "Ho_Foursquare_NYC"

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

os.makedirs(f"analysis/{DATASET_NAME}/plots", exist_ok=True)

def analyze_and_visualize(dataset, dataset_name="Dataset"):
   print(f"Analyzing {dataset_name}...")
   
   # Frequency Analysis of Individual IDs
   all_ids = [id_ for seq, label in dataset for id_ in seq]
   id_counts = Counter(all_ids)
   most_common_ids = id_counts.most_common(20)
   ids, counts = zip(*most_common_ids)
   plt.figure(figsize=(10, 6))
   plt.bar(ids, counts, color='skyblue', edgecolor='black')
   plt.title(f'Top 20 Most Frequent IDs in {dataset_name}')
   plt.xlabel('ID')
   plt.ylabel('Frequency')
   plt.xticks(rotation=45)
   
   # Save the plot
   plt.savefig(f"analysis/{DATASET_NAME}/plots/{dataset_name}_frequency.png")


analyze_and_visualize(train_dataset, "Train Dataset")
analyze_and_visualize(test_dataset, "Test Dataset")