from collections import Counter
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np

DATASET_NAME = "HO_Rome_Res8"

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

os.makedirs(f"analysis/{DATASET_NAME}/plots", exist_ok=True)

def analyze_and_visualize(dataset, dataset_name="Dataset"):
   print(f"Analyzing {dataset_name}...")
   
   # Co-Occurrence Heatmap
   all_ids = [id_ for seq, label in dataset for id_ in seq]
   id_counts = Counter(all_ids)
   unique_ids = list(id_counts.keys())
   id_index = {id_: idx for idx, id_ in enumerate(unique_ids)}
   co_occurrence_matrix = np.zeros((len(unique_ids), len(unique_ids)))

   for seq, _ in dataset:
      for i in range(len(seq)):
         for j in range(i + 1, len(seq)):
               idx_i = id_index[seq[i]]
               idx_j = id_index[seq[j]]
               co_occurrence_matrix[idx_i, idx_j] += 1
               co_occurrence_matrix[idx_j, idx_i] += 1

   most_common_ids = [id_ for id_, _ in id_counts.most_common(10)]
   indices = [id_index[id_] for id_ in most_common_ids]
   heatmap_data = co_occurrence_matrix[np.ix_(indices, indices)]

   plt.figure(figsize=(12, 10))
   sns.heatmap(heatmap_data, xticklabels=most_common_ids, yticklabels=most_common_ids, cmap='Blues', annot=True)
   plt.title(f'Co-Occurrence Heatmap of Top 20 Most Frequent IDs in {dataset_name}')
   plt.xlabel('ID')
   plt.ylabel('ID')
   
   # Save the plot
   plt.savefig(f"analysis/{DATASET_NAME}/plots/{dataset_name}_co_occurrence_heatmap.png")


analyze_and_visualize(train_dataset, "Train Dataset")
analyze_and_visualize(test_dataset, "Test Dataset")