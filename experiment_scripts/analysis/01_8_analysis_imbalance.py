from collections import Counter
import os
import numpy as np
from scipy.stats import entropy
import torch

DATASET_NAME = "Ho_Foursquare_NYC"

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

os.makedirs(f"analysis/{DATASET_NAME}/plots", exist_ok=True)

def analyze_and_visualize(dataset, dataset_name="Dataset"):
   print(f"Analyzing {dataset_name}...")
   
   # Imbalance Analysis
   labels = [label for _, label in dataset]
   label_counts = Counter(labels)
   frequencies = np.array(list(label_counts.values()))

   # Imbalance Ratio (IR)
   max_freq = np.max(frequencies)
   min_freq = np.min(frequencies)
   imbalance_ratio = max_freq / min_freq
   print(f"Imbalance Ratio (IR): {imbalance_ratio:.2f}")

   # Gini Index
   total_samples = np.sum(frequencies)
   probs = frequencies / total_samples
   gini_index = 1 - np.sum(probs**2)
   print(f"Gini Index: {gini_index:.4f}")

   # Entropy
   ent = entropy(probs)
   max_ent = np.log(len(label_counts))  # Maximum entropy possible
   entropy_ratio = ent / max_ent
   print(f"Entropy: {ent:.4f}")
   print(f"Entropy Ratio (normalized): {entropy_ratio:.4f}")

   # Mean/Variance Ratio
   mean_freq = np.mean(frequencies)
   variance_freq = np.var(frequencies)
   mean_variance_ratio = mean_freq / variance_freq
   print(f"Mean/Variance Ratio: {mean_variance_ratio:.4f}") 


analyze_and_visualize(train_dataset, "Train Dataset")
analyze_and_visualize(test_dataset, "Test Dataset")