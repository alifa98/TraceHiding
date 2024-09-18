from collections import Counter
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

DATASET_NAME = "Ho_Foursquare_NYC"

train_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_dataset = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)

os.makedirs(f"analysis/{DATASET_NAME}/plots/length_by_label", exist_ok=True)

def analyze_and_visualize(dataset, dataset_name="Dataset", group_size=30):
   data = {
      'Sequence Length': [len(seq) for seq, label in dataset],
      'Label': [label for seq, label in dataset]
   }
   df = pd.DataFrame(data)

   # Get label frequencies
   label_counts = Counter(df['Label'])
   sorted_labels = [label for label, count in label_counts.most_common()]

   # Group labels and plot violin plots
   num_groups = len(sorted_labels) // group_size + (1 if len(sorted_labels) % group_size else 0)

   for i in range(num_groups):
      # Select the current group of labels
      group_labels = sorted_labels[i * group_size:(i + 1) * group_size]
      group_df = df[df['Label'].isin(group_labels)]

      # Plot the violin plot for the current group
      plt.figure(figsize=(12, 8))
      sns.violinplot(x='Label', y='Sequence Length', data=group_df, inner='box', density_norm='width', palette='pastel')
      plt.title(f'Violin Plot of Sequence Lengths by Label Group {i+1} in {dataset_name}')
      plt.xlabel('Label')
      plt.ylabel('Sequence Length')
      plt.xticks(rotation=90)  # Rotate label names for better readability
      
      # Save the plot
      plt.savefig(f"analysis/{DATASET_NAME}/plots/length_by_label/{dataset_name}_sequence_length_group_{i+1}.png")


analyze_and_visualize(train_dataset, "Train Dataset")
analyze_and_visualize(test_dataset, "Test Dataset")