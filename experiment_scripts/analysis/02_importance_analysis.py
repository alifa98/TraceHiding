import os
import sys

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utility.FrequencyOfVisitImportance import FrequencyImportance
from utility.functions import custom_collate_fn
from utility.EntropyImportance import EntropyImportance
from utility.CoverageDiversityImportance import CoverageDiversityImportance
from torch.utils.data import DataLoader
import torch
import pandas as pd

DATASET_NAME = "Ho_Foursquare_NYC"

os.makedirs(f"analysis/{DATASET_NAME}/importance_analysis/", exist_ok=True)

train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)


entropy_calculator = EntropyImportance()
coverage_calculator = CoverageDiversityImportance()
frequency_calculator = FrequencyImportance()

entropy_calculator.prepare(train_data + test_data)
coverage_calculator.prepare(train_data + test_data)
frequency_calculator.prepare(train_data + test_data)


train_dloader = DataLoader(train_data, batch_size=len(train_data), collate_fn=custom_collate_fn, num_workers=24)
test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate_fn, num_workers=24)

all_entropy_importances = []
all_coverage_importances = []
all_frequency_importances = []
all_users = []

for batch in train_dloader:
    
    batch_importance = entropy_calculator.calculate_importance(batch)
    all_entropy_importances.extend(batch_importance)
    
    batch_importance = coverage_calculator.calculate_importance(batch)
    all_coverage_importances.extend(batch_importance)
    
    batch_importance = frequency_calculator.calculate_importance(batch)
    all_frequency_importances.extend(batch_importance)
    
    all_users.extend(batch[1].tolist())
    
data = pd.DataFrame({
    'entropy': all_entropy_importances,
    'coverage': all_coverage_importances,
    'frequency': all_frequency_importances,
    'user': all_users
    })



import seaborn as sns
import matplotlib.pyplot as plt

# Plotting a density plot for overall importance scores
plt.figure(figsize=(8, 6))
sns.kdeplot(data['entropy'], fill=True, label='Entropy Importance')
sns.kdeplot(data['coverage'], fill=True, label='Coverage Diversity Importance')
sns.kdeplot(data['frequency'], fill=True, label='Frequency of Visit Importance')

plt.title(f'Density Plot of importance scores')
plt.xlabel('Importance Score')
plt.ylabel('Density')
plt.legend()

plt.savefig(f"analysis/{DATASET_NAME}/importance_analysis/importances.pdf", bbox_inches='tight', format='pdf')
print(f"Saved importance plot to analysis/{DATASET_NAME}/importance_analysis/importances.pdf")


# Plotting density plots for each user (Randomly sample 10 users to visualize)
# users = data['user'].unique()
# sample_users = users[:10]
# data = data[data['user'].isin(sample_users)]
# plt.figure(figsize=(10, 6))
# sns.kdeplot(data=data, x='importance', hue='user', fill=True, common_norm=False)
# plt.title(f'Density Plot of {IMPORTANCE_NAME} importance Score by User')
# plt.xlabel('Importance Score')
# plt.ylabel('Density')

