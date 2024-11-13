import os
import sys

import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utility.FrequencyOfVisitImportance import FrequencyImportance
from utility.functions import custom_collate_fn
from utility.EntropyImportance import EntropyImportance
from utility.CoverageDiversityImportance import CoverageDiversityImportance
from utility.UserUniquenessImportance import UserUniquenessImportance
from torch.utils.data import DataLoader
import torch
import pandas as pd

DATASET_NAME = "HO_Rome_Res8"

os.makedirs(f"analysis/{DATASET_NAME}/importance_analysis/", exist_ok=True)

train_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_train.pt", weights_only=False)
test_data = torch.load(f"experiments/{DATASET_NAME}/splits/{DATASET_NAME}_test.pt", weights_only=False)


entropy_calculator = EntropyImportance()
coverage_calculator = CoverageDiversityImportance()
uniqueness_calculator = UserUniquenessImportance()
frequency_calculator = FrequencyImportance()

entropy_calculator.prepare(train_data + test_data)
coverage_calculator.prepare(train_data + test_data)
frequency_calculator.prepare(train_data + test_data)
uniqueness_calculator.prepare(train_data + test_data)


train_dloader = DataLoader(train_data, batch_size=len(train_data), collate_fn=custom_collate_fn, num_workers=24)
test_dloader = DataLoader(test_data, batch_size=len(test_data), collate_fn=custom_collate_fn, num_workers=24)

all_entropy_importances = []
all_coverage_importances = []
all_frequency_importances = []
all_uniqueness_importances = []
all_users = []

for batch in train_dloader:
    
    batch_importance = entropy_calculator.calculate_importance(batch)
    all_entropy_importances.extend(batch_importance)
    
    batch_importance = coverage_calculator.calculate_importance(batch)
    all_coverage_importances.extend(batch_importance)
    
    batch_importance = frequency_calculator.calculate_importance(batch)
    all_frequency_importances.extend(batch_importance)
    
    batch_importance = uniqueness_calculator.calculate_importance(batch)
    all_uniqueness_importances.extend(batch_importance)
    
    all_users.extend(batch[1].tolist())
    
data = pd.DataFrame({
    'entropy': all_entropy_importances,
    'coverage': all_coverage_importances,
    'frequency': all_frequency_importances,
    'user_uniqueness': all_uniqueness_importances,
    'user': all_users
    })

data_grouped = data.groupby('user').mean().reset_index()


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors# Normalize entropy values to range from 0 to 1 for gradient color mapping

cmap = plt.cm.viridis  # Choose a colormap for the gradient (e.g., viridis, plasma, etc.)


### For Grouped Data
plt.figure(figsize=(16, 9))

data_grouped['entropy_rank'] = data_grouped['entropy'].rank(ascending=False)
norm_ent = mcolors.Normalize(vmin=data_grouped['entropy'].min(), vmax=data_grouped['entropy'].max())
plt.subplot(2, 2, 1)
for rank, entropy_value in zip(data_grouped['entropy_rank'], data_grouped['entropy']):
    plt.vlines(rank, ymin=0, ymax=entropy_value, color=cmap(norm_ent(entropy_value)), linewidth=0.5)
plt.xlabel('Rank')
plt.ylabel('Entropy Importance')
plt.title('Entropy Importance Rank')
plt.grid()

data_grouped['coverage_rank'] = data_grouped['coverage'].rank(ascending=False)
norm_cov = mcolors.Normalize(vmin=data_grouped['coverage'].min(), vmax=data_grouped['coverage'].max())
plt.subplot(2, 2, 2)
for rank, coverage_value in zip(data_grouped['coverage_rank'], data_grouped['coverage']):
    plt.vlines(rank, ymin=0, ymax=coverage_value, color=cmap(norm_cov(coverage_value)), linewidth=0.5)
plt.xlabel('Rank')
plt.ylabel('Coverage Importance')
plt.title('Coverage Importance Rank')
plt.grid()

data_grouped['frequency_rank'] = data_grouped['frequency'].rank(ascending=False)
norm_freq = mcolors.Normalize(vmin=data_grouped['frequency'].min(), vmax=data_grouped['frequency'].max())
plt.subplot(2, 2, 3)
for rank, frequency_value in zip(data_grouped['frequency_rank'], data_grouped['frequency']):
    plt.vlines(rank, ymin=0, ymax=frequency_value, color=cmap(norm_freq(frequency_value)), linewidth=0.5)
plt.xlabel('Rank')
plt.ylabel('Frequency Importance')
plt.title('Frequency Importance Rank')
plt.grid()

data_grouped['user_uniqueness_rank'] = data_grouped['user_uniqueness'].rank(ascending=False)
norm_uniq = mcolors.Normalize(vmin=data_grouped['user_uniqueness'].min(), vmax=data_grouped['user_uniqueness'].max())
plt.subplot(2, 2, 4)
for rank, user_uniqueness_value in zip(data_grouped['user_uniqueness_rank'], data_grouped['user_uniqueness']):
    plt.vlines(rank, ymin=0, ymax=user_uniqueness_value, color=cmap(norm_uniq(user_uniqueness_value)), linewidth=0.5)
plt.xlabel('Rank')
plt.ylabel('User Uniqueness Importance')
plt.title('User Uniqueness Importance Rank')
plt.grid()

plt.savefig(f"analysis/{DATASET_NAME}/importance_analysis/rank_users.pdf")


# For Individual Data Points
plt.figure(figsize=(16, 9))

data['entropy_rank'] = data['entropy'].rank(ascending=False)
norm_ent = mcolors.Normalize(vmin=data['entropy'].min(), vmax=data['entropy'].max())
plt.subplot(2, 2, 1)
for rank, entropy_value in zip(data['entropy_rank'], data['entropy']):
    plt.vlines(rank, ymin=0, ymax=entropy_value, color=cmap(norm_ent(entropy_value)), linewidth=0.5)
plt.xlabel('Rank')
plt.ylabel('Entropy Importance')
plt.title('Entropy Importance Rank')
plt.grid()

data['coverage_rank'] = data['coverage'].rank(ascending=False)
norm_cov = mcolors.Normalize(vmin=data['coverage'].min(), vmax=data['coverage'].max())
plt.subplot(2, 2, 2)
for rank, coverage_value in zip(data['coverage_rank'], data['coverage']):
    plt.vlines(rank, ymin=0, ymax=coverage_value, color=cmap(norm_cov(coverage_value)), linewidth=0.5)
plt.xlabel('Rank')
plt.ylabel('Coverage Importance')
plt.title('Coverage Importance Rank')
plt.grid()

data['frequency_rank'] = data['frequency'].rank(ascending=False)
norm_freq = mcolors.Normalize(vmin=data['frequency'].min(), vmax=data['frequency'].max())
plt.subplot(2, 2, 3)
for rank, frequency_value in zip(data['frequency_rank'], data['frequency']):
    plt.vlines(rank, ymin=0, ymax=frequency_value, color=cmap(norm_freq(frequency_value)), linewidth=0.5)
plt.xlabel('Rank')
plt.ylabel('Frequency Importance')
plt.title('Frequency Importance Rank')
plt.grid()

data['user_uniqueness_rank'] = data['user_uniqueness'].rank(ascending=False)
norm_uniq = mcolors.Normalize(vmin=data['user_uniqueness'].min(), vmax=data['user_uniqueness'].max())
plt.subplot(2, 2, 4)
for rank, user_uniqueness_value in zip(data['user_uniqueness_rank'], data['user_uniqueness']):
    plt.vlines(rank, ymin=0, ymax=user_uniqueness_value, color=cmap(norm_uniq(user_uniqueness_value)), linewidth=0.5)
plt.xlabel('Rank')
plt.ylabel('User Uniqueness Importance')
plt.title('User Uniqueness Importance Rank')
plt.grid()

# save the plot in pdf
plt.savefig(f"analysis/{DATASET_NAME}/importance_analysis/rank_trajectories.pdf")